"""Centroid detection — calibrate a FOV, then find its somata.

Second in the order of operations, and the first that has an opinion about
cells. Cellpose runs on the anatomical mean image an already-completed motion
correction produced; nothing here runs motion correction, and a FOV without a
corrected stack fails fast rather than quietly correcting one first.

Calibration is per-FOV because the things it sets are per-FOV facts: how big a
soma actually is in these pixels, how permissive the detector has to be, and
which checkpoint transfers to this preparation. The deployed checkpoint is
fine-tuned on cranial-window FOVs and does not transfer everywhere — on the
reference prism FOV it found 2 somata where stock cyto3 found 9.

Flow
----
1. Pick a FOV. Its saved calibration (if any) loads into the fields.
2. Type a diameter; a dashed circle of that size draws on the preview, so a
   real neuron can be lined up against it by pan/zoom rather than guessed.
3. Save. **Save & clear existing output** additionally drops that FOV's prior
   centroids so the next run recomputes instead of resuming stale ones.
4. **Run centroid discovery** detects across the selected FOVs and, with flow
   persistence on, caches the field the boundaries page later draws from.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from roigbiv.pipeline.calibration import load_calibration, write_calibration
from roigbiv.pipeline.centroids import clear_centroid_output
from roigbiv.ui.components import fov_select, run_panel, workspace_bar
from roigbiv.ui.components.figure import build_roi_figure
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.pipeline_runner import get_pipeline_runner

RUN_ID = "roigbiv-centroids-run-btn"
FOV_SELECT_ID = "roigbiv-centroids-fov-select"
PREVIEW_ID = "roigbiv-centroids-preview"
OVERLAY_ID = "roigbiv-centroids-overlay-toggle"
DIAMETER_ID = "roigbiv-centroids-diameter"
THRESHOLD_ID = "roigbiv-centroids-threshold"
MODEL_ID = "roigbiv-centroids-model"
SAVE_ID = "roigbiv-centroids-save"
SAVE_CLEAR_ID = "roigbiv-centroids-save-clear"
READOUT_ID = "roigbiv-centroids-readout"
FORCE_CPU_ID = "roigbiv-centroids-force-cpu"
PERSIST_FLOWS_ID = "roigbiv-centroids-persist-flows"
TICK_ID = "roigbiv-centroids-tick"

DEFAULT_DIAMETER_PX = 12          # PipelineConfig.diameter's default (types.py)
DEFAULT_CELLPROB_THRESHOLD = -2.0  # PipelineConfig.cellprob_threshold

# Model choices offered per-FOV. "" = leave cfg.cellpose_model alone (the
# deployed checkpoint). Stock cyto3 is offered because the deployed checkpoint
# is fine-tuned on cranial-window FOVs and does not transfer to every
# preparation — see the module docstring.
MODEL_OPTIONS = [
    {"label": "Deployed checkpoint (default)", "value": ""},
    {"label": "cyto3 (stock)", "value": "cyto3"},
    {"label": "cyto2 (stock)", "value": "cyto2"},
    {"label": "nuclei (stock)", "value": "nuclei"},
]


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    return html.Div([
        run_panel.tick(),
        run_panel.page_tick(TICK_ID),
        dbc.Row([
            dbc.Col(_left_column(), md=5, lg=4, className="pe-md-4"),
            dbc.Col(_right_column(), md=7, lg=8),
        ], className="g-3"),
    ])


def _left_column() -> html.Div:
    workspace = get_app_state().workspace
    _, value = fov_select.options_and_value(workspace)
    calib = _calibration_for(value)
    return html.Div([
        html.H4("Calibration", className="mb-3"),
        html.Small("Per-FOV — a measured soma size, not a global setting.",
                   className="text-muted d-block mb-3"),
        fov_select.select(FOV_SELECT_ID, workspace, className="mb-3"),
        dbc.Card(dbc.CardBody([
            _field("Cell diameter (px)",
                   dbc.Input(id=DIAMETER_ID, type="number", min=1, step=0.5,
                             value=(calib.diameter_px if calib
                                    else DEFAULT_DIAMETER_PX))),
            _field("cellprob_threshold",
                   dbc.Input(id=THRESHOLD_ID, type="number", min=-6, max=6,
                             step=0.5,
                             value=(calib.cellprob_threshold if calib
                                    else DEFAULT_CELLPROB_THRESHOLD))),
            _field("Cellpose model",
                   dbc.Select(id=MODEL_ID, options=MODEL_OPTIONS,
                              value=(calib.cellpose_model or "") if calib else "")),
            html.Div([
                dbc.Button("Save calibration", id=SAVE_ID, size="sm",
                           color="secondary", className="me-2"),
                dbc.Button("Save & clear existing output", id=SAVE_CLEAR_ID,
                           size="sm", color="warning", outline=True),
            ], className="mt-2"),
            html.Div(_readout_text(calib, fov_select.resolve_output_dir(value)),
                     id=READOUT_ID, className="small text-muted mt-2"),
        ]), className="mb-3"),

        html.H5("Detection run", className="mb-2"),
        dbc.Switch(id=FORCE_CPU_ID, label="Force CPU", value=False),
        dbc.Switch(id=PERSIST_FLOWS_ID, label="Cache the flow field",
                   value=True),
        html.Small(
            "The cached flow field is what the Boundaries page draws from. "
            "Turning it off saves ~6 MB per 512² FOV and makes seeded "
            "boundaries impossible for that FOV.",
            className="text-muted d-block mb-2"),
        dbc.Button("Run centroid discovery", id=RUN_ID, color="primary",
                   className="mt-2 w-100", n_clicks=0,
                   disabled=run_panel.run_disabled()),
        html.Small(
            "Requires an already motion-corrected stack — a pre-corrected "
            "input, or a prior motion-correction run's output. It does not "
            "run motion correction first.",
            className="text-muted d-block mt-2"),
    ])


def _right_column() -> html.Div:
    workspace = get_app_state().workspace
    _, value = fov_select.options_and_value(workspace)
    calib = _calibration_for(value)
    diameter = calib.diameter_px if calib else DEFAULT_DIAMETER_PX
    return html.Div([
        html.Div([
            html.H5("Detection preview", className="mb-0 me-3"),
            dbc.Switch(id=OVERLAY_ID, label="Show centroids", value=True,
                       className="mb-0"),
        ], className="d-flex align-items-center mb-2"),
        dcc.Graph(id=PREVIEW_ID,
                  figure=_preview_figure(value, True, diameter),
                  config={"displaylogo": False, "scrollZoom": True},
                  style={"height": "620px"}),
        html.Hr(),
        run_panel.layout(title="Run status"),
    ])


def _field(label: str, control) -> dbc.Row:
    return dbc.Row([
        dbc.Col(html.Span(label, className="small"), md=6,
                className="d-flex align-items-center"),
        dbc.Col(control, md=6),
    ], className="mb-2")


def _calibration_for(value: Optional[str]):
    out_dir = fov_select.resolve_output_dir(value)
    return load_calibration(out_dir) if out_dir is not None else None


def _readout_text(calib, out_dir: Optional[Path]) -> str:
    if out_dir is None:
        return ""
    if calib is None:
        return ("Not calibrated — enter a cell diameter (px); the dashed "
                "circle on the preview updates as you type.")
    text = (f"Calibrated: {calib.diameter_px:.1f}px diameter, "
            f"cellprob_threshold={calib.cellprob_threshold:g}, "
            f"model={calib.cellpose_model or 'deployed'}")
    if (out_dir / "centroids.json").exists():
        text += (" — this FOV already has centroid output; it will be recomputed "
                 "on the next run because the saved settings changed.")
    return text


def _preview_figure(value: Optional[str], show_centroids: bool,
                    diameter_px: Optional[float]):
    """Mean projection, optionally with discovered centroids and a size circle.

    The dashed circle is drawn at the image centre at the typed diameter —
    pan/zoom lines a real neuron up against it, which is a measurement rather
    than a guess at what "12 px" looks like in this preparation.
    """
    mean, title, output_dir = fov_select.mean_and_title(value)
    rois = []
    if show_centroids and output_dir is not None and mean is not None:
        from roigbiv.ui.services.loaders import load_centroids
        rois = load_centroids(output_dir, mean.shape)
    fig = build_roi_figure(mean, rois, show_overlay=show_centroids, title=title)
    if diameter_px and mean is not None:
        _add_calibration_circle(fig, mean.shape, diameter_px)
    return fig


def _add_calibration_circle(fig, shape: tuple[int, int], diameter_px: float) -> None:
    H, W = shape
    cy, cx = H / 2.0, W / 2.0
    r = diameter_px / 2.0
    fig.add_shape(
        type="circle",
        x0=cx - r, x1=cx + r, y0=cy - r, y1=cy + r,
        line=dict(color="#FFD400", width=2, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
    )


def centroid_overrides(force_cpu, persist_flows) -> dict:
    """Form values → ``PipelineConfig`` overrides for a centroids-only run.

    No motion-correction keys at all: ``run_centroids and not foundation_only``
    routes through ``workspace._run_centroids_only``, which never reaches the
    registration path, so passing MC tunables here would be describing work that
    does not happen.
    """
    return {
        "foundation_only": False,
        "run_centroids": True,
        "force_cpu": bool(force_cpu),
        "centroid_persist_flows": bool(persist_flows),
    }


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output(RUN_ID, "disabled"),
        Input(workspace_bar.WORKSPACE_VERSION, "data"),
        Input(TICK_ID, "n_intervals"),
        prevent_initial_call=True,
    )
    def _sync_run_button(_version, _n):
        return run_panel.run_disabled()

    @app.callback(
        Output(FOV_SELECT_ID, "options"),
        Output(FOV_SELECT_ID, "value"),
        Input(workspace_bar.WORKSPACE_VERSION, "data"),
        Input(TICK_ID, "n_intervals"),
        State(FOV_SELECT_ID, "value"),
        prevent_initial_call=True,
    )
    def _refresh_fovs(_version, _n, current):
        return fov_select.options_and_value(get_app_state().workspace, current)

    @app.callback(
        Output(run_panel.TICK_ID, "disabled", allow_duplicate=True),
        Output(run_panel.BANNER_ID, "children", allow_duplicate=True),
        Input(RUN_ID, "n_clicks"),
        State(FORCE_CPU_ID, "value"),
        State(PERSIST_FLOWS_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_run(_n: int, force_cpu, persist_flows):
        state = get_app_state()
        if state.workspace is None:
            return True, dbc.Alert("Scan a workspace first.", color="warning")
        selected = state.selected_tifs
        if selected is not None and len(selected) == 0:
            return True, dbc.Alert("Select at least one TIF to run.",
                                   color="warning")
        runner = get_pipeline_runner()
        result = runner.start(
            state.workspace, centroid_overrides(force_cpu, persist_flows),
            registry_config=state.registry_config,
            selected_tifs=workspace_bar.selected_run_paths(state.workspace,
                                                           selected),
        )
        if result == "busy":
            return False, dbc.Alert(
                "Pipeline is running for another session — try again shortly.",
                color="warning")
        if not result:
            return False, dbc.Alert(
                "A run is already active — wait for it to finish.",
                color="warning")
        return False, run_panel.render_banner(runner.snapshot())

    @app.callback(
        Output(PREVIEW_ID, "figure"),
        Input(FOV_SELECT_ID, "value"),
        Input(OVERLAY_ID, "value"),
        Input(DIAMETER_ID, "value"),
        prevent_initial_call=True,
    )
    def _render_preview(value, show_centroids, diameter):
        return _preview_figure(value, bool(show_centroids), diameter)

    @app.callback(
        Output(DIAMETER_ID, "value"),
        Output(THRESHOLD_ID, "value"),
        Output(MODEL_ID, "value"),
        Output(READOUT_ID, "children"),
        Input(FOV_SELECT_ID, "value"),
        prevent_initial_call=True,
    )
    def _seed_calibration(value):
        # Refresh the fields + readout for the newly selected FOV's own saved
        # calibration (or the default reference values if it has none yet).
        out_dir = fov_select.resolve_output_dir(value)
        if out_dir is None:
            return DEFAULT_DIAMETER_PX, DEFAULT_CELLPROB_THRESHOLD, "", ""
        calib = load_calibration(out_dir)
        return (
            calib.diameter_px if calib else DEFAULT_DIAMETER_PX,
            calib.cellprob_threshold if calib else DEFAULT_CELLPROB_THRESHOLD,
            (calib.cellpose_model or "") if calib else "",
            _readout_text(calib, out_dir),
        )

    @app.callback(
        Output(READOUT_ID, "children", allow_duplicate=True),
        Input(SAVE_ID, "n_clicks"),
        Input(SAVE_CLEAR_ID, "n_clicks"),
        State(FOV_SELECT_ID, "value"),
        State(DIAMETER_ID, "value"),
        State(THRESHOLD_ID, "value"),
        State(MODEL_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_save(_n_save, _n_save_clear, value, diameter, threshold, model):
        out_dir = fov_select.resolve_output_dir(value)
        if out_dir is None:
            return "Select a processed or pre-corrected FOV first."
        if not diameter or diameter <= 0:
            return "Enter a cell diameter (px) before saving."
        if threshold is None:
            return "Enter a cellprob threshold before saving."
        calib = write_calibration(out_dir, diameter, threshold, model or None)
        if dash.ctx.triggered_id == SAVE_CLEAR_ID:
            clear_centroid_output(out_dir, out_dir.name)
            return (f"Calibrated: {calib.diameter_px:.1f}px diameter, "
                    f"cellprob_threshold={calib.cellprob_threshold:g}, "
                    f"model={calib.cellpose_model or 'deployed'}. Cleared this "
                    "FOV's prior centroid output — the next run will recompute.")
        return _readout_text(calib, out_dir)
