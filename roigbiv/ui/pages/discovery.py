"""Centroid discovery — calibrate a FOV, find its somata, tune their boundaries.

Second in the order of operations, and the first that has an opinion about
cells. Cellpose runs on the anatomical mean image an already-completed motion
correction produced; nothing here runs motion correction, and a FOV without a
corrected stack fails fast rather than quietly correcting one first.

This page used to be two: Centroids (calibrate + run detection) and Boundaries
(tune the seeded outlines detection produces). They are one workflow on one FOV
— calibrate, detect, correct, tune — and splitting them meant switching pages
mid-correction just to see whether an edit changed the boundary it seeded.

Calibration is per-FOV because the things it sets are per-FOV facts: how big a
soma actually is in these pixels, how permissive the detector has to be, and
which checkpoint transfers to this preparation. The deployed checkpoint is
fine-tuned on cranial-window FOVs and does not transfer everywhere — on the
reference prism FOV it found 2 somata where stock cyto3 found 9.

The FOV picker is unrestricted (any FOV motion correction has touched, whether
or not centroid discovery has run yet) because calibration is meant to work
ahead of the first run. Boundary tuning is different — it recomputes a cached
flow field, which only exists once detection has written one — so that section
only appears once ``centroids.json`` exists for the selected FOV.

The preview
-----------
One OpenSeadragon+SVG viewer (``assets/discovery_sheet.js``, against
:mod:`roigbiv.ui.routes.discovery_api`), replacing two separate Plotly
``dcc.Graph`` previews. It draws three things over the mean projection:

* the effective centroids, as small circles — editable via the **Edit
  centroids** switch: drag to move, right-click to delete, click empty
  background to add. No selection concept: unlike the Tracking page's
  contact sheet, a Discovery gesture always names its own target, so there is
  nothing to select first.
* a dashed reference circle at the image centre, sized to the typed diameter
  — pan/zoom lines a real neuron up against it, a measurement rather than a
  guess at what "12 px" looks like in this preparation.
* the seeded boundary preview (read-only), when the boundary-tuning section is
  showing — Cellpose's own flow-field pixel dynamics, cached, partitioned live
  as ``capture_px`` / ``min_area`` move. See the module docstring this section
  used to live under, :mod:`roigbiv.pipeline.boundaries`, for what is actually
  being drawn and why a fallback disk is not a failure.

Editing writes through :mod:`roigbiv.ui.services.discovery_edit_ops`, a thin,
registry-free wrapper over :mod:`roigbiv.pipeline.centroid_edits` — additive
JSONL, never touching ``centroids.json`` itself. This is deliberately not the
Tracking page's ``cell_edit_ops``: that module's ``apply_gesture`` is coupled
to cross-session ``TrackedFOV`` state a single-FOV, single-session page has no
business dragging in.

Trace extraction
-----------------
A third section, below boundary tuning, appears once this FOV has a
``merged_masks.tif`` — the one label image the registry actually matches
against, produced either by a full pipeline cascade or by the Tracking
page stamping this page's saved centroids. It is deliberately NOT keyed off
the live boundary preview above it: that preview's label_ids are not yet
registry-stable. "Extract traces" runs
:func:`roigbiv.pipeline.discovery_extract.extract_from_merged_masks` via
:mod:`roigbiv.ui.services.extraction_runner` (mean always, median/mode
opt-in), producing a ``traces/`` bundle whose rows are keyed by
``local_label_id`` — the same numbers now drawn on the centroid markers
above — and by ``global_cell_id`` wherever this FOV has already gone through
Tracking. This replaces the old ad hoc ``.npy``-plus-manual-alignment
workflow with the structured bundle :mod:`roigbiv.pipeline.traces_io` already
established for the cascade's own automatic extraction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, dcc, html

from roigbiv.pipeline.calibration import load_calibration, write_calibration
from roigbiv.pipeline.centroids import clear_centroid_output
from roigbiv.ui.components import fov_select, run_panel, workspace_bar
from roigbiv.ui.components.errors import user_error
from roigbiv.ui.components.forms import HELP_TEXT, button_tooltip, help_icon
from roigbiv.ui.services import boundary_preview
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.extraction_runner import get_extraction_runner
from roigbiv.ui.services.pipeline_runner import get_pipeline_runner

RUN_ID = "roigbiv-discovery-run-btn"
FOV_SELECT_ID = "roigbiv-discovery-fov-select"
DIAMETER_ID = "roigbiv-discovery-diameter"
THRESHOLD_ID = "roigbiv-discovery-threshold"
MODEL_ID = "roigbiv-discovery-model"
SAVE_ID = "roigbiv-discovery-save"
SAVE_CLEAR_ID = "roigbiv-discovery-save-clear"
READOUT_ID = "roigbiv-discovery-readout"
FORCE_CPU_ID = "roigbiv-discovery-force-cpu"
PERSIST_FLOWS_ID = "roigbiv-discovery-persist-flows"
TICK_ID = "roigbiv-discovery-tick"

BOUNDARY_SECTION_ID = "roigbiv-discovery-boundary-section"
CAPTURE_ID = "roigbiv-discovery-capture"
MIN_AREA_ID = "roigbiv-discovery-min-area"
STATS_ID = "roigbiv-discovery-stats"
SAVE_BOUNDARY_ID = "roigbiv-discovery-save-boundary"
SAVE_ALL_BOUNDARY_ID = "roigbiv-discovery-save-boundary-all"
BOUNDARY_STATUS_ID = "roigbiv-discovery-boundary-status"

EXTRACT_SECTION_ID = "roigbiv-discovery-extract-section"
EXTRACT_STATS_ID = "roigbiv-discovery-extract-stats"
EXTRACT_BTN_ID = "roigbiv-discovery-extract-btn"
EXTRACT_STATUS_ID = "roigbiv-discovery-extract-status"

SHEET_ID = "roigbiv-discovery-sheet"
EDIT_ID = "roigbiv-discovery-edit"
EDIT_BOUNDARY_ID = "roigbiv-discovery-edit-boundary"
SHOW_CENTROIDS_ID = "roigbiv-discovery-show-centroids"
SHOW_BOUNDARIES_ID = "roigbiv-discovery-show-boundaries"
EDIT_MSG_ID = "roigbiv-discovery-edit-msg"
VIEW_ID = "roigbiv-discovery-view"
BOUNDARY_STORE_ID = "roigbiv-discovery-boundary-store"
BOUNDARY_SINK_ID = "roigbiv-discovery-boundary-sink"

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
    out_dir = fov_select.resolve_output_dir(value)
    capture, min_area = _boundary_settings_for(out_dir)
    stats, _contours = _boundary_preview_payload(out_dir, capture, min_area)
    return html.Div([
        html.H4("Calibration", className="mb-3"),
        fov_select.select(FOV_SELECT_ID, workspace, className="mb-3"),
        dbc.Card(dbc.CardBody([
            _field("Cell diameter (px)",
                   dbc.Input(id=DIAMETER_ID, type="number", min=1, step=0.5,
                             value=(calib.diameter_px if calib
                                    else DEFAULT_DIAMETER_PX)),
                   help_id=DIAMETER_ID),
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
            html.Div(_readout_text(calib, out_dir),
                     id=READOUT_ID, className="small text-muted mt-2"),
        ]), className="mb-3"),

        html.H5("Detection run", className="mb-2"),
        dbc.Switch(id=FORCE_CPU_ID, label="Force CPU", value=False),
        html.Div([
            dbc.Switch(id=PERSIST_FLOWS_ID, label="Cache the flow field",
                       value=True, className="mb-0"),
            *help_icon(PERSIST_FLOWS_ID, HELP_TEXT[PERSIST_FLOWS_ID]),
        ], className="d-flex align-items-center mb-2"),
        dbc.Button("Run centroid discovery", id=RUN_ID, color="primary",
                   className="mt-2 w-100", n_clicks=0,
                   disabled=run_panel.run_disabled()),
        button_tooltip(RUN_ID, HELP_TEXT[RUN_ID]),

        html.Div(_boundary_section(out_dir, capture, min_area, stats),
                 id=BOUNDARY_SECTION_ID,
                 style=_boundary_style(out_dir)),

        html.Div(_extraction_section(out_dir),
                 id=EXTRACT_SECTION_ID,
                 style=_extraction_style(out_dir)),
    ])


def _boundary_section(out_dir, capture, min_area, stats) -> list:
    return [
        html.Hr(),
        html.H5("Boundary tuning", className="mb-2"),
        dbc.Card(dbc.CardBody([
            html.Div([
                html.Span("capture_px", className="small me-2"),
                *help_icon(CAPTURE_ID, HELP_TEXT[CAPTURE_ID]),
            ], className="mb-1 d-flex align-items-center"),
            dcc.Slider(id=CAPTURE_ID, min=2, max=60, step=1,
                       value=capture, marks=None,
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Div([
                html.Span("min_area (px)", className="small me-2"),
                html.Span("drop anything smaller", className="text-muted small"),
            ], className="mb-1 mt-3"),
            dbc.Input(id=MIN_AREA_ID, type="number", min=0, step=10,
                      value=min_area, style={"width": "120px"}),
        ]), className="mb-3"),
        html.Div(id=STATS_ID, children=stats),
        html.Div([
            dbc.Button("Save this FOV", id=SAVE_BOUNDARY_ID, color="primary",
                       className="me-2"),
            dbc.Button("Apply to every FOV", id=SAVE_ALL_BOUNDARY_ID,
                       color="secondary", outline=True),
        ], className="mt-3"),
        html.Div(id=BOUNDARY_STATUS_ID, className="mt-2"),
    ]


def _extraction_section(out_dir: Optional[Path]) -> list:
    return [
        html.Hr(),
        html.H5("Trace extraction", className="mb-2"),
        html.Div([
            html.Span("additional stats", className="small text-muted me-2"),
            *help_icon(EXTRACT_STATS_ID, HELP_TEXT[EXTRACT_STATS_ID]),
        ], className="d-flex align-items-center mb-1"),
        dbc.Checklist(
            id=EXTRACT_STATS_ID,
            options=[{"label": "median", "value": "median"},
                     {"label": "mode", "value": "mode"}],
            value=[], inline=True, className="mb-2"),
        dbc.Button("Extract traces", id=EXTRACT_BTN_ID, color="primary",
                   outline=True, size="sm"),
        html.Div(_extraction_status(out_dir), id=EXTRACT_STATUS_ID,
                 className="mt-2"),
    ]


def _has_merged_masks(out_dir: Optional[Path]) -> bool:
    return out_dir is not None and (out_dir / "merged_masks.tif").exists()


def _extraction_style(out_dir: Optional[Path]) -> dict:
    return {} if _has_merged_masks(out_dir) else {"display": "none"}


def _extraction_status(out_dir: Optional[Path]):
    """A per-ROI table (label_id, global_cell_id) from the freshest bundle —
    the "no more manual row-to-cell alignment" payoff. ``None``/absent
    ``traces/`` reads as "not extracted yet", not an error.
    """
    if out_dir is None:
        return None
    from roigbiv.ui.services.trace_viz import list_local_rois_for_session

    rows = list_local_rois_for_session(out_dir)
    if not rows:
        return html.Div("No trace bundle for this FOV yet.",
                        className="text-muted small")
    n_linked = sum(1 for r in rows if r.get("global_cell_id"))
    header = html.Div([
        html.Span(
            f"{len(rows)} ROI(s) extracted — {n_linked} linked to a "
            f"cross-session cell. "),
        html.A("Download traces (.h5)",
               href=f"/api/discovery/{out_dir.name}/traces.h5",
               className="small"),
    ], className="small mb-1")
    table = dbc.Table(
        [html.Thead(html.Tr([html.Th("label_id"), html.Th("global_cell_id")])),
         html.Tbody([
             html.Tr([html.Td(r["local_label_id"]),
                     html.Td(r.get("global_cell_id") or "—")])
             for r in rows[:200]  # guard against a very large ROI count
         ])],
        size="sm", striped=True, className="mb-0")
    return html.Div([header, table])


def _right_column() -> html.Div:
    return html.Div([
        html.Div([
            html.H5("Preview", className="mb-0 me-3"),
            dbc.Switch(id=SHOW_CENTROIDS_ID, label="Show centroids", value=True,
                       className="mb-0 me-3"),
            dbc.Switch(id=SHOW_BOUNDARIES_ID, label="Show boundaries", value=True,
                       className="mb-0 me-3"),
            html.Span([
                dbc.Switch(id=EDIT_ID, label="Edit centroids", value=False,
                           className="mb-0"),
                *help_icon(EDIT_ID, HELP_TEXT[EDIT_ID]),
            ], className="d-flex align-items-center me-3"),
            html.Span([
                dbc.Switch(id=EDIT_BOUNDARY_ID, label="Edit boundaries",
                           value=False, className="mb-0"),
                *help_icon(EDIT_BOUNDARY_ID, HELP_TEXT[EDIT_BOUNDARY_ID]),
            ], className="d-flex align-items-center"),
        ], className="d-flex align-items-center flex-wrap mb-2"),
        # Filled by assets/discovery_sheet.js — see the module docstring.
        html.Div(id=SHEET_ID),
        html.Div(id=EDIT_MSG_ID, className="text-muted small mt-1"),
        dcc.Store(id=VIEW_ID),
        dcc.Store(id=BOUNDARY_STORE_ID),
        html.Div(id=BOUNDARY_SINK_ID, style={"display": "none"}),
        html.Hr(),
        run_panel.layout(title="Run status"),
    ])


def _field(label: str, control, help_id: Optional[str] = None) -> dbc.Row:
    label_children = [label]
    if help_id is not None:
        label_children += help_icon(help_id, HELP_TEXT[help_id])
    return dbc.Row([
        dbc.Col(html.Span(label_children, className="small"), md=6,
                className="d-flex align-items-center"),
        dbc.Col(control, md=6),
    ], className="mb-2")


# ── calibration helpers ──────────────────────────────────────────────────────


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


# ── boundary helpers ─────────────────────────────────────────────────────────


def _cfg():
    """A config carrying no boundary overrides, so the page's controls win."""
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(no_viewer=True)
    cfg.boundary_capture_px = None
    cfg.boundary_min_area = None
    return cfg


def _has_centroids(out_dir: Optional[Path]) -> bool:
    return out_dir is not None and (out_dir / "centroids.json").exists()


def _boundary_style(out_dir: Optional[Path]) -> dict:
    return {} if _has_centroids(out_dir) else {"display": "none"}


def _boundary_settings_for(out_dir: Optional[Path]) -> tuple[float, int]:
    if out_dir is None:
        return 12.0, 0
    try:
        capture, min_area = boundary_preview.default_settings(out_dir, _cfg())
    except Exception:  # noqa: BLE001 — an unreadable FOV still gets a slider
        return 12.0, 0
    return float(capture), int(min_area)


def _boundary_preview_payload(out_dir: Optional[Path], capture, min_area):
    """``(stats_children, contours_payload)`` for the boundary preview.

    Both come from one :func:`boundary_preview.preview` call — see
    ``roigbiv.ui.services.boundary_preview`` for why splitting them would
    double the per-tick cost of the slider.
    """
    if not _has_centroids(out_dir):
        return None, {"contours": {}}
    try:
        result = boundary_preview.preview(
            out_dir, _cfg(), capture_px=float(capture or 12.0),
            min_area=int(min_area or 0))
    except boundary_preview.NoFlowCache as exc:
        return (dbc.Alert(str(exc), color="warning", className="py-2 mb-0"),
                {"contours": {}})
    except Exception as exc:  # noqa: BLE001 — disk or decode, both the user's
        return user_error(exc, "computing boundaries for this FOV"), {"contours": {}}
    return _stats_card(result), _contours_payload(result)


def _contours_payload(result) -> dict:
    from roigbiv.ui.services.loaders import label_shapes

    contours = {}
    for label_id, (_centroid, rings, _area) in label_shapes(result.labels).items():
        contours[str(label_id)] = {
            "origin": result.origins.get(label_id, "flow"),
            "rings": [_ring(ys, xs) for ys, xs in rings if ys],
        }
    return {"contours": contours}


def _ring(ys, xs) -> list:
    """One contour as ``[[x, y], ...]`` — SVG order, not array order."""
    return [[round(float(x), 1), round(float(y), 1)] for y, x in zip(ys, xs)]


def _stats_card(result):
    """Seeds, fallbacks, orphan pixels and the size the disks would have been."""
    flow_areas = [a for label, a in result.areas.items()
                  if result.origins.get(label) == "flow" and a > 0]
    disk_area = np.pi * result.fallback_radius ** 2

    rows = [
        _stat("seeds", result.n_seeds),
        _stat("disk fallbacks", result.n_disk_fallback,
              warn=result.n_disk_fallback > result.n_seeds // 2),
        _stat("orphan px", result.n_orphan_basin_px),
    ]
    detail = [
        html.Div(
            f"median flow-derived area "
            f"{int(np.median(flow_areas))} px — a disk of radius "
            f"{result.fallback_radius} would be {int(disk_area)} px",
            className="small text-muted mt-2")
        if flow_areas else
        html.Div("no seed captured a flow basin at this capture_px",
                 className="small text-warning mt-2"),
    ]
    if result.n_disk_fallback > result.n_seeds // 2:
        detail.append(html.Div(
            "More than half the seeds fell back to disks. capture_px rarely "
            "fixes this — it usually means the detector never fired there.",
            className="small text-warning mt-1"))
    for warning in result.warnings:
        detail.append(html.Div(warning, className="small text-warning mt-1"))

    return dbc.Card(dbc.CardBody(
        [dbc.Row([dbc.Col(r) for r in rows], className="g-2")] + detail),
        className="mb-0")


def _stat(label: str, value: int, *, warn: bool = False) -> html.Div:
    return html.Div([
        html.Div(str(value),
                 className=("roigbiv-track-seq h5 mb-0"
                            + (" text-warning" if warn else ""))),
        html.Div(label, className="text-muted small"),
    ])


def _save_report(written: list, skipped: list, failed: list):
    if failed:
        color = "danger"
    elif written:
        color = "success"
    else:
        color = "warning"
    lines = [html.Div(f"{len(written)} FOV(s) written.")]
    if skipped:
        lines.append(html.Div(
            f"{len(skipped)} skipped — no flow cache, or a full cascade owns "
            f"the geometry: {', '.join(skipped)}",
            className="small"))
    for msg in failed:
        lines.append(html.Div(msg, className="small font-monospace"))
    return dbc.Alert(lines, color=color, className="py-2 mb-0")


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
        Output(DIAMETER_ID, "value"),
        Output(THRESHOLD_ID, "value"),
        Output(MODEL_ID, "value"),
        Output(READOUT_ID, "children"),
        Output(BOUNDARY_SECTION_ID, "style"),
        Output(CAPTURE_ID, "value"),
        Output(MIN_AREA_ID, "value"),
        Output(EXTRACT_SECTION_ID, "style"),
        Output(EXTRACT_STATUS_ID, "children"),
        Input(FOV_SELECT_ID, "value"),
        prevent_initial_call=True,
    )
    def _seed_for_fov(value):
        # Refresh calibration fields, the boundary section's visibility, and
        # its sliders for the newly selected FOV — one callback so the three
        # cannot show one FOV's numbers over another's picture.
        out_dir = fov_select.resolve_output_dir(value)
        if out_dir is None:
            return (DEFAULT_DIAMETER_PX, DEFAULT_CELLPROB_THRESHOLD, "", "",
                    {"display": "none"}, 12.0, 0, {"display": "none"}, None)
        calib = load_calibration(out_dir)
        capture, min_area = _boundary_settings_for(out_dir)
        return (
            calib.diameter_px if calib else DEFAULT_DIAMETER_PX,
            calib.cellprob_threshold if calib else DEFAULT_CELLPROB_THRESHOLD,
            (calib.cellpose_model or "") if calib else "",
            _readout_text(calib, out_dir),
            _boundary_style(out_dir),
            capture, min_area,
            _extraction_style(out_dir), _extraction_status(out_dir),
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

    @app.callback(
        Output(STATS_ID, "children"),
        Output(BOUNDARY_STORE_ID, "data"),
        Input(FOV_SELECT_ID, "value"),
        Input(CAPTURE_ID, "value"),
        Input(MIN_AREA_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_boundary_preview(value, capture, min_area):
        out_dir = fov_select.resolve_output_dir(value)
        return _boundary_preview_payload(out_dir, capture, min_area)

    @app.callback(
        Output(BOUNDARY_STATUS_ID, "children"),
        Input(SAVE_BOUNDARY_ID, "n_clicks"),
        Input(SAVE_ALL_BOUNDARY_ID, "n_clicks"),
        State(FOV_SELECT_ID, "value"),
        State(CAPTURE_ID, "value"),
        State(MIN_AREA_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_save_boundary(_n, _n_all, value, capture, min_area):
        from roigbiv.pipeline.boundaries import write_boundaries

        state = get_app_state()
        if state.workspace is None:
            return dbc.Alert("Scan a workspace first.", color="warning",
                             className="py-2 mb-0")
        if dash.ctx.triggered_id == SAVE_ALL_BOUNDARY_ID:
            targets = sorted(
                p.parent for p in
                Path(state.workspace.output_root).glob("*/centroids.json"))
        else:
            out_dir = fov_select.resolve_output_dir(value)
            if out_dir is None:
                return dbc.Alert("Select a FOV first.", color="warning",
                                 className="py-2 mb-0")
            targets = [out_dir]

        written, skipped, failed = [], [], []
        for out_dir in targets:
            try:
                drawn = write_boundaries(
                    out_dir, _cfg(), capture_px=float(capture or 12.0),
                    min_area=int(min_area or 0))
            except Exception as exc:  # noqa: BLE001 — reported per FOV below
                failed.append(f"{out_dir.name}: {type(exc).__name__}: {exc}")
                continue
            if drawn is None or not drawn.written:
                skipped.append(out_dir.name)
            else:
                written.append(out_dir.name)

        return _save_report(written, skipped, failed)

    @app.callback(
        Output(EXTRACT_STATUS_ID, "children", allow_duplicate=True),
        Input(EXTRACT_BTN_ID, "n_clicks"),
        State(FOV_SELECT_ID, "value"),
        State(EXTRACT_STATS_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_extract(_n, value, stats):
        out_dir = fov_select.resolve_output_dir(value)
        if out_dir is None:
            return dbc.Alert("Select a FOV first.", color="warning",
                             className="py-2 mb-0")
        if not _has_merged_masks(out_dir):
            return dbc.Alert(
                "No merged_masks.tif for this FOV yet — save boundaries and "
                "run Tracking first.", color="warning", className="py-2 mb-0")
        started = get_extraction_runner().start(
            out_dir, out_dir.name, tuple(stats or []))
        if not started:
            return dbc.Alert("Extraction already running for this session.",
                             color="warning", className="py-2 mb-0")
        return dbc.Alert("Extraction started…", color="info",
                         className="py-2 mb-0")

    @app.callback(
        Output(EXTRACT_STATUS_ID, "children", allow_duplicate=True),
        Output(EXTRACT_BTN_ID, "disabled"),
        Input(TICK_ID, "n_intervals"),
        State(FOV_SELECT_ID, "value"),
        prevent_initial_call=True,
    )
    def _on_extract_tick(_n, value):
        snap = get_extraction_runner().snapshot()
        if snap.active:
            return (html.Div([html.Div(line, className="small font-monospace")
                              for line in snap.logs[-5:]]), True)
        out_dir = fov_select.resolve_output_dir(value)
        if snap.error and snap.stem == (out_dir.name if out_dir else None):
            return (dbc.Alert(f"Extraction failed: {snap.error}",
                              color="danger", className="py-2 mb-0"), False)
        if out_dir is None:
            return dash.no_update, False
        return _extraction_status(out_dir), False

    _register_clientside(app)


def _register_clientside(app: dash.Dash) -> None:
    """The two handoffs to ``assets/discovery_sheet.js``.

    Both are browser-side because neither has a server-side answer: the
    viewer's state lives in an OpenSeadragon instance Dash does not own, and
    routing this through a callback would mean tearing it down on every
    slider move — exactly the zoom loss this page was built to avoid.
    """
    app.clientside_callback(
        """
        function(value, editOn, editBoundaryOn, diameter, showCentroids,
                  showBoundaries) {
            var stem = null;
            if (value && typeof value === "string"
                && value.indexOf("summary:") === 0) {
                var payload = value.slice("summary:".length);
                var parts = payload.split("/").filter(function(p) {
                    return p.length > 0;
                });
                stem = parts.length ? parts[parts.length - 1] : null;
            }
            var config = {
                stem: stem, edit_on: !!editOn,
                edit_boundary_on: !!editBoundaryOn,
                diameter_px: diameter || null,
                show_centroids: showCentroids !== false,
                show_boundaries: showBoundaries !== false,
            };
            // The first render fires as the page mounts, which can beat the
            // asset that answers it; without the retry the viewer would stay
            // blank until some unrelated control moved.
            (function attempt(tries) {
                if (window.roigbivDiscovery) {
                    window.roigbivDiscovery.render(config);
                } else if (tries > 0) {
                    setTimeout(function() { attempt(tries - 1); }, 50);
                }
            })(20);
            return config;
        }
        """,
        Output(VIEW_ID, "data"),
        Input(FOV_SELECT_ID, "value"),
        Input(EDIT_ID, "value"),
        Input(EDIT_BOUNDARY_ID, "value"),
        Input(DIAMETER_ID, "value"),
        Input(SHOW_CENTROIDS_ID, "value"),
        Input(SHOW_BOUNDARIES_ID, "value"),
    )

    # The two edit switches are mutually exclusive — a centroid drag and a
    # boundary-ring click would otherwise contend for the same pointerdown on
    # a centroid marker. Turning one on turns the other off.
    app.clientside_callback(
        """
        function(boundaryOn, centroidOn) {
            if (boundaryOn && centroidOn) { return false; }
            return window.dash_clientside.no_update;
        }
        """,
        Output(EDIT_ID, "value"),
        Input(EDIT_BOUNDARY_ID, "value"),
        State(EDIT_ID, "value"),
        prevent_initial_call=True,
    )
    app.clientside_callback(
        """
        function(centroidOn, boundaryOn) {
            if (centroidOn && boundaryOn) { return false; }
            return window.dash_clientside.no_update;
        }
        """,
        Output(EDIT_BOUNDARY_ID, "value"),
        Input(EDIT_ID, "value"),
        State(EDIT_BOUNDARY_ID, "value"),
        prevent_initial_call=True,
    )

    app.clientside_callback(
        """
        function(data) {
            if (window.roigbivDiscovery) {
                window.roigbivDiscovery.setBoundaries(data);
            }
            return "";
        }
        """,
        Output(BOUNDARY_SINK_ID, "children"),
        Input(BOUNDARY_STORE_ID, "data"),
    )
