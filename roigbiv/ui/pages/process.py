"""Pipeline page — scan a workspace, set motion-correction params, run.

The pipeline is currently scoped to **Foundation only**: motion correction +
SVD/L+S background separation + summary images. Stage 1–4 ROI detection,
classification, and registry matching are not run from this page (the CLI
still supports them; see ``roigbiv-pipeline --help``).

Flow
----
1. User pastes / types a path into the input field and clicks **Scan**.
2. Workspace summary card shows what was discovered (input / output /
   registry / TIF count + TIF list with validity ticks).
3. User sets ``fs`` + motion-correction tunables in the form and clicks
   **Run pipeline**.
4. Background runner streams logs; interval polls render them live.
5. Per-FOV summary rows show up under the log as they complete, including
   MC quality metrics (sharpness / banding / anisotropy / contrast) computed
   from each FOV's motion-corrected temporal mean.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from roigbiv.io import validate_tif
from roigbiv.pipeline.loaders import _maybe_read_tif
from roigbiv.pipeline.workspace import WorkspacePaths, resolve_workspace
from roigbiv.ui.components.figure import build_roi_figure
from roigbiv.ui.components.forms import HELP_TEXT, help_icon, labeled_with_help
from roigbiv.ui.components.log_stream import log_stream
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.loaders import list_motion_corrected_fovs, mc_input_mean
from roigbiv.ui.services.pipeline_runner import RunSnapshot, get_pipeline_runner


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    state = get_app_state()
    workspace = state.workspace
    # Rehydrate the run UI on refresh: the PipelineRunner persists per Flask
    # session, so a mid-run reload re-enables the interval and repaints the
    # last snapshot rather than dropping back to an empty Run-status panel.
    snap = get_pipeline_runner().snapshot()
    has_run = snap.started_at is not None
    return html.Div([
        dcc.Interval(id="roigbiv-process-tick", interval=1500,
                     disabled=not snap.active),
        # Benign sink for the TIF-selection sync callback (writes the user's
        # subset into server-side AppState); kept in the always-present layout
        # so its Output target exists before the first scan.
        dcc.Store(id="roigbiv-tif-select-sink"),
        dbc.Row([
            dbc.Col(_left_column(workspace, run_active=snap.active),
                    md=5, lg=4, className="pe-md-4"),
            dbc.Col(_right_column(snap if has_run else None),
                    md=7, lg=8),
        ], className="g-3"),
    ])


def _left_column(workspace: Optional[WorkspacePaths],
                 run_active: bool = False) -> html.Div:
    return html.Div([
        html.H4("Workspace", className="mb-3"),
        dbc.InputGroup([
            dbc.Input(
                id="roigbiv-input-path",
                placeholder="Path to a .tif file or a directory of stacks",
                value=str(workspace.input_root) if workspace else "",
                type="text",
                readonly=workspace is not None,
            ),
            dbc.Button("Scan", id="roigbiv-scan-btn", color="primary",
                       n_clicks=0, disabled=workspace is not None),
        ], className="mb-3"),
        html.Div(
            id="roigbiv-scan-result",
            children=_workspace_summary(workspace) if workspace else None,
        ),
        html.Hr(className="roigbiv-h-line"),
        html.H5("Motion-correction parameters", className="mb-2"),
        _params_form(),
        dbc.Button("Run pipeline", id="roigbiv-run-btn",
                   color="primary", className="mt-3 w-100", n_clicks=0,
                   disabled=workspace is None or run_active),
        dbc.Button("Stop run", id="roigbiv-stop-btn",
                   color="danger", outline=True, className="mt-2 w-100",
                   n_clicks=0, disabled=not run_active),
    ])


def _field_row(label: str, target_id: str, control) -> dbc.Row:
    """Label (with hover-help) + control, two columns."""
    return dbc.Row([
        dbc.Col(labeled_with_help(label, target_id, HELP_TEXT[target_id]),
                md=6, className="d-flex align-items-center"),
        dbc.Col(control, md=6),
    ], className="mb-2")


def _switch_row(switch: dbc.Switch, target_id: str):
    """A ``dbc.Switch`` with a trailing hover-help info icon."""
    return html.Div(
        [switch, *help_icon(target_id, HELP_TEXT[target_id])],
        className="d-flex align-items-center mb-1",
    )


def _stage_card(title: str, body: list) -> dbc.Card:
    return dbc.Card(dbc.CardBody([html.H6(title, className="mb-3"), *body]),
                    className="mb-3")


def _params_form() -> html.Div:
    foundation = _stage_card("Foundation", [
        _field_row("fs (Hz)", "roigbiv-param-fs",
                   dbc.Input(id="roigbiv-param-fs", type="number",
                             value=7.5, step=0.5)),
        _field_row("tau (s)", "roigbiv-param-tau",
                   dbc.Input(id="roigbiv-param-tau", type="number",
                             value=1.0, step=0.1)),
        _field_row("k_background", "roigbiv-param-k",
                   dbc.Input(id="roigbiv-param-k", type="number",
                             value=30, step=1)),
        _field_row("Motion correction", "roigbiv-param-mc-backend",
                   dbc.Select(
                       id="roigbiv-param-mc-backend",
                       options=[
                           {"label": "Row-wise non-rigid (rowwise-pcc)",
                            "value": "rowwise-pcc"},
                           {"label": "Suite2p (phasecorr)",
                            "value": "phasecorr"},
                           {"label": "Legacy SIMA HMM2D (legacy)",
                            "value": "legacy"},
                       ],
                       value="phasecorr",
                   )),
        html.Small(
            "Legacy = genuine SIMA HiddenMarkov2D from the original "
            "notebook. CPU-only and slow (~16 min/session); requires "
            "the 'sima-legacy' conda env (build once with "
            "envs/build_sima_legacy.sh).",
            id="roigbiv-param-mc-backend-help",
            className="text-muted d-block mt-1",
        ),
        _switch_row(
            dbc.Switch(id="roigbiv-param-force-cpu",
                       label="Force CPU", value=False),
            "roigbiv-param-force-cpu",
        ),
    ])
    rowwise = _stage_card("rowwise-pcc", [
        _field_row("mc_strip_height (px)", "roigbiv-param-mc-strip-height",
                   dbc.Input(id="roigbiv-param-mc-strip-height", type="number",
                             value=32, step=8, min=8, max=256)),
        _field_row("mc_max_displacement (px)", "roigbiv-param-mc-max-displacement",
                   dbc.Input(id="roigbiv-param-mc-max-displacement", type="number",
                             value=50, step=1, min=1)),
        _field_row("mc_n_template_iters", "roigbiv-param-mc-n-template-iters",
                   dbc.Input(id="roigbiv-param-mc-n-template-iters", type="number",
                             value=2, step=1, min=1)),
        _field_row("mc_subpixel_upsample", "roigbiv-param-mc-subpixel-upsample",
                   dbc.Input(id="roigbiv-param-mc-subpixel-upsample", type="number",
                             value=10, step=1, min=1)),
        _field_row("mc_frame_batch", "roigbiv-param-mc-frame-batch",
                   dbc.Input(id="roigbiv-param-mc-frame-batch", type="number",
                             value=256, step=1, min=1)),
        _field_row("mc_smooth_sigma_rows", "roigbiv-param-mc-smooth-sigma-rows",
                   dbc.Input(id="roigbiv-param-mc-smooth-sigma-rows", type="number",
                             value=6.0, step=0.5, min=0.0)),
        _field_row("mc_smooth_sigma_time", "roigbiv-param-mc-smooth-sigma-time",
                   dbc.Input(id="roigbiv-param-mc-smooth-sigma-time", type="number",
                             value=1.0, step=0.5, min=0.0)),
        _switch_row(
            dbc.Switch(id="roigbiv-param-mc-strip-confidence-weight",
                       label="Strip confidence weighting", value=True),
            "roigbiv-param-mc-strip-confidence-weight",
        ),
        _switch_row(
            dbc.Switch(id="roigbiv-param-mc-prefilter",
                       label="DoG prefilter", value=False),
            "roigbiv-param-mc-prefilter",
        ),
        _field_row("mc_prefilter_sigma_low", "roigbiv-param-mc-prefilter-sigma-low",
                   dbc.Input(id="roigbiv-param-mc-prefilter-sigma-low", type="number",
                             value=1.0, step=0.5, min=0.0)),
        _field_row("mc_prefilter_sigma_high", "roigbiv-param-mc-prefilter-sigma-high",
                   dbc.Input(id="roigbiv-param-mc-prefilter-sigma-high", type="number",
                             value=8.0, step=0.5, min=0.0)),
    ])
    legacy = _stage_card("legacy (SIMA)", [
        _field_row("mc_sima_env", "roigbiv-param-mc-sima-env",
                   dbc.Input(id="roigbiv-param-mc-sima-env", type="text",
                             value="sima-legacy")),
        _field_row("mc_granularity", "roigbiv-param-mc-granularity",
                   dbc.Select(
                       id="roigbiv-param-mc-granularity",
                       options=[{"label": "row", "value": "row"},
                                {"label": "frame", "value": "frame"}],
                       value="row",
                   )),
    ])
    phasecorr = _stage_card("phasecorr (Suite2p)", [
        _field_row("mc_s2p_block_size — h (px)", "roigbiv-param-mc-s2p-block-h",
                   dbc.Input(id="roigbiv-param-mc-s2p-block-h", type="number",
                             value=64, step=8, min=8)),
        _field_row("mc_s2p_block_size — w (px)", "roigbiv-param-mc-s2p-block-w",
                   dbc.Input(id="roigbiv-param-mc-s2p-block-w", type="number",
                             value=64, step=8, min=8)),
        _field_row("mc_s2p_smooth_sigma", "roigbiv-param-mc-s2p-smooth-sigma",
                   dbc.Input(id="roigbiv-param-mc-s2p-smooth-sigma", type="number",
                             value=1.15, step=0.05, min=0.0)),
        _field_row("mc_s2p_smooth_sigma_time", "roigbiv-param-mc-s2p-smooth-sigma-time",
                   dbc.Input(id="roigbiv-param-mc-s2p-smooth-sigma-time", type="number",
                             value=0.0, step=0.5, min=0.0)),
        _field_row("mc_s2p_maxregshift", "roigbiv-param-mc-s2p-maxregshift",
                   dbc.Input(id="roigbiv-param-mc-s2p-maxregshift", type="number",
                             value=0.1, step=0.01, min=0.0, max=1.0)),
        _switch_row(
            dbc.Switch(id="roigbiv-param-mc-s2p-nonrigid",
                       label="Non-rigid registration", value=True),
            "roigbiv-param-mc-s2p-nonrigid",
        ),
        _field_row("mc_s2p_maxregshift_nr (px)", "roigbiv-param-mc-s2p-maxregshift-nr",
                   dbc.Input(id="roigbiv-param-mc-s2p-maxregshift-nr", type="number",
                             value=5, step=1, min=1)),
        _field_row("mc_s2p_nimg_init", "roigbiv-param-mc-s2p-nimg-init",
                   dbc.Input(id="roigbiv-param-mc-s2p-nimg-init", type="number",
                             value=300, step=10, min=1)),
        _switch_row(
            dbc.Switch(id="roigbiv-param-mc-s2p-two-step-registration",
                       label="Two-step registration", value=False),
            "roigbiv-param-mc-s2p-two-step-registration",
        ),
        _switch_row(
            dbc.Switch(id="roigbiv-param-mc-s2p-one-photon-reg",
                       label="1-photon-style high-pass (1Preg)", value=True),
            "roigbiv-param-mc-s2p-one-photon-reg",
        ),
        _field_row("mc_s2p_spatial_hp_reg (px)", "roigbiv-param-mc-s2p-spatial-hp-reg",
                   dbc.Input(id="roigbiv-param-mc-s2p-spatial-hp-reg", type="number",
                             value=42, step=1, min=1)),
        _field_row("mc_s2p_pre_smooth", "roigbiv-param-mc-s2p-pre-smooth",
                   dbc.Input(id="roigbiv-param-mc-s2p-pre-smooth", type="number",
                             value=0.0, step=0.5, min=0.0)),
        _field_row("mc_s2p_spatial_taper (px)", "roigbiv-param-mc-s2p-spatial-taper",
                   dbc.Input(id="roigbiv-param-mc-s2p-spatial-taper", type="number",
                             value=40.0, step=1.0, min=0.0)),
    ])
    notifications = _stage_card("Notifications", [
        _field_row("Slack channel ID", "roigbiv-param-slack-channel",
                   dbc.Input(id="roigbiv-param-slack-channel", type="text",
                             placeholder="C0123ABCD (optional)")),
        html.Small(
            "Posts a run summary to this Slack channel when the run "
            "finishes (foundation-only runs have no ROI overlays to attach). "
            "Requires ROIGBIV_SLACK_TOKEN exported in the environment that "
            "launched roigbiv-ui. See docs/slack-notifications.md.",
            id="roigbiv-param-slack-channel-help",
            className="text-muted d-block mt-1",
        ),
    ])
    form = html.Div([foundation, rowwise, legacy, phasecorr, notifications])
    _persist_param_controls(form)
    return form


def _persist_param_controls(tree) -> None:
    """Mark every ``roigbiv-param-*`` control for browser persistence in-place.

    Walks the built form tree and turns on native Dash persistence
    (``localStorage``, constant key — these tunables are workspace-independent)
    for each parameter control, so values survive page navigation / reload
    instead of resetting to the hardcoded ``value=`` defaults. Centralized here
    so new params are covered automatically; the ``_prop_names`` guard skips the
    ``roigbiv-param-*-help`` ``html.Small`` spans that share the id prefix but
    don't support persistence.
    """
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            stack.extend(node)
            continue
        cid = getattr(node, "id", None)
        if (isinstance(cid, str) and cid.startswith("roigbiv-param-")
                and "persistence" in getattr(node, "_prop_names", ())):
            node.persistence = True
            node.persistence_type = "local"
        children = getattr(node, "children", None)
        if children is not None:
            stack.append(children)


def _mc_mean_and_title(value: Optional[str]):
    """Resolve a self-describing dropdown ``value`` to ``(mean, title)``.

    The value is self-describing (see :func:`list_motion_corrected_fovs`):

    * ``summary:{output_dir}`` — a processed FOV; read its precomputed
      ``summary/mean_M.tif``.
    * ``input:{tif_path}`` — a pre-corrected input not yet run; compute a sampled
      temporal mean on demand (:func:`mc_input_mean`).

    ``None`` / unparseable returns ``(None, None)``.
    """
    if value and ":" in value:
        kind, payload = value.split(":", 1)
        if kind == "summary":
            return (_maybe_read_tif(Path(payload) / "summary" / "mean_M.tif"),
                    Path(payload).name)
        if kind == "input":
            return (mc_input_mean(Path(payload)),
                    f"{Path(payload).stem.replace('_mc', '')} (input)")
    return None, None


def _mc_preview_figure(value: Optional[str]):
    """Render the read-only MC preview for a dropdown ``value``."""
    mean, title = _mc_mean_and_title(value)
    return build_roi_figure(mean, [], show_overlay=False, title=title)


def _mc_options_and_value(workspace, current: Optional[str] = None):
    """Build the MC dropdown ``(options, value)`` from a workspace.

    Shared by the layout seed, the run-tick refresh, and the scan handler so the
    three can't drift. Keeps ``current`` selected if it still exists, else
    defaults to the first FOV. Values are the self-describing strings from
    :func:`list_motion_corrected_fovs`.
    """
    fovs = list_motion_corrected_fovs(workspace)
    options = [{"label": label, "value": value} for label, value in fovs]
    values = {opt["value"] for opt in options}
    if current in values:
        value = current
    elif options:
        value = options[0]["value"]
    else:
        value = None
    return options, value


def _tif_options_and_values(workspace):
    """Build the TIF-selection checklist ``(options, all_values)``.

    One option per ``workspace.tifs`` entry; ``value`` is ``str(tif)`` (the
    resolved path already stored in the workspace — stable and unique, so the
    run can map a selection back to ``Path`` objects). Each label carries the
    validity tick + name + shape so the checklist doubles as the discovery
    summary. Shared by :func:`_workspace_summary` and the scan rebuild so the
    rendered options can't drift.
    """
    options: list[dict] = []
    values: list[str] = []
    if workspace is None:
        return options, values
    for tif in workspace.tifs:
        value = str(tif)
        values.append(value)
        try:
            _, shape = validate_tif(tif)
            label = html.Span([
                html.Span("OK ", className="text-success fw-bold"),
                html.Span(tif.name, className="me-2"),
                html.Span(f"{shape[0]}×{shape[1]}×{shape[2]}",
                          className="text-muted small"),
            ])
        except ValueError as exc:
            label = html.Span([
                html.Span("! ", className="text-danger fw-bold"),
                html.Span(tif.name, className="me-2"),
                html.Span(str(exc), className="text-danger small"),
            ])
        options.append({"label": label, "value": value})
    return options, values


def _sync_select_all_values(trigger, master_value, child_value, all_values):
    """Pure decision core for the Select-all ↔ checklist sync.

    Returns ``(child_value_out, master_value_out)``; either element may be
    :data:`no_update` to leave that control untouched. Breaks the master/child
    feedback loop: a master *uncheck* only clears the children when they are
    *currently* all-selected (a genuine user toggle) — not when it is the
    programmatic echo of a partial child selection that just drove the master
    to empty.
    """
    child_set = set(child_value or [])
    full = set(all_values)
    if trigger == "roigbiv-tif-select-all":
        checked = bool(master_value and "all" in master_value)
        if checked:
            return (no_update if child_set == full else list(all_values)), no_update
        if all_values and child_set == full:
            return [], no_update
        return no_update, no_update
    # Child changed → reflect all-or-not in the master, but only if it differs
    # (else the master update would re-trigger this callback needlessly).
    desired = ["all"] if (all_values and child_set == full) else []
    if list(master_value or []) == desired:
        return no_update, no_update
    return no_update, desired


def _selected_run_paths(workspace, selected):
    """Map the stored selection (set of path strings, or ``None`` = all) to the
    ordered ``Path`` subset of ``workspace.tifs`` to run."""
    return [t for t in workspace.tifs
            if selected is None or str(t) in selected]


def _live_pane(pane_id: str, title: str, subtitle: str,
               overlay: bool = False) -> dbc.Col:
    """One image pane of the live card, optionally with a valid-crop overlay."""
    inner = [
        html.Img(id=pane_id,
                 style={"width": "100%", "display": "block",
                        # Nearest-neighbour: the preview is already decimated,
                        # and browser smoothing would hide the residual motion
                        # this pane exists to reveal.
                        "imageRendering": "pixelated",
                        "background": "var(--bs-tertiary-bg)",
                        "aspectRatio": "1 / 1"}),
    ]
    if overlay:
        inner.append(html.Div(
            id=f"{pane_id}-crop",
            style={"display": "none", "position": "absolute",
                   "border": "1px dashed var(--bs-warning)",
                   "pointerEvents": "none"},
        ))
    return dbc.Col([
        html.Div([
            html.Span(title, id=f"{pane_id}-title", className="small fw-bold"),
            html.Span(subtitle, className="small text-muted ms-2"),
        ], className="mb-1"),
        html.Div(inner, style={"position": "relative"}),
    ], md=4)


def _live_mc_section() -> html.Div:
    """Live view of the FOV being motion-corrected, fed by the sidecar.

    The images refresh far faster than the run-status tick, so the hot path is a
    clientside ``fetch`` of ``/api/mc-preview/list`` that only rewrites the three
    ``<img>`` sources; the browser pulls the PNGs from Flask out of band. Nothing
    here runs Python per tick — the pipeline is a daemon thread of this same
    process during a UI run, so the GIL is worth protecting.

    The slower half (shift/confidence traces, quality metrics, valid-crop
    rectangle, scrub range) rides the existing 1.5 s run-status interval and
    reads ``state.json`` from disk directly.
    """
    return html.Div([
        dcc.Interval(id="roigbiv-mc-live-tick", interval=450, disabled=True),
        # Output sink for the clientside fetch loop, which writes the image
        # sources with set_props rather than through callback return values.
        dcc.Store(id="roigbiv-mc-live-sink"),
        html.Div([
            html.H5("Live motion correction", className="mb-0 me-3"),
            dbc.Switch(id="roigbiv-mc-live-blink", label="Blink A/B",
                       value=False, className="mb-0"),
        ], className="d-flex align-items-center mt-3 mb-1"),
        html.Div(id="roigbiv-mc-live-status",
                 className="small text-muted mb-2",
                 children="Waiting for a run…"),
        dbc.Row([
            _live_pane("roigbiv-mc-live-raw", "Raw", "before"),
            _live_pane("roigbiv-mc-live-corr", "Corrected", "after",
                       overlay=True),
            _live_pane("roigbiv-mc-live-avg", "Raw average", "cumulative mean"),
        ], className="g-2"),
        html.Div(id="roigbiv-mc-live-metrics", className="mt-2"),
        dcc.Graph(id="roigbiv-mc-live-shifts",
                  figure=_empty_shift_figure(),
                  config={"displaylogo": False, "displayModeBar": False},
                  style={"height": "220px"}),
        html.Div([
            html.Span("Scrub", className="small text-muted me-2"),
            dcc.Slider(id="roigbiv-mc-live-scrub", min=0, max=0, step=None,
                       value=0, marks=None, disabled=True,
                       tooltip={"placement": "bottom"}),
        ], id="roigbiv-mc-live-scrub-wrap", className="mt-1"),
        html.Hr(),
    ])


def _mc_preview_section() -> html.Div:
    """Read-only motion-correction preview: mean projection per FOV.

    Seeded from the active workspace at render time so a page reload (or a
    workspace with prior outputs / pre-corrected inputs) shows FOVs immediately;
    the interval tick keeps the list fresh as Foundation finishes each FOV during
    a live run, and the scan handler seeds it after an interactive scan.
    """
    state = get_app_state()
    options, value = _mc_options_and_value(state.workspace)
    # Persist the previewed FOV per workspace; False = no persistence when no
    # workspace is resolved yet (a constant key would leak across workspaces).
    mc_key = str(state.workspace.input_root) if state.workspace else False
    return html.Div([
        html.H5("Motion-correction preview", className="mb-2 mt-3"),
        dbc.Select(id="roigbiv-mc-fov-select", options=options, value=value,
                   className="mb-2",
                   persistence=mc_key, persistence_type="local"),
        dcc.Graph(id="roigbiv-mc-preview",
                  figure=_mc_preview_figure(value),
                  config={"displaylogo": False, "scrollZoom": True},
                  style={"height": "720px"}),
    ])


def _launched_config_summary(snap: Optional["RunSnapshot"]):
    """Read-only echo of the overrides that actually launched the run.

    Rendered from the runner snapshot at page-load time, so it survives
    navigation / reload and — unlike the live, persisted parameter form — never
    misrepresents an in-progress run if the user edits the form afterward.
    Returns ``None`` before any run has started.
    """
    if snap is None or snap.started_at is None or not snap.overrides:
        return None
    ov = snap.overrides

    def _item(label: str, value) -> html.Div:
        return html.Div(
            [html.Span(f"{label}: ", className="text-muted"),
             html.Span(str(value), className="fw-semibold")],
            className="small me-3 d-inline-block",
        )

    return dbc.Card(dbc.CardBody([
        html.H6("Launched config", className="mb-2"),
        html.Div([
            _item("FOVs", snap.n_fovs),
            _item("fs", ov.get("fs")),
            _item("tau", ov.get("tau")),
            _item("MC", ov.get("motion_correction_backend")),
        ]),
    ]), className="roigbiv-card-accent mb-3")


def _right_column(snap: Optional["RunSnapshot"] = None) -> html.Div:
    progress, label = _progress_for(snap)
    config_card = _launched_config_summary(snap)
    return html.Div([
        html.H4("Run status", className="mb-3"),
        *([config_card] if config_card is not None else []),
        html.Div(id="roigbiv-run-timer", className="mb-2",
                 children=(_format_timer(snap.started_at, snap.completed_at)
                           if snap else "")),
        dbc.Progress(id="roigbiv-run-progress", value=progress, label=label,
                     striped=True, className="mb-3"),
        html.Div(id="roigbiv-run-banner", children=_render_banner(snap)),
        html.Div(id="roigbiv-run-log",
                 children=log_stream(snap.logs if snap else [])),
        _live_mc_section(),
        _mc_preview_section(),
        html.Hr(),
        html.H5("Per-FOV results", className="mb-2"),
        html.Div(id="roigbiv-run-results",
                 children=_render_results(snap.results_summary if snap else [])),
    ])


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("roigbiv-scan-result", "children"),
        Output("roigbiv-run-btn", "disabled"),
        Output("roigbiv-active-registry", "children", allow_duplicate=True),
        Output("roigbiv-mc-fov-select", "options", allow_duplicate=True),
        Output("roigbiv-mc-fov-select", "value", allow_duplicate=True),
        Input("roigbiv-scan-btn", "n_clicks"),
        State("roigbiv-input-path", "value"),
        prevent_initial_call=True,
    )
    def _on_scan(_n: int, path: Optional[str]):
        state = get_app_state()
        if not path:
            return (dbc.Alert("Enter a path first.", color="warning"), True,
                    no_update, no_update, no_update)
        try:
            workspace = resolve_workspace(Path(path))
        except FileNotFoundError as exc:
            return (dbc.Alert(str(exc), color="danger"), True,
                    no_update, no_update, no_update)
        state.set_workspace(workspace)
        # Seed the MC preview immediately: the run tick is disabled while idle,
        # so without this the list would only populate on page reload or a run.
        options, value = _mc_options_and_value(workspace)
        return (
            _workspace_summary(workspace),
            False,
            f"registry: {workspace.db_path}",
            options,
            value,
        )

    @app.callback(
        Output("roigbiv-active-registry", "children", allow_duplicate=True),
        Input("roigbiv-url", "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _sync_registry_label(_pathname: str):
        state = get_app_state()
        if state.workspace is not None:
            return f"registry: {state.workspace.db_path}"
        return no_update

    @app.callback(
        Output("roigbiv-process-tick", "disabled"),
        Output("roigbiv-run-banner", "children"),
        Input("roigbiv-run-btn", "n_clicks"),
        State("roigbiv-param-fs", "value"),
        State("roigbiv-param-tau", "value"),
        State("roigbiv-param-k", "value"),
        State("roigbiv-param-mc-backend", "value"),
        State("roigbiv-param-force-cpu", "value"),
        State("roigbiv-param-mc-strip-height", "value"),
        State("roigbiv-param-mc-max-displacement", "value"),
        State("roigbiv-param-mc-n-template-iters", "value"),
        State("roigbiv-param-mc-subpixel-upsample", "value"),
        State("roigbiv-param-mc-frame-batch", "value"),
        State("roigbiv-param-mc-smooth-sigma-rows", "value"),
        State("roigbiv-param-mc-smooth-sigma-time", "value"),
        State("roigbiv-param-mc-strip-confidence-weight", "value"),
        State("roigbiv-param-mc-prefilter", "value"),
        State("roigbiv-param-mc-prefilter-sigma-low", "value"),
        State("roigbiv-param-mc-prefilter-sigma-high", "value"),
        State("roigbiv-param-mc-sima-env", "value"),
        State("roigbiv-param-mc-granularity", "value"),
        State("roigbiv-param-mc-s2p-block-h", "value"),
        State("roigbiv-param-mc-s2p-block-w", "value"),
        State("roigbiv-param-mc-s2p-smooth-sigma", "value"),
        State("roigbiv-param-mc-s2p-smooth-sigma-time", "value"),
        State("roigbiv-param-mc-s2p-maxregshift", "value"),
        State("roigbiv-param-mc-s2p-nonrigid", "value"),
        State("roigbiv-param-mc-s2p-maxregshift-nr", "value"),
        State("roigbiv-param-mc-s2p-nimg-init", "value"),
        State("roigbiv-param-mc-s2p-two-step-registration", "value"),
        State("roigbiv-param-mc-s2p-one-photon-reg", "value"),
        State("roigbiv-param-mc-s2p-spatial-hp-reg", "value"),
        State("roigbiv-param-mc-s2p-pre-smooth", "value"),
        State("roigbiv-param-mc-s2p-spatial-taper", "value"),
        State("roigbiv-param-slack-channel", "value"),
        prevent_initial_call=True,
    )
    def _on_run(_n: int, fs, tau, k, mc_backend, force_cpu,
                mc_strip_height, mc_max_displacement, mc_n_template_iters,
                mc_subpixel_upsample, mc_frame_batch, mc_smooth_sigma_rows,
                mc_smooth_sigma_time, mc_strip_confidence_weight, mc_prefilter,
                mc_prefilter_sigma_low, mc_prefilter_sigma_high, mc_sima_env,
                mc_granularity, mc_s2p_block_h, mc_s2p_block_w,
                mc_s2p_smooth_sigma, mc_s2p_smooth_sigma_time,
                mc_s2p_maxregshift, mc_s2p_nonrigid, mc_s2p_maxregshift_nr,
                mc_s2p_nimg_init, mc_s2p_two_step_registration,
                mc_s2p_one_photon_reg, mc_s2p_spatial_hp_reg,
                mc_s2p_pre_smooth, mc_s2p_spatial_taper, slack_channel):
        state = get_app_state()
        if state.workspace is None:
            return True, dbc.Alert("Scan a workspace first.", color="warning")
        selected = state.selected_tifs
        if selected is not None and len(selected) == 0:
            return True, dbc.Alert("Select at least one TIF to run.",
                                   color="warning")
        selected_paths = _selected_run_paths(state.workspace, selected)
        overrides = {
            "fs": float(fs or 7.5),
            "tau": float(tau or 1.0),
            "k_background": int(k or 30),
            "motion_correction_backend": mc_backend or "phasecorr",
            "force_cpu": bool(force_cpu),
            "foundation_only": True,
            "mc_strip_height": int(mc_strip_height) if mc_strip_height is not None else 32,
            "mc_max_displacement": int(mc_max_displacement) if mc_max_displacement is not None else 50,
            "mc_n_template_iters": int(mc_n_template_iters) if mc_n_template_iters is not None else 2,
            "mc_subpixel_upsample": int(mc_subpixel_upsample) if mc_subpixel_upsample is not None else 10,
            "mc_frame_batch": int(mc_frame_batch) if mc_frame_batch is not None else 256,
            "mc_smooth_sigma_rows": float(mc_smooth_sigma_rows) if mc_smooth_sigma_rows is not None else 6.0,
            "mc_smooth_sigma_time": float(mc_smooth_sigma_time) if mc_smooth_sigma_time is not None else 1.0,
            "mc_strip_confidence_weight": bool(mc_strip_confidence_weight),
            "mc_prefilter": bool(mc_prefilter),
            "mc_prefilter_sigma_low": float(mc_prefilter_sigma_low) if mc_prefilter_sigma_low is not None else 1.0,
            "mc_prefilter_sigma_high": float(mc_prefilter_sigma_high) if mc_prefilter_sigma_high is not None else 8.0,
            "mc_sima_env": mc_sima_env or "sima-legacy",
            "mc_granularity": mc_granularity or "row",
            "mc_s2p_block_size": [
                int(mc_s2p_block_h) if mc_s2p_block_h is not None else 64,
                int(mc_s2p_block_w) if mc_s2p_block_w is not None else 64,
            ],
            "mc_s2p_smooth_sigma": float(mc_s2p_smooth_sigma) if mc_s2p_smooth_sigma is not None else 1.15,
            "mc_s2p_smooth_sigma_time": float(mc_s2p_smooth_sigma_time) if mc_s2p_smooth_sigma_time is not None else 0.0,
            "mc_s2p_maxregshift": float(mc_s2p_maxregshift) if mc_s2p_maxregshift is not None else 0.1,
            "mc_s2p_nonrigid": bool(mc_s2p_nonrigid),
            "mc_s2p_maxregshift_nr": int(mc_s2p_maxregshift_nr) if mc_s2p_maxregshift_nr is not None else 5,
            "mc_s2p_nimg_init": int(mc_s2p_nimg_init) if mc_s2p_nimg_init is not None else 300,
            "mc_s2p_two_step_registration": bool(mc_s2p_two_step_registration),
            "mc_s2p_one_photon_reg": bool(mc_s2p_one_photon_reg),
            "mc_s2p_spatial_hp_reg": int(mc_s2p_spatial_hp_reg) if mc_s2p_spatial_hp_reg is not None else 42,
            "mc_s2p_pre_smooth": float(mc_s2p_pre_smooth) if mc_s2p_pre_smooth is not None else 0.0,
            "mc_s2p_spatial_taper": float(mc_s2p_spatial_taper) if mc_s2p_spatial_taper is not None else 40.0,
        }
        runner = get_pipeline_runner()
        slack_channel = (slack_channel or "").strip() or None
        result = runner.start(state.workspace, overrides,
                              registry_config=state.registry_config,
                              slack_channel=slack_channel,
                              selected_tifs=selected_paths)
        if result == "busy":
            return False, dbc.Alert(
                "Pipeline is running for another session — try again shortly.",
                color="warning",
            )
        if not result:
            return False, dbc.Alert(
                "A pipeline run is already active — wait for it to finish.",
                color="warning",
            )
        # Seed the banner from the fresh snapshot (current_stage = Foundation);
        # the interval tick takes over from here.
        return False, _render_banner(runner.snapshot())

    @app.callback(
        Output("roigbiv-run-log", "children"),
        Output("roigbiv-run-progress", "value"),
        Output("roigbiv-run-progress", "label"),
        Output("roigbiv-run-results", "children"),
        Output("roigbiv-process-tick", "disabled", allow_duplicate=True),
        Output("roigbiv-run-timer", "children"),
        Output("roigbiv-run-banner", "children", allow_duplicate=True),
        Output("roigbiv-run-btn", "disabled", allow_duplicate=True),
        Output("roigbiv-stop-btn", "disabled", allow_duplicate=True),
        Input("roigbiv-process-tick", "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def _on_tick(_n):
        runner = get_pipeline_runner()
        snap = runner.snapshot()
        progress, label = _progress_for(snap)
        state = get_app_state()
        return (
            log_stream(snap.logs),
            progress, label,
            _render_results(snap.results_summary),
            not snap.active,
            _format_timer(snap.started_at, snap.completed_at),
            _render_banner(snap),
            state.workspace is None or snap.active,
            # Stop is actionable only while a run is in flight (and not already
            # stopping).
            (not snap.active) or snap.stopping,
        )

    # ── live motion-correction view ───────────────────────────────────────
    # Hot path: no Python per tick. The browser polls the tiny /list endpoint
    # and rewrites the three <img> sources itself; the PNGs come from Flask out
    # of band. During a UI run the pipeline is a daemon thread of this process,
    # so keeping the 450 ms loop off the callback machinery keeps it off the GIL.
    app.clientside_callback(
        """
        function(n, blink) {
            const D = window.dash_clientside;
            if (!D || !D.set_props) { return D ? D.no_update : null; }
            fetch('/api/mc-preview/list', {cache: 'no-store'})
                .then(function(r) { return r.ok ? r.json() : []; })
                .then(function(list) {
                    if (!list || !list.length) { return; }
                    const s = list[0];
                    if (!s || s.seq === undefined || s.seq === null || s.seq < 0) {
                        return;
                    }
                    const q = 'stem=' + encodeURIComponent(s.stem)
                            + '&seq=' + s.seq;
                    const url = function(k) {
                        return '/api/mc-preview/image?' + q + '&kind=' + k;
                    };
                    // Blink mode alternates the middle pane between the raw and
                    // corrected versions of the SAME frame. Flipping one set of
                    // pixels in place is far more sensitive to residual motion
                    // than comparing two panes side by side. Both frames are
                    // already cached (immutable URLs), so the flip is instant.
                    const mid = blink ? ((n % 2) ? 'raw' : 'corr') : 'corr';
                    D.set_props('roigbiv-mc-live-raw', {src: url('raw')});
                    D.set_props('roigbiv-mc-live-corr', {src: url(mid)});
                    D.set_props('roigbiv-mc-live-corr-title', {children:
                        blink ? ('A/B · ' + (mid === 'raw' ? 'raw' : 'corrected'))
                              : 'Corrected'});
                    if (s.has_avg) {
                        D.set_props('roigbiv-mc-live-avg', {src: url('avg')});
                    }
                })
                .catch(function() { /* transient: next tick retries */ });
            return D.no_update;
        }
        """,
        Output("roigbiv-mc-live-sink", "data"),
        Input("roigbiv-mc-live-tick", "n_intervals"),
        State("roigbiv-mc-live-blink", "value"),
    )

    @app.callback(
        Output("roigbiv-mc-live-tick", "disabled"),
        Output("roigbiv-mc-live-status", "children"),
        Output("roigbiv-mc-live-shifts", "figure"),
        Output("roigbiv-mc-live-metrics", "children"),
        Output("roigbiv-mc-live-corr-crop", "style"),
        Output("roigbiv-mc-live-scrub", "min"),
        Output("roigbiv-mc-live-scrub", "max"),
        Output("roigbiv-mc-live-scrub", "marks"),
        Output("roigbiv-mc-live-scrub", "disabled"),
        Input("roigbiv-process-tick", "n_intervals"),
        State("roigbiv-theme", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _on_live_tick(_n, theme):
        # The slow half: traces, metrics, crop box and scrub range, all derived
        # from state.json read straight off disk (no HTTP hop — this is the same
        # process that wrote it). Rides the existing 1.5 s run-status interval
        # because redrawing Plotly at the image cadence would be pure jank.
        from roigbiv.ui.services.mc_preview import latest_state

        state = latest_state(get_app_state().workspace)
        records = (state or {}).get("records") or []
        terminal = (state or {}).get("phase") in _TERMINAL_LIVE_PHASES
        # Scrubbing is offered only once the run is over: while frames are
        # still arriving the fast loop owns the image sources and would fight
        # any slider position the user picked.
        can_scrub = bool(terminal and len(records) > 1)
        return (
            not _live_tick_active(state),
            _live_status_text(state),
            _shift_figure(state, theme),
            _render_live_metrics(state),
            _crop_overlay_style(state),
            min(records) if records else 0,
            max(records) if records else 0,
            {str(r): "" for r in records} if can_scrub else None,
            not can_scrub,
        )

    @app.callback(
        Output("roigbiv-mc-live-raw", "src", allow_duplicate=True),
        Output("roigbiv-mc-live-corr", "src", allow_duplicate=True),
        Output("roigbiv-mc-live-avg", "src", allow_duplicate=True),
        Input("roigbiv-mc-live-scrub", "value"),
        prevent_initial_call=True,
    )
    def _on_scrub(seq):
        # Only reachable once the run has finished (the slider is disabled
        # during a run), so this cannot race the clientside fast loop.
        from roigbiv.ui.services.mc_preview import latest_state

        state = latest_state(get_app_state().workspace)
        if state is None or seq is None:
            return no_update, no_update, no_update
        stem = state.get("stem", "")
        base = f"/api/mc-preview/image?stem={stem}&seq={int(seq)}&kind="
        return f"{base}raw", f"{base}corr", f"{base}avg"

    @app.callback(
        Output("roigbiv-run-banner", "children", allow_duplicate=True),
        Output("roigbiv-stop-btn", "disabled", allow_duplicate=True),
        Input("roigbiv-stop-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_stop(_n: int):
        # Cooperative stop: flag the in-flight run to halt at the next stage
        # boundary. Disable the button once a stop is requested; the tick
        # refreshes the banner from "Stopping…" to "Run stopped." when it ends.
        runner = get_pipeline_runner()
        requested = runner.abort()
        snap = runner.snapshot()
        return _render_banner(snap), (not requested)

    @app.callback(
        Output("roigbiv-mc-fov-select", "options", allow_duplicate=True),
        Output("roigbiv-mc-fov-select", "value", allow_duplicate=True),
        Input("roigbiv-process-tick", "n_intervals"),
        State("roigbiv-mc-fov-select", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def _refresh_mc_fovs(_n, current):
        # Cheap: enumeration only globs summary names + reads _mc filename
        # suffixes, never pixels (the heavy mean read is in _render_mc_preview,
        # on selection change). Keeps the user's current selection if it still
        # exists. Rides the same 1.5 s run-status interval.
        return _mc_options_and_value(get_app_state().workspace, current)

    @app.callback(
        Output("roigbiv-tif-select", "value", allow_duplicate=True),
        Output("roigbiv-tif-select-all", "value", allow_duplicate=True),
        Input("roigbiv-tif-select-all", "value"),
        Input("roigbiv-tif-select", "value"),
        State("roigbiv-tif-select", "options"),
        prevent_initial_call=True,
    )
    def _on_select_all(master_value, child_value, options):
        # Single combined callback (master + child as inputs) keyed on the
        # trigger id: avoids the destructive feedback loop a two-callback master/
        # child pair hits when the programmatic master update echoes back. Pure
        # logic in _sync_select_all_values.
        all_values = [opt["value"] for opt in (options or [])]
        return _sync_select_all_values(dash.ctx.triggered_id, master_value,
                                       child_value, all_values)

    @app.callback(
        Output("roigbiv-tif-select-sink", "data"),
        Input("roigbiv-tif-select", "value"),
        prevent_initial_call=False,
    )
    def _sync_selected_tifs(value):
        # Mirror the checklist selection into server-side AppState so the run
        # path (_on_run, server-side) reads it without a new callback State.
        # Fires on the *restored* value too (persistence re-mounts the checklist
        # on nav/reload), keeping AppState in step with what's displayed. Guard
        # the workspace-less initial render: the sink lives in the always-present
        # layout, so this can fire before any scan.
        if get_app_state().workspace is None:
            return no_update
        get_app_state().set_selected_tifs(value or [])
        return len(value or [])

    @app.callback(
        Output("roigbiv-mc-preview", "figure"),
        Input("roigbiv-mc-fov-select", "value"),
        prevent_initial_call=True,
    )
    def _render_mc_preview(value):
        return _mc_preview_figure(value)


# ── rendering helpers ──────────────────────────────────────────────────────


def _progress_for(snap: Optional["RunSnapshot"]) -> tuple[int, str]:
    """Progress-bar value (0–100) and ``done / total`` label from a snapshot."""
    if snap and snap.n_fovs > 0:
        done = snap.n_done + snap.n_failed
        return int(round(100 * done / snap.n_fovs)), f"{done} / {snap.n_fovs}"
    return 0, ""


def _render_banner(snap: Optional["RunSnapshot"]):
    """Live run-status banner: current stage while active, outcome when done."""
    if snap is None or snap.started_at is None:
        return None
    # Error wins over stopped: a crash on the post-stop path (e.g. backfill)
    # still sets the abort event, so guard stopped on the absence of an error
    # or the failure would be masked as a clean "Run stopped."
    if snap.error:
        return dbc.Alert("Run failed — see log below.",
                         color="danger", className="py-2 mb-2")
    if snap.stopped:
        return dbc.Alert("Run stopped.",
                         color="secondary", className="py-2 mb-2")
    if not snap.active:
        return dbc.Alert("Run complete.",
                         color="success", className="py-2 mb-2")
    if snap.stopping:
        stage = snap.current_stage or "current stage"
        return dbc.Alert(
            [html.Span("Stopping · ", className="fw-bold"),
             html.Span(f"finishing {stage}, then halting…")],
            color="warning", className="py-2 mb-2",
        )
    stage = snap.current_stage or "Pipeline run started"
    return dbc.Alert(
        [html.Span("Running · ", className="fw-bold"), html.Span(stage)],
        color="info", className="py-2 mb-2",
    )


def _workspace_summary(workspace: WorkspacePaths) -> html.Div:
    # The TIF list is a checklist: the user picks which detected stacks to run.
    # All are selected by default; the "Select all" master toggles the set. The
    # checklist lives inside roigbiv-scan-result, so a re-scan rebuilds it (and
    # AppState.set_workspace resets the stored selection to all) — no separate
    # seeding output is needed on the scan callback.
    options, values = _tif_options_and_values(workspace)
    # Persist the selection keyed to workspace identity so it survives reload
    # but never bleeds a stale selection onto a different workspace.
    ws_key = str(workspace.input_root)
    return dbc.Card(dbc.CardBody([
        html.H6("Workspace resolved", className="mb-2"),
        html.Small("Select which detected TIF stacks to run.",
                   className="text-muted d-block mb-2"),
        dbc.Checklist(
            id="roigbiv-tif-select-all",
            options=[{"label": "Select all", "value": "all"}],
            value=["all"] if values else [],
            className="fw-bold mb-1",
            persistence=ws_key, persistence_type="local",
        ),
        dbc.Checklist(
            id="roigbiv-tif-select",
            options=options,
            value=list(values),
            className="ms-3 mb-0",
            persistence=ws_key, persistence_type="local",
        ),
    ]), className="roigbiv-card-accent mt-2")


def _format_timer(
    started_at: Optional[float],
    completed_at: Optional[float],
) -> "str | html.Div":
    import time
    if started_at is None:
        return ""
    start_str = time.strftime("%H:%M:%S", time.localtime(started_at))
    end_ts = completed_at if completed_at is not None else time.time()
    elapsed_s = int(end_ts - started_at)
    h, rem = divmod(elapsed_s, 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
    return html.Div(
        [
            html.Span(f"Started: {start_str}", className="me-4"),
            html.Span(f"Elapsed: {elapsed_str}"),
        ],
        style={
            "fontFamily": "var(--roigbiv-font-mono)",
            "fontSize": "0.80rem",
            "color": "var(--roigbiv-accent)",
        },
    )


def _fmt_metric(v) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


# ── live motion-correction rendering ───────────────────────────────────────

_LIVE_PHASE_LABELS = {
    "starting": "Starting…",
    "converting": "Reading stack…",
    "building_reference": "Building reference frame…",
    "registering": "Registering",
    "done": "Registration complete",
    "skipped_precorrected": "Input already motion-corrected — nothing to correct",
    "skipped_resume": "Already registered (resumed) — registration not re-run",
    "unsupported": "No live preview for this backend",
    "degraded": "Preview writes failed — live view unavailable",
    "aborted": "Run aborted",
}


#: Phases after which no more frames will arrive — the fast image loop can stop
#: and the scrubber can take over. Mirrors
#: :data:`roigbiv.pipeline.mc_preview.TERMINAL_PHASES`.
_TERMINAL_LIVE_PHASES = frozenset({
    "done", "skipped_precorrected", "skipped_resume", "unsupported",
    "degraded", "aborted",
})


def _live_tick_active(state: Optional[dict]) -> bool:
    """Whether the fast image loop should keep polling.

    Keeps running on a stale sidecar so a briefly-wedged run resumes painting
    on its own, but stops once the phase says no further frames are coming.
    """
    if state is None:
        return False
    return (state.get("phase") or "starting") not in _TERMINAL_LIVE_PHASES


def _empty_shift_figure(theme: Optional[str] = None):
    import plotly.graph_objects as go

    from roigbiv.ui.services import theme as theme_svc

    fig = go.Figure()
    fig.update_layout(
        template=theme_svc.plotly_template(theme),
        margin=dict(l=48, r=48, t=8, b=28),
        showlegend=False,
        xaxis=dict(title="frame"),
        yaxis=dict(title="shift (px)"),
    )
    return fig


def _shift_figure(state: Optional[dict], theme: Optional[str] = None):
    """Rigid displacement and phase-correlation confidence vs frame.

    ``cmax`` shares the x axis on a secondary y: a shift trace that looks
    plausible while confidence collapses is the signature of the registration
    locking onto noise, which neither trace shows alone.
    """
    import plotly.graph_objects as go

    from roigbiv.ui.services import theme as theme_svc

    shifts = (state or {}).get("shifts") or {}
    frames = shifts.get("frame") or []
    if not frames:
        return _empty_shift_figure(theme)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frames, y=shifts.get("x") or [], name="x shift",
                             mode="lines", line=dict(width=1.4)))
    fig.add_trace(go.Scatter(x=frames, y=shifts.get("y") or [], name="y shift",
                             mode="lines", line=dict(width=1.4)))
    cmax = [c for c in (shifts.get("cmax") or []) if c is not None]
    if cmax:
        fig.add_trace(go.Scatter(
            x=frames, y=shifts.get("cmax") or [], name="confidence",
            mode="lines", yaxis="y2",
            line=dict(width=1.0, dash="dot",
                      color=theme_svc.axis_muted_color(theme))))
    fig.update_layout(
        template=theme_svc.plotly_template(theme),
        margin=dict(l=48, r=48, t=8, b=28),
        showlegend=True,
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=10)),
        xaxis=dict(title="frame"),
        yaxis=dict(title="shift (px)"),
        yaxis2=dict(title="cmax", overlaying="y", side="right",
                    showgrid=False),
    )
    return fig


def _render_live_metrics(state: Optional[dict]) -> html.Div:
    """Quality metrics for the frame currently on screen.

    Same four numbers, same formatter, as the post-run per-FOV table — they are
    computed on the full-resolution corrected frame precisely so the live and
    final readouts are comparable.
    """
    m = (state or {}).get("live_metrics") or {}
    if not m:
        return html.Div()
    pairs = [("Sharpness", "lap_var_smooth"), ("Banding", "banding_score"),
             ("Anisotropy", "grad_anisotropy_xy"), ("Contrast", "contrast_rms")]
    return html.Div(
        [html.Span([html.Span(f"{label} ", className="text-muted"),
                    html.Span(_fmt_metric(m.get(key)), className="fw-bold")],
                   className="me-3 small")
         for label, key in pairs])


def _crop_overlay_style(state: Optional[dict]) -> dict:
    """Inset box marking the region unaffected by the ``np.roll`` edge wrap.

    Suite2p shifts frames with ``np.roll``, so pixels pushed past one edge
    reappear on the opposite one. Without this outline that wrapped strip reads
    as a registration artifact rather than an expected consequence.
    """
    crop = (state or {}).get("valid_crop_frac")
    if not crop or len(crop) != 4:
        return {"display": "none"}
    x0, y0, x1, y1 = crop
    return {
        "display": "block", "position": "absolute", "pointerEvents": "none",
        "border": "1px dashed var(--bs-warning)",
        "left": f"{x0 * 100:.3f}%", "top": f"{y0 * 100:.3f}%",
        "width": f"{(x1 - x0) * 100:.3f}%", "height": f"{(y1 - y0) * 100:.3f}%",
    }


def _live_status_text(state: Optional[dict]) -> str:
    if state is None:
        return "Waiting for a run…"
    phase = state.get("phase") or "starting"
    label = _LIVE_PHASE_LABELS.get(phase, phase)
    parts = [f"{state.get('stem', '?')} · {state.get('backend', '?')}", label]
    n_total = state.get("n_total") or 0
    if phase == "registering" and n_total:
        parts.append(f"frame {state.get('n_done', 0)} / {n_total}")
    if (state.get("pass_index") or 0) > 0:
        parts.append(f"pass {int(state['pass_index']) + 1}")
    if state.get("stale") and phase not in ("done", "skipped_precorrected",
                                            "skipped_resume", "unsupported"):
        parts.append("(no recent update)")
    return " · ".join(parts)


def _render_results(summaries: list[dict]) -> html.Div:
    if not summaries:
        return html.Div(html.Em("No FOV results yet.", className="text-muted"))
    rows = []
    for s in summaries:
        status = "FAILED" if s.get("error") else "OK"
        duration = f"{s.get('duration_s', 0):.1f}s"
        m = s.get("mc_metrics") or {}
        rows.append(html.Tr([
            html.Td(status,
                    className=("text-danger fw-bold" if s.get("error")
                               else "text-success fw-bold")),
            html.Td(s.get("stem")),
            html.Td(duration),
            html.Td(_fmt_metric(m.get("lap_var_smooth"))),
            html.Td(_fmt_metric(m.get("banding_score"))),
            html.Td(_fmt_metric(m.get("grad_anisotropy_xy"))),
            html.Td(_fmt_metric(m.get("contrast_rms"))),
        ]))
    return dbc.Table(
        [html.Thead(html.Tr([
            html.Th(""), html.Th("FOV"), html.Th("Duration"),
            html.Th("Sharpness"), html.Th("Banding"),
            html.Th("Anisotropy"), html.Th("Contrast"),
        ])), html.Tbody(rows)],
        size="sm", striped=True, borderless=False,
        className="mb-0",
    )
