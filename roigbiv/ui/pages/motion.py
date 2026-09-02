"""Motion correction — parameters, a live view of the registration, and metrics.

The first operation in the pipeline and, on wall-clock, most of it. This page
owns that and nothing else: no centroid calibration, no detection settings, no
workspace scanner. Those moved to :mod:`roigbiv.ui.pages.discovery` and
:mod:`roigbiv.ui.components.workspace_bar` respectively, because a page that
also held them made "did motion correction finish" and "did detection work" one
question with one Run button and one shared results table.

Flow
----
1. Scan a workspace from the navbar (any page can).
2. Set ``fs`` + the motion-correction tunables; each backend has its own card.
3. **Run motion correction** launches Foundation only — motion correction, SVD
   / L+S background separation, summary images.
4. The live view repaints the frame being registered while the shared run panel
   (:mod:`roigbiv.ui.components.run_panel`) streams logs and per-FOV metrics.
5. The preview shows any FOV's mean projection once its Foundation is done.
"""
from __future__ import annotations

from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from roigbiv.ui.components import fov_select, run_panel, sidebar, workspace_bar
from roigbiv.ui.components.figure import build_roi_figure
from roigbiv.ui.components.forms import (
    HELP_TEXT, button_tooltip, help_icon, labeled_with_help,
)
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.pipeline_runner import get_pipeline_runner

RUN_ID = "roigbiv-motion-run-btn"
FOV_SELECT_ID = "roigbiv-mc-fov-select"
PREVIEW_ID = "roigbiv-mc-preview"
TICK_ID = "roigbiv-motion-tick"

SIDEBAR_TOGGLE_ID = "roigbiv-motion-sidebar-toggle"
SIDEBAR_STORE_ID = "roigbiv-motion-sidebar-store"
PARAMS_COLLAPSE_ID = "roigbiv-motion-params-collapse"
LEFT_COL_ID = "roigbiv-motion-left-col"
RIGHT_COL_ID = "roigbiv-motion-right-col"

# Left column width classes. Default is collapsed, so the layout is built
# closed and the clientside toggle (sidebar.register_collapsible_toggle)
# swaps to the open classes on expand. Closed is a bare icon rail — the Run
# button lives in the right column now, so nothing left column-side needs
# more than the toggle chevron's own width.
_LEFT_OPEN_CLASS = "col-md-5 col-lg-4 pe-md-4"
_LEFT_CLOSED_CLASS = "col-auto pe-md-2"
_RIGHT_OPEN_CLASS = "col-md-7 col-lg-8"
_RIGHT_CLOSED_CLASS = "col"


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    return html.Div([
        run_panel.tick(),
        run_panel.page_tick(TICK_ID),
        dbc.Row([
            dbc.Col(_left_column(), id=LEFT_COL_ID,
                    className=_LEFT_CLOSED_CLASS),
            dbc.Col(_right_column(), id=RIGHT_COL_ID,
                    className=_RIGHT_CLOSED_CLASS),
        ], className="g-3"),
    ])


def _left_column() -> html.Div:
    return html.Div([
        sidebar.sidebar_toggle(toggle_id=SIDEBAR_TOGGLE_ID,
                               store_id=SIDEBAR_STORE_ID,
                               default_open=False),
        dbc.Collapse(html.Div([
            html.H4("Motion-correction parameters", className="mb-3"),
            _params_form(),
        ]), id=PARAMS_COLLAPSE_ID, is_open=False),
    ])


def _right_column() -> html.Div:
    return html.Div([
        html.Div([
            dbc.Button("Run motion correction", id=RUN_ID,
                       color="primary", n_clicks=0,
                       disabled=run_panel.run_disabled()),
            button_tooltip(RUN_ID, HELP_TEXT[RUN_ID]),
        ], className="mb-3"),
        run_panel.layout(),
        _live_mc_section(),
        _mc_preview_section(),
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
    a live run.
    """
    workspace = get_app_state().workspace
    _, value = fov_select.options_and_value(workspace)
    return html.Div([
        html.H5("Motion-correction preview", className="mb-2 mt-3"),
        fov_select.select(FOV_SELECT_ID, workspace, className="mb-2"),
        dcc.Graph(id=PREVIEW_ID,
                  figure=_preview_figure(value),
                  config={"displaylogo": False, "scrollZoom": True},
                  style={"height": "720px"}),
    ])


def _preview_figure(value: Optional[str]):
    """The mean projection for a dropdown ``value``, with no overlay.

    Deliberately bare: what this page is for is judging whether registration
    worked, and an ROI overlay on top of it answers a different question.
    """
    mean, title, _ = fov_select.mean_and_title(value)
    return build_roi_figure(mean, [], show_overlay=False, title=title)


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    sidebar.register_collapsible_toggle(
        app, toggle_id=SIDEBAR_TOGGLE_ID, store_id=SIDEBAR_STORE_ID,
        collapse_id=PARAMS_COLLAPSE_ID,
        left_col_id=LEFT_COL_ID, right_col_id=RIGHT_COL_ID,
        left_open_class=_LEFT_OPEN_CLASS, left_closed_class=_LEFT_CLOSED_CLASS,
        right_open_class=_RIGHT_OPEN_CLASS, right_closed_class=_RIGHT_CLOSED_CLASS,
    )

    @app.callback(
        Output(RUN_ID, "disabled"),
        Input(workspace_bar.WORKSPACE_VERSION, "data"),
        Input(TICK_ID, "n_intervals"),
        prevent_initial_call=True,
    )
    def _sync_run_button(_version, _n):
        # Two reasons the button changes state, one place that decides: a fresh
        # scan (there is now something to run) and the runner going busy or idle
        # (possibly from the Centroids page).
        return run_panel.run_disabled()

    @app.callback(
        Output(run_panel.TICK_ID, "disabled", allow_duplicate=True),
        Output(run_panel.BANNER_ID, "children", allow_duplicate=True),
        Input(RUN_ID, "n_clicks"),
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
        overrides = motion_overrides(
            fs, tau, k, mc_backend, force_cpu, mc_strip_height,
            mc_max_displacement, mc_n_template_iters, mc_subpixel_upsample,
            mc_frame_batch, mc_smooth_sigma_rows, mc_smooth_sigma_time,
            mc_strip_confidence_weight, mc_prefilter, mc_prefilter_sigma_low,
            mc_prefilter_sigma_high, mc_sima_env, mc_granularity,
            mc_s2p_block_h, mc_s2p_block_w, mc_s2p_smooth_sigma,
            mc_s2p_smooth_sigma_time, mc_s2p_maxregshift, mc_s2p_nonrigid,
            mc_s2p_maxregshift_nr, mc_s2p_nimg_init,
            mc_s2p_two_step_registration, mc_s2p_one_photon_reg,
            mc_s2p_spatial_hp_reg, mc_s2p_pre_smooth, mc_s2p_spatial_taper,
        )
        runner = get_pipeline_runner()
        result = runner.start(
            state.workspace, overrides,
            registry_config=state.registry_config,
            slack_channel=(slack_channel or "").strip() or None,
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
        Input(TICK_ID, "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def _on_live_tick(_n):
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
            _shift_figure(state),
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
        Output(FOV_SELECT_ID, "options"),
        Output(FOV_SELECT_ID, "value"),
        Input(TICK_ID, "n_intervals"),
        Input(workspace_bar.WORKSPACE_VERSION, "data"),
        State(FOV_SELECT_ID, "value"),
        prevent_initial_call=True,
    )
    def _refresh_fovs(_n, _version, current):
        # Cheap: enumeration only globs summary names + reads _mc filename
        # suffixes, never pixels (the heavy mean read is in _render_preview, on
        # selection change). Keeps the current selection if it still exists.
        return fov_select.options_and_value(get_app_state().workspace, current)

    @app.callback(
        Output(PREVIEW_ID, "figure"),
        Input(FOV_SELECT_ID, "value"),
        prevent_initial_call=True,
    )
    def _render_preview(value):
        return _preview_figure(value)


def motion_overrides(fs, tau, k, mc_backend, force_cpu, mc_strip_height,
                     mc_max_displacement, mc_n_template_iters,
                     mc_subpixel_upsample, mc_frame_batch,
                     mc_smooth_sigma_rows, mc_smooth_sigma_time,
                     mc_strip_confidence_weight, mc_prefilter,
                     mc_prefilter_sigma_low, mc_prefilter_sigma_high,
                     mc_sima_env, mc_granularity, mc_s2p_block_h,
                     mc_s2p_block_w, mc_s2p_smooth_sigma,
                     mc_s2p_smooth_sigma_time, mc_s2p_maxregshift,
                     mc_s2p_nonrigid, mc_s2p_maxregshift_nr, mc_s2p_nimg_init,
                     mc_s2p_two_step_registration, mc_s2p_one_photon_reg,
                     mc_s2p_spatial_hp_reg, mc_s2p_pre_smooth,
                     mc_s2p_spatial_taper) -> dict:
    """Form values → ``PipelineConfig`` overrides for a Foundation-only run.

    Separate from the callback so the mapping is testable without Dash, and so
    every ``None`` (an empty numeric input) falls back to the same default the
    form displays rather than reaching ``PipelineConfig`` as ``None``.
    """
    return {
        "fs": float(fs or 7.5),
        "tau": float(tau or 1.0),
        "k_background": int(k or 30),
        "motion_correction_backend": mc_backend or "phasecorr",
        "force_cpu": bool(force_cpu),
        "foundation_only": True,
        "run_centroids": False,
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
                    html.Span(run_panel.fmt_metric(m.get(key)),
                              className="fw-bold")],
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
