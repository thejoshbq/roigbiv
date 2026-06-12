"""Review page — multi-session viewer (read-only).

This page renders a FOV's sessions in a multi-session grid; clicking an ROI
opens a right-side drawer with its metadata + cross-session traces. ROI
editing is intentionally *not* in the UI: the in-app draw tools were retired
in favour of opening the output dir in Fiji/ImageJ and round-tripping edits
through ``roigbiv-reingest``. Each session card surfaces an "Open output
folder" button so researchers can launch their preferred external editor.

The "Active edit session" dropdown still exists — it picks which session's
output dir the Open / Reingest helpers target.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from roigbiv.ui.components.errors import user_error, user_error_figure
from roigbiv.ui.components.forms import HELP_TEXT, help_icon, labeled_with_help
from roigbiv.ui.components.log_stream import log_stream
from roigbiv.ui.components.roi_panel import (
    DETAILS_COLLAPSE_ID,
    DETAILS_TOGGLE_ID,
    roi_panel,
)
from roigbiv.ui.components.sidebar import (
    segmented,
    sidebar_toggle,
)
from roigbiv.ui.components.trace_figure import (
    build_mean_multi,
    build_mean_single,
    build_roi_across_sessions,
    build_roi_single,
)
from roigbiv.ui.logging import get_logger
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.cellpose_trainer import get_trainer
from roigbiv.ui.services.loaders import (
    FOVBundle,
    load_cross_session_bundle,
)
from roigbiv.ui.services.registry_service import list_fovs
from roigbiv.ui.services.theme import axis_muted_color, plotly_template
from roigbiv.ui.services.trace_viz import (
    collect_cross_session_traces,
    collect_sessions_for_fov,
    fetch_single_roi_data,
    load_session_traces,
)


log = get_logger("review")

# F corrected is the default (neuropil-subtracted fluorescence); dF/F is the
# legacy baseline-normalized signal, kept available but no longer primary.
KIND_OPTIONS = [("f", "F corrected"), ("dff", "dF/F")]
COLOR_OPTIONS = [
    ("single", "Single"),
    ("stage", "Stage"),
    ("feature", "Feature"),
    ("gcid", "Cross-session"),
]

SIDEBAR_COL_ID = "roigbiv-review-sidebar-col"
MAIN_COL_ID = "roigbiv-review-main-col"
SIDEBAR_STORE_ID = "roigbiv-review-sidebar-state"
SIDEBAR_TOGGLE_ID = "roigbiv-review-sidebar-toggle"

RIGHT_SIDEBAR_COL_ID = "roigbiv-review-right-sidebar-col"
RIGHT_SIDEBAR_STORE_ID = "roigbiv-review-right-sidebar-state"
RIGHT_SIDEBAR_TOGGLE_ID = "roigbiv-review-right-sidebar-toggle"


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    return html.Div([
        # Session-scoped stores (memory — cleared on tab close).
        dcc.Store(id="roigbiv-review-state", storage_type="memory"),
        dcc.Store(id="roigbiv-review-selected-roi", storage_type="memory"),
        dcc.Store(id="roigbiv-review-output-dir", storage_type="memory"),
        # Bridges between the embedded editor iframe and Dash:
        #  · style-bridge / msg-init are dummy sinks for clientside callbacks
        #  · roi-msg carries the editor's selectAnnotation → ROI drawer
        dcc.Store(id="roigbiv-review-style-bridge", storage_type="memory"),
        dcc.Store(id="roigbiv-review-msg-init", storage_type="memory"),
        dcc.Store(id="roigbiv-review-roi-msg", storage_type="memory"),
        dcc.Interval(id="roigbiv-trainer-tick", interval=2000, disabled=True),
        dcc.Download(id="roigbiv-review-export-download"),
        html.Div([
            sidebar_toggle(toggle_id=SIDEBAR_TOGGLE_ID,
                           store_id=SIDEBAR_STORE_ID),
            sidebar_toggle(toggle_id=RIGHT_SIDEBAR_TOGGLE_ID,
                           store_id=RIGHT_SIDEBAR_STORE_ID),
        ], className="d-flex justify-content-between mb-2"),
        dbc.Row([
            dbc.Col(
                [_selector_card(),
                 _view_controls_card(),
                 _export_card(),
                 _external_edit_card(),
                 _finetune_card()],
                id=SIDEBAR_COL_ID, md=3, className="pe-md-3",
            ),
            dbc.Col([
                html.H4(id="roigbiv-review-title", children="Review",
                        className="mb-2"),
                _canvas_toolbar(),
                html.Div(id="roigbiv-review-canvas"),
                html.Hr(),
                html.H5("FOV signal — per-session mean",
                        className="mb-2 text-muted"),
                dcc.Graph(
                    id="roigbiv-review-fov-trace",
                    figure=_placeholder_fig("Select a FOV to load traces."),
                    config=_TRACE_CONFIG,
                    style={"height": "420px"},
                ),
            ], id=MAIN_COL_ID, md=6),
            dbc.Col([
                _roi_details_card(),
                _roi_trace_card(),
            ], id=RIGHT_SIDEBAR_COL_ID, md=3, className="ps-md-3"),
        ], className="g-3"),
    ])


_TRACE_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d",
                               "toggleSpikelines"],
    "scrollZoom": True,
    "responsive": True,
}


def _selector_card() -> dbc.Card:
    # The Sessions checklist drives the cross-session (longitudinal) overlay;
    # the Active-session radio — co-located beneath it and constrained to the
    # checked sessions — picks which one the editor iframe and external-edit /
    # export handoffs target. This replaces the former redundant third dropdown.
    return dbc.Card(dbc.CardBody([
        html.H6("FOV", className="mb-2"),
        dbc.Select(id="roigbiv-review-fov-select",
                   options=[], value=None, className="mb-2"),
        dbc.Button("Refresh", id="roigbiv-review-refresh",
                   size="sm", outline=True, color="secondary",
                   n_clicks=0, className="mb-3"),
        html.H6("Sessions", className="mb-1"),
        html.Small(
            "Check sessions to overlay across days; the selected radio is the "
            "active session (editor + export target).",
            className="text-muted d-block mb-2",
        ),
        dbc.Checklist(id="roigbiv-review-session-check",
                      options=[], value=[], switch=False,
                      className="mb-2"),
        html.Small("Active session", className="text-muted d-block mb-1"),
        dbc.RadioItems(id="roigbiv-review-active-session",
                       options=[], value=None),
    ]), className="mb-3")


def _view_controls_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.Div(
            [html.H6("Signal", className="mb-0 me-1"),
             *help_icon("roigbiv-review-kind", HELP_TEXT["roigbiv-review-kind"])],
            className="d-flex align-items-center mb-2",
        ),
        segmented("roigbiv-review-kind", KIND_OPTIONS, value="f"),
        html.Div(
            [html.H6("Color", className="mb-0 me-1"),
             *help_icon("roigbiv-review-color",
                        HELP_TEXT["roigbiv-review-color"])],
            className="d-flex align-items-center mt-3 mb-2",
        ),
        segmented("roigbiv-review-color", COLOR_OPTIONS, value="stage"),
    ]), className="mb-3")


def _canvas_toolbar() -> html.Div:
    """Editor-area toolbar (above the canvas) hosting the Overlay toggle.

    Lives in the main column rather than ``_render_canvas`` so the Switch is
    not recreated on every canvas re-render — the clientside style bridge that
    pushes Overlay into the iframe keys off this stable component.
    """
    return html.Div(
        [
            html.Span("Overlay", className="me-2 text-muted small"),
            dbc.Switch(id="roigbiv-review-overlay", value=True,
                       className="d-inline-block m-0"),
            *help_icon("roigbiv-review-overlay",
                       HELP_TEXT["roigbiv-review-overlay"]),
        ],
        className="d-flex align-items-center mb-2",
    )


def _roi_details_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("ROI details", className="mb-2"),
        html.Div(id="roigbiv-review-right-roi",
                 children=roi_panel(None, None)),
    ]), className="mb-3")


def _roi_trace_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("ROI signal — across sessions", className="mb-2 text-muted"),
        dcc.Graph(
            id="roigbiv-review-roi-trace",
            figure=_placeholder_fig(
                "Click an ROI to load its traces."
            ),
            config=_TRACE_CONFIG,
            style={"height": "260px"},
        ),
    ]), className="mb-3")


def _export_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("Download traces", className="mb-2"),
        html.Small(
            "HDF5 file: index = time (s), columns = neuron ID. "
            "Merge sessions with pd.concat([df1, df2], axis=1).",
            className="text-muted d-block mb-2",
        ),
        dbc.Select(
            id="roigbiv-review-export-kind",
            options=[
                {"label": "dF/F + F corrected", "value": "dff,f"},
                {"label": "dF/F only",           "value": "dff"},
                {"label": "All channels",        "value": "dff,f,raw,neuropil"},
            ],
            value="dff,f",
            className="mb-2",
        ),
        dbc.Button(
            "Download .h5", id="roigbiv-review-export-btn",
            size="sm", color="primary", className="w-100", n_clicks=0,
        ),
        html.Div(id="roigbiv-review-export-status", className="mt-1 small text-muted"),
    ]), className="mb-3")


def _external_edit_card() -> dbc.Card:
    """Context for the embedded ROI editor (the main pane).

    The editor for the active session is embedded directly above; edits draw /
    delete ROIs and autosave to ``corrections/corrections.jsonl`` on the server.
    This card just surfaces the active output directory for external-tool
    (Fiji / ``roigbiv-reingest``) round-trips.
    """
    return dbc.Card(dbc.CardBody([
        html.H6("ROI editing", className="mb-2"),
        html.P(
            "Draw, edit, or delete ROIs in the editor above — changes autosave "
            "to the corrections log. Color and Overlay controls apply live. "
            "For Fiji / ImageJ round-trips, point roigbiv-reingest at the path "
            "below.",
            className="text-muted small",
        ),
        html.Div(
            id="roigbiv-review-output-path",
            className="text-muted small font-monospace mt-1",
        ),
    ]), className="mb-3")


def _finetune_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("Fine-tune model", className="mb-2"),
        dbc.Row([
            dbc.Col(labeled_with_help("Epochs", "roigbiv-trainer-epochs",
                                      HELP_TEXT["roigbiv-trainer-epochs"]),
                    width=5, className="small d-flex align-items-center"),
            dbc.Col(dbc.Input(id="roigbiv-trainer-epochs", type="number",
                              value=200, min=1, step=10, size="sm"), width=7),
        ], className="mb-1 g-1"),
        dbc.Row([
            dbc.Col(labeled_with_help("LR", "roigbiv-trainer-lr",
                                      HELP_TEXT["roigbiv-trainer-lr"]),
                    width=5, className="small d-flex align-items-center"),
            dbc.Col(dbc.Input(id="roigbiv-trainer-lr", type="number",
                              value=0.05, step=0.005, size="sm"), width=7),
        ], className="mb-2 g-1"),
        dbc.Button("Start training", id="roigbiv-trainer-train-btn",
                   size="sm", color="warning", className="w-100 mb-2", n_clicks=0),
        dbc.Alert(
            "Deploy overwrites models/deployed/current_model (Git-LFS tracked). "
            "Previous model is backed up with a timestamp suffix.",
            color="warning", className="small py-1 px-2 mb-1",
        ),
        dbc.Button("Deploy model", id="roigbiv-trainer-deploy-btn",
                   size="sm", color="danger", className="w-100 mb-2", n_clicks=0),
        dbc.Button("Reset", id="roigbiv-trainer-reset-btn",
                   size="sm", outline=True, color="secondary",
                   className="w-100 mb-3", n_clicks=0),
        html.Div(id="roigbiv-trainer-status"),
        html.Div(id="roigbiv-trainer-log",
                 children=log_stream([]), className="mt-2"),
    ]), className="mb-3")


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("roigbiv-review-fov-select", "options"),
        Output("roigbiv-review-fov-select", "value"),
        Input("roigbiv-review-refresh", "n_clicks"),
        Input("roigbiv-url", "search"),
        State("roigbiv-review-fov-select", "value"),
    )
    def _load_fov_options(_n, search, current):
        state = get_app_state()
        cfg = state.registry_config
        if cfg is None:
            return [], None
        try:
            rows = list_fovs(cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            log.exception("Listing FOVs failed")
            return [], None
        options = [
            {"label": f"{r.animal_id or '—'} · {r.region or '—'} · "
                      f"{r.fov_id[:8]}…  ({r.n_sessions} sess.)",
             "value": r.fov_id}
            for r in rows
        ]
        preselect = _preselect_from_search(search) or current
        if preselect and preselect not in {o["value"] for o in options}:
            preselect = None
        return options, preselect or (options[0]["value"] if options else None)

    @app.callback(
        Output("roigbiv-review-session-check", "options"),
        Output("roigbiv-review-session-check", "value"),
        Output("roigbiv-review-state", "data"),
        Input("roigbiv-review-fov-select", "value"),
    )
    def _load_cross_session(fov_id):
        if not fov_id:
            return [], [], {}
        cfg = get_app_state().registry_config
        try:
            bundle = load_cross_session_bundle(fov_id, cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            log.exception("Loading cross-session bundle failed")
            return [], [], {"error": f"{type(exc).__name__}: {exc or ''}"}
        options = [
            {"label": (s.session_date.isoformat()
                       if s.session_date else s.session_id[:8]),
             "value": s.session_id}
            for s in bundle.sessions
        ]
        default = [options[0]["value"]] if options else []
        return options, default, {"fov_id": fov_id,
                                  "session_ids": [o["value"] for o in options]}

    @app.callback(
        Output("roigbiv-review-active-session", "options"),
        Output("roigbiv-review-active-session", "value"),
        Input("roigbiv-review-session-check", "value"),
        State("roigbiv-review-state", "data"),
        State("roigbiv-review-active-session", "value"),
    )
    def _update_active(checked, viewer_state, current):
        if not checked:
            return [], None
        fov_id = (viewer_state or {}).get("fov_id")
        cfg = get_app_state().registry_config
        try:
            labels = _session_labels(fov_id, checked, cfg=cfg) if fov_id else {}
        except Exception:  # noqa: BLE001
            log.exception("Resolving session labels failed")
            labels = {sid: sid[:8] for sid in checked}
        options = [{"label": labels.get(sid, sid[:8]), "value": sid}
                   for sid in checked]
        value = current if current in checked else checked[0]
        return options, value

    @app.callback(
        Output("roigbiv-review-output-dir", "data"),
        Input("roigbiv-review-active-session", "value"),
        State("roigbiv-review-state", "data"),
    )
    def _active_to_output_dir(active_session, viewer_state):
        if not (active_session and viewer_state
                and viewer_state.get("fov_id")):
            return None
        cfg = get_app_state().registry_config
        try:
            bundle = load_cross_session_bundle(viewer_state["fov_id"], cfg=cfg)
        except Exception:  # noqa: BLE001
            log.exception("Resolving output_dir failed")
            return None
        for sref in bundle.sessions:
            if sref.session_id == active_session:
                return str(sref.output_dir)
        return None

    @app.callback(
        Output("roigbiv-review-canvas", "children"),
        Output("roigbiv-review-title", "children"),
        Input("roigbiv-review-output-dir", "data"),
        State("roigbiv-review-color", "value"),
        State("roigbiv-review-overlay", "value"),
    )
    def _render_canvas(output_dir, color_mode, overlay_on):
        # The main pane is the embedded ROI editor (OpenSeadragon + Annotorious)
        # for the *active* session. Color/Overlay are seeded into the iframe URL
        # only on (re)load; live changes are driven by the clientside style
        # bridge via postMessage so the iframe never reloads and edit/playback
        # state survives.
        if not output_dir:
            return (html.Em("Select a FOV and active session to edit ROIs.",
                            className="text-muted"), "Review")
        stem = Path(output_dir).name
        dir_b64 = base64.urlsafe_b64encode(output_dir.encode()).decode()
        color = color_mode or "stage"
        overlay = "1" if (True if overlay_on is None else overlay_on) else "0"
        src = (f"/roi-editor/{stem}?dir={dir_b64}"
               f"&color={color}&overlay={overlay}")
        iframe = html.Iframe(
            id="roigbiv-review-editor-iframe",
            src=src,
            style={"width": "100%", "height": "78vh", "border": "0",
                   "borderRadius": "6px", "background": "#0f1117"},
        )
        return iframe, f"Review · {stem}"

    # Live recolor / overlay bridge: push the Color + Overlay controls into the
    # embedded editor iframe via postMessage (no reload, no lost edits).
    app.clientside_callback(
        """
        function(color, overlay) {
            var f = document.getElementById('roigbiv-review-editor-iframe');
            if (f && f.contentWindow) {
                f.contentWindow.postMessage(
                    {type: 'roigbiv-style', color: color || 'stage',
                     overlay: !!overlay},
                    window.location.origin);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("roigbiv-review-style-bridge", "data"),
        Input("roigbiv-review-color", "value"),
        Input("roigbiv-review-overlay", "value"),
    )

    # Install (once) a window message listener that mirrors the editor's
    # selectAnnotation into a Dash store via set_props. Re-runs harmlessly when
    # the canvas re-renders; a window flag guards against duplicate listeners.
    app.clientside_callback(
        """
        function(_children) {
            if (!window._roigbivRoiListener) {
                window._roigbivRoiListener = true;
                window.addEventListener('message', function(e) {
                    if (e.origin !== window.location.origin) return;
                    var d = e.data || {};
                    if (d.type !== 'roigbiv-roi-selected') return;
                    if (d.label_id === undefined || d.label_id === null) return;
                    window._roigbivRoiN = (window._roigbivRoiN || 0) + 1;
                    if (window.dash_clientside && window.dash_clientside.set_props) {
                        window.dash_clientside.set_props(
                            'roigbiv-review-roi-msg',
                            {data: {label_id: d.label_id, n: window._roigbivRoiN}});
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("roigbiv-review-msg-init", "data"),
        Input("roigbiv-review-canvas", "children"),
    )

    @app.callback(
        Output("roigbiv-review-selected-roi", "data"),
        Input("roigbiv-review-roi-msg", "data"),
        State("roigbiv-review-active-session", "value"),
        prevent_initial_call=True,
    )
    def _roi_msg_to_selection(msg, active_session):
        # The editor reports only the ROI label; pair it with the active
        # session (the iframe is always showing that session's ROIs).
        if not msg or not active_session:
            return no_update
        label_id = msg.get("label_id")
        if label_id is None:
            return no_update
        # click_counter increments on every selection so the drawer re-opens
        # even when the same ROI is picked twice.
        return {"session_id": active_session,
                "local_label_id": int(label_id),
                "click_counter": _click_counter_inc()}

    @app.callback(
        Output(RIGHT_SIDEBAR_STORE_ID, "data", allow_duplicate=True),
        Input("roigbiv-review-selected-roi", "data"),
        prevent_initial_call=True,
    )
    def _auto_expand_right(selected):
        if not selected:
            return no_update
        # Auto-open the right sidebar whenever the user clicks a new ROI, so
        # a manual collapse doesn't hide the details they just summoned.
        return {"is_open": True}

    @app.callback(
        Output("roigbiv-review-right-roi", "children"),
        Input("roigbiv-review-selected-roi", "data"),
        State("roigbiv-review-state", "data"),
    )
    def _render_drawer(selected, viewer_state):
        if not (selected and viewer_state and viewer_state.get("fov_id")):
            return roi_panel(None, None)
        session_id = selected.get("session_id")
        if session_id not in (viewer_state.get("session_ids") or []):
            return roi_panel(None, None)
        cfg = get_app_state().registry_config
        try:
            bundle = load_cross_session_bundle(viewer_state["fov_id"], cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            return user_error(exc, "Loading drawer contents")
        fb = bundle.bundles.get(session_id)
        return roi_panel(fb, int(selected["local_label_id"]))

    @app.callback(
        Output(DETAILS_COLLAPSE_ID, "is_open"),
        Input(DETAILS_TOGGLE_ID, "n_clicks"),
        State(DETAILS_COLLAPSE_ID, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_details(n_clicks, is_open):
        if not n_clicks:
            return is_open
        return not is_open

    @app.callback(
        Output("roigbiv-review-fov-trace", "figure"),
        Input("roigbiv-review-state", "data"),
        Input("roigbiv-review-session-check", "value"),
        Input("roigbiv-review-kind", "value"),
        Input("roigbiv-theme", "data"),
    )
    def _render_fov_trace(viewer_state, selected_ids, kind, theme):
        if not (viewer_state and viewer_state.get("fov_id")):
            return _placeholder_fig("Select a FOV to load traces.", theme)
        kind = kind or "f"
        fov_id = viewer_state["fov_id"]
        cfg = get_app_state().registry_config
        fov_meta = _lookup_fov_meta(fov_id, cfg=cfg)
        sel_set = set(selected_ids or [])
        try:
            all_sessions = collect_sessions_for_fov(fov_id, kind=kind, cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            return user_error_figure(exc, "Loading FOV-level traces",
                                     theme=theme)
        if not all_sessions:
            return _placeholder_fig("No sessions on this FOV yet.", theme)
        chosen = [s for s in all_sessions if s.session_id in sel_set]
        if not chosen:
            chosen = all_sessions[:1]
        try:
            if len(chosen) == 1:
                return build_mean_single(fov_meta, chosen[0], theme=theme)
            return build_mean_multi(fov_meta, chosen, theme=theme)
        except Exception as exc:  # noqa: BLE001
            return user_error_figure(exc, "Building FOV-level trace figure",
                                     theme=theme)

    @app.callback(
        Output("roigbiv-review-roi-trace", "figure"),
        Input("roigbiv-review-selected-roi", "data"),
        Input("roigbiv-review-kind", "value"),
        Input("roigbiv-theme", "data"),
        State("roigbiv-review-state", "data"),
    )
    def _render_roi_trace(selected, kind, theme, viewer_state):
        if not (selected and viewer_state and viewer_state.get("fov_id")):
            return _placeholder_fig(
                "Click an ROI in a session above to load traces.", theme,
            )
        session_id = selected.get("session_id")
        if session_id not in (viewer_state.get("session_ids") or []):
            return _placeholder_fig(
                "Click an ROI in a session above to load traces.", theme,
            )
        fov_id = viewer_state["fov_id"]
        kind = kind or "f"
        local_label_id = int(selected["local_label_id"])
        cfg = get_app_state().registry_config
        fov_meta = _lookup_fov_meta(fov_id, cfg=cfg)
        try:
            bundle = load_cross_session_bundle(fov_id, cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            return user_error_figure(exc, "Loading ROI traces", theme=theme)
        fb = bundle.bundles.get(session_id)
        if fb is None:
            return _placeholder_fig(
                "Selected session is no longer available.", theme,
            )
        gcid = _lookup_global_cell_id(fb, local_label_id)
        try:
            if gcid:
                pairs = collect_cross_session_traces(fov_id, gcid, kind=kind, cfg=cfg)
            else:
                ref = next((s for s in bundle.sessions
                            if s.session_id == session_id), None)
                if ref is None:
                    return _placeholder_fig("Selected session not found.",
                                            theme)
                sess = load_session_traces(
                    Path(ref.output_dir),
                    kind=kind,
                    session_date=ref.session_date,
                    session_id=ref.session_id,
                )
                row = sess.row_for_local_label(local_label_id)
                pairs = [(sess, row)] if row is not None else []
        except Exception as exc:  # noqa: BLE001
            return user_error_figure(exc, "Collecting ROI traces", theme=theme)
        if len(pairs) == 1:
            sess, _row = pairs[0]
            try:
                roi_data = fetch_single_roi_data(sess.output_dir, local_label_id)
            except Exception:  # noqa: BLE001
                roi_data = None
            if roi_data is not None:
                return build_roi_single(fov_meta, roi_data, theme=theme)
        return build_roi_across_sessions(fov_meta, pairs, session_id,
                                         theme=theme)

    @app.callback(
        Output("roigbiv-review-output-path", "children"),
        Input("roigbiv-review-output-dir", "data"),
    )
    def _update_output_path(output_dir):
        if not output_dir:
            return "(select an active session to populate)"
        return output_dir

    # ── Trace export callback ───────────────────────────────────────────────

    @app.callback(
        Output("roigbiv-review-export-download", "data"),
        Output("roigbiv-review-export-status", "children"),
        Input("roigbiv-review-export-btn", "n_clicks"),
        State("roigbiv-review-output-dir", "data"),
        State("roigbiv-review-export-kind", "value"),
        prevent_initial_call=True,
    )
    def _on_export(n_clicks, output_dir, kind_str):
        if not output_dir:
            return no_update, "Select an active session first."
        from roigbiv.pipeline.export_io import export_fov_traces_to_tempfile
        kinds = tuple(k.strip() for k in (kind_str or "dff,f").split(",") if k.strip())
        try:
            tmp_path = export_fov_traces_to_tempfile(Path(output_dir), kinds=kinds)
        except Exception as exc:  # noqa: BLE001
            log.exception("Trace export failed")
            return no_update, f"Export failed: {exc}"
        stem = Path(output_dir).name
        return dcc.send_file(str(tmp_path), filename=f"{stem}_traces.h5"), ""

    # ── Fine-tune callbacks ─────────────────────────────────────────────────

    @app.callback(
        Output("roigbiv-trainer-tick", "disabled"),
        Output("roigbiv-trainer-status", "children"),
        Input("roigbiv-trainer-train-btn", "n_clicks"),
        State("roigbiv-review-output-dir", "data"),
        State("roigbiv-trainer-epochs", "value"),
        State("roigbiv-trainer-lr", "value"),
        prevent_initial_call=True,
    )
    def _on_train(_n, output_dir, epochs, lr):
        import time as _time
        if not output_dir:
            return True, dbc.Alert("Select an active session first.",
                                   color="warning", className="small py-1")
        run_id = f"hitl_{_time.strftime('%Y%m%d_%H%M%S')}"
        data_dir = Path(output_dir) / "hitl_staging" / "images"
        masks_dir = Path(output_dir) / "hitl_staging" / "masks"
        ok = get_trainer().start_training(
            run_id, data_dir, masks_dir,
            epochs=int(epochs or 200),
            lr=float(lr or 0.05),
        )
        if not ok:
            return False, dbc.Alert("Trainer is busy.", color="warning",
                                    className="small py-1")
        return False, dbc.Alert(f"Training started: {run_id}", color="info",
                                className="small py-1")

    @app.callback(
        Output("roigbiv-trainer-status", "children", allow_duplicate=True),
        Input("roigbiv-trainer-deploy-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_deploy(_n):
        trainer = get_trainer()
        snap = trainer.snapshot()
        if snap.state != "done":
            return dbc.Alert(
                f"Cannot deploy: trainer state is '{snap.state}' (must be 'done').",
                color="warning", className="small py-1",
            )
        if not snap.run_id:
            return dbc.Alert("No run_id recorded — restart training.",
                             color="danger", className="small py-1")
        try:
            backup = trainer.deploy(snap.run_id)
            msg = f"Deployed {snap.run_id}."
            if backup:
                msg += f" Previous model backed up as {backup.name}."
            return dbc.Alert(msg, color="success", className="small py-1")
        except FileNotFoundError as exc:
            return dbc.Alert(str(exc), color="danger", className="small py-1")

    @app.callback(
        Output("roigbiv-trainer-tick", "disabled", allow_duplicate=True),
        Output("roigbiv-trainer-status", "children", allow_duplicate=True),
        Output("roigbiv-trainer-log", "children", allow_duplicate=True),
        Input("roigbiv-trainer-reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _on_trainer_reset(_n):
        get_trainer().reset()
        return True, None, log_stream([])

    @app.callback(
        Output("roigbiv-trainer-log", "children"),
        Output("roigbiv-trainer-status", "children", allow_duplicate=True),
        Output("roigbiv-trainer-tick", "disabled", allow_duplicate=True),
        Input("roigbiv-trainer-tick", "n_intervals"),
        prevent_initial_call="initial_duplicate",
    )
    def _trainer_tick(_n):
        snap = get_trainer().snapshot()
        _COLORS = {
            "idle": "secondary", "ingesting": "info",
            "training": "warning", "done": "success", "error": "danger",
        }
        badge = dbc.Badge(
            snap.state.upper(),
            color=_COLORS.get(snap.state, "secondary"),
            className="me-1",
        )
        extra = snap.ingest_summary or (f"ERROR: {snap.error}" if snap.error else "")
        status = html.Div([badge, html.Small(extra, className="text-muted")])
        return (
            log_stream(snap.logs),
            status,
            snap.state not in ("ingesting", "training"),
        )

# ── helpers ────────────────────────────────────────────────────────────────


_click_counter_state: dict[str, int] = {"n": 0}


def _click_counter_inc() -> int:
    _click_counter_state["n"] += 1
    return _click_counter_state["n"]


def _session_labels(fov_id: Optional[str], session_ids: list[str], cfg=None) -> dict:
    if not fov_id:
        return {}
    bundle = load_cross_session_bundle(fov_id, cfg=cfg)
    out: dict[str, str] = {}
    for s in bundle.sessions:
        if s.session_id in session_ids:
            out[s.session_id] = (s.session_date.isoformat()
                                  if s.session_date else s.session_id[:8])
    return out


def _preselect_from_search(search: Optional[str]) -> Optional[str]:
    if not search:
        return None
    parsed = urlparse(search if search.startswith("?") else "?" + search)
    params = parse_qs(parsed.query or "")
    values = params.get("fov_id")
    if values and values[0]:
        return values[0]
    return None


def _lookup_fov_meta(fov_id: str, cfg=None) -> dict:
    try:
        rows = list_fovs(cfg=cfg)
    except Exception:  # noqa: BLE001
        log.exception("FOV meta lookup failed")
        return {"fov_id": fov_id}
    for r in rows:
        if r.fov_id == fov_id:
            return {"fov_id": fov_id, "animal_id": r.animal_id,
                    "region": r.region}
    return {"fov_id": fov_id}


def _lookup_global_cell_id(fb: FOVBundle, local_label_id: int) -> Optional[str]:
    for roi in fb.rois:
        if int(roi.label_id) == int(local_label_id):
            return getattr(roi, "global_cell_id", None)
    return None


def _placeholder_fig(message: str, theme: Optional[str] = None) -> dict:
    return {
        "data": [],
        "layout": {
            "autosize": True,
            "template": plotly_template(theme),
            "paper_bgcolor": "#000000",
            "plot_bgcolor": "#000000",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": message,
                "showarrow": False,
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5,
                "font": {"size": 14, "color": axis_muted_color(theme)},
            }],
        },
    }


