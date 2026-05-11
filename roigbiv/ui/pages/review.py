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

from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import dash
import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, dcc, html, no_update

from roigbiv.ui.components.errors import user_error, user_error_figure
from roigbiv.ui.components.figure import build_roi_figure
from roigbiv.ui.components.log_stream import log_stream
from roigbiv.ui.components.roi_panel import (
    DETAILS_COLLAPSE_ID,
    DETAILS_TOGGLE_ID,
    roi_panel,
)
from roigbiv.ui.components.sidebar import (
    segmented,
    sidebar_toggle,
    workspace_summary_card,
)
from roigbiv.ui.components.trace_figure import (
    build_mean_multi,
    build_mean_single,
    build_roi_across_sessions,
)
from roigbiv.ui.logging import get_logger
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.cellpose_trainer import (
    CellposeNotFoundError,
    get_trainer,
)
from roigbiv.ui.services.external_editor import (
    EditorNotFoundError,
    launch_editor,
    resolve_mask_target,
)
from roigbiv.ui.services.loaders import (
    CrossSessionBundle,
    FOVBundle,
    load_cross_session_bundle,
)
from roigbiv.ui.services.registry_service import list_fovs
from roigbiv.ui.services.theme import axis_muted_color, plotly_template
from roigbiv.ui.services.trace_viz import (
    collect_cross_session_traces,
    collect_sessions_for_fov,
    load_session_traces,
)


log = get_logger("review")

KIND_OPTIONS = [("dff", "dF/F"), ("f", "F corrected")]
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
    state = get_app_state()
    return html.Div([
        # Session-scoped stores (memory — cleared on tab close).
        dcc.Store(id="roigbiv-review-state", storage_type="memory"),
        dcc.Store(id="roigbiv-review-selected-roi", storage_type="memory"),
        dcc.Store(id="roigbiv-review-output-dir", storage_type="memory"),
        # Fine-tuning poll interval (disabled until ingest or training is active).
        dcc.Interval(id="roigbiv-finetune-poll", interval=2_000,
                     disabled=True, n_intervals=0),
        html.Div([
            sidebar_toggle(toggle_id=SIDEBAR_TOGGLE_ID,
                           store_id=SIDEBAR_STORE_ID),
            sidebar_toggle(toggle_id=RIGHT_SIDEBAR_TOGGLE_ID,
                           store_id=RIGHT_SIDEBAR_STORE_ID),
        ], className="d-flex justify-content-between mb-2"),
        dbc.Row([
            dbc.Col(
                [workspace_summary_card(state.workspace),
                 _selector_card(),
                 _view_controls_card(),
                 _external_edit_card(),
                 _finetune_card()],
                id=SIDEBAR_COL_ID, md=3, className="pe-md-3",
            ),
            dbc.Col([
                html.H4(id="roigbiv-review-title", children="Review",
                        className="mb-2"),
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
    return dbc.Card(dbc.CardBody([
        html.H6("FOV", className="mb-2"),
        dbc.Select(id="roigbiv-review-fov-select",
                   options=[], value=None, className="mb-2"),
        dbc.Button("Refresh", id="roigbiv-review-refresh",
                   size="sm", outline=True, color="secondary",
                   n_clicks=0, className="mb-3"),
        html.H6("Sessions", className="mb-2"),
        dbc.Checklist(id="roigbiv-review-session-check",
                      options=[], value=[], switch=False,
                      className="mb-3"),
        html.H6("Active session", className="mb-1"),
        html.Small(
            "External-edit handoff (Open output folder) targets this session.",
            className="text-muted d-block mb-2",
        ),
        dbc.Select(id="roigbiv-review-active-session",
                   options=[], value=None),
    ]), className="mb-3")


def _view_controls_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("Signal", className="mb-2"),
        segmented("roigbiv-review-kind", KIND_OPTIONS, value="dff"),
        html.H6("Color", className="mt-3 mb-2"),
        segmented("roigbiv-review-color", COLOR_OPTIONS, value="stage"),
        html.Div([
            html.H6("Overlay", className="mt-3 mb-1 d-inline-block me-2"),
            dbc.Switch(id="roigbiv-review-overlay",
                       value=True, className="d-inline-block"),
        ]),
        html.Small(
            "Turn off to inspect the raw mean projection.",
            className="text-muted d-block",
        ),
    ]), className="mb-3")


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


def _external_edit_card() -> dbc.Card:
    """Fiji/ImageJ handoff for the active session.

    Drawing tools live in third-party software; this card launches the
    user's TIFF editor on the active mask and surfaces the reingest-CLI
    hint so they can round-trip edits through ``roigbiv-reingest``.
    """
    return dbc.Card(dbc.CardBody([
        html.H6("Edit ROIs in Fiji / ImageJ", className="mb-2"),
        html.P(
            "Open the active session's mask in your TIFF editor "
            "(Fiji / ImageJ preferred; GIMP supported), edit, save, "
            "and run roigbiv-reingest to fold the changes into the "
            "corrections log.",
            className="text-muted small",
        ),
        dbc.Button(
            [html.I(className="bi bi-pencil-square me-2"),
             "Open mask in Fiji / ImageJ"],
            id="roigbiv-review-open-folder",
            color="primary", outline=True, size="sm",
            className="mb-2",
            n_clicks=0,
        ),
        html.Div(
            id="roigbiv-review-output-path",
            className="text-muted small font-monospace",
        ),
        html.Div(
            id="roigbiv-review-open-folder-feedback",
            className="mt-2",
        ),
        html.Hr(className="my-2"),
        html.Small("Then on the terminal:", className="text-muted d-block"),
        html.Code(
            "roigbiv-reingest --output-dir <path> "
            "--new-mask edited.tif",
            className="d-block small",
            style={"whiteSpace": "pre-wrap", "wordBreak": "break-word"},
        ),
    ]), className="mb-3")


def _finetune_card() -> dbc.Card:
    """Four-step Cellpose fine-tuning workflow surfaced from the Review page.

    Steps: GUI launch → ingest *_seg.npy corrections → train → deploy.
    Paths are auto-filled from the active session's hitl_staging/ directory
    and can be overridden by the user.
    """
    default_run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return dbc.Card(dbc.CardBody([
        html.H6("Cellpose fine-tuning", className="mb-2"),

        # ── Step 1: GUI launch ──────────────────────────────────────────────
        html.H6("1 · Cellpose GUI", className="mt-2 mb-1 small fw-semibold"),
        html.P(
            "Opens the active session's staging images in Cellpose GUI with "
            "the deployed model pre-loaded. Correct masks, then save "
            "(creates *_seg.npy).",
            className="text-muted small",
        ),
        dbc.Button(
            [html.I(className="bi bi-window me-2"), "Open in Cellpose GUI"],
            id="roigbiv-finetune-launch-gui",
            color="primary", outline=True, size="sm",
            className="mb-1 w-100",
            n_clicks=0,
        ),
        html.Div(id="roigbiv-finetune-gui-feedback", className="mt-1"),

        html.Hr(className="my-2"),

        # ── Step 2: Ingest ──────────────────────────────────────────────────
        html.H6("2 · Ingest corrections", className="mt-2 mb-1 small fw-semibold"),
        html.P(
            "Convert *_seg.npy files saved by Cellpose GUI into *_masks.tif "
            "for retraining.",
            className="text-muted small",
        ),
        html.Label("Annotations dir", className="small text-muted"),
        dbc.Input(
            id="roigbiv-finetune-annotated-dir", type="text",
            placeholder="auto-filled from active session",
            size="sm", className="mb-2 font-monospace",
        ),
        html.Label("Masks out dir", className="small text-muted"),
        dbc.Input(
            id="roigbiv-finetune-masks-dir", type="text",
            placeholder="auto-filled from active session",
            size="sm", className="mb-2 font-monospace",
        ),
        dbc.Button(
            [html.I(className="bi bi-arrow-down-circle me-2"),
             "Ingest corrections"],
            id="roigbiv-finetune-ingest",
            color="secondary", outline=True, size="sm",
            className="mb-1 w-100",
            n_clicks=0,
        ),
        html.Div(id="roigbiv-finetune-ingest-feedback", className="mt-1"),

        html.Hr(className="my-2"),

        # ── Step 3: Training ────────────────────────────────────────────────
        html.H6("3 · Train model", className="mt-2 mb-1 small fw-semibold"),
        html.Label("Run ID", className="small text-muted"),
        dbc.Input(
            id="roigbiv-finetune-run-id", type="text",
            value=default_run_id,
            size="sm", className="mb-2",
        ),
        dbc.Row([
            dbc.Col([
                html.Label("Epochs", className="small text-muted"),
                dbc.Input(
                    id="roigbiv-finetune-epochs", type="number",
                    value=200, min=10, max=1000, step=10,
                    size="sm",
                ),
            ], width=6),
            dbc.Col([
                html.Label("LR", className="small text-muted"),
                dbc.Input(
                    id="roigbiv-finetune-lr", type="number",
                    value=0.05, min=0.001, max=0.5, step=0.001,
                    size="sm",
                ),
            ], width=6),
        ], className="mb-2"),
        dbc.Button(
            [html.I(className="bi bi-play-circle me-2"), "Start training"],
            id="roigbiv-finetune-start",
            color="success", outline=True, size="sm",
            className="mb-2 w-100",
            n_clicks=0,
        ),
        html.Div(id="roigbiv-finetune-status", className="mb-2"),
        html.Div(id="roigbiv-finetune-log"),

        html.Hr(className="my-2"),

        # ── Step 4: Deploy ──────────────────────────────────────────────────
        html.H6("4 · Deploy model", className="mt-2 mb-1 small fw-semibold"),
        html.P(
            "Copy the trained checkpoint to models/deployed/current_model. "
            "The existing model is auto-backed up.",
            className="text-muted small",
        ),
        dbc.Button(
            [html.I(className="bi bi-check-circle me-2"),
             "Deploy to current_model"],
            id="roigbiv-finetune-deploy",
            color="warning", outline=True, size="sm",
            className="mb-1 w-100",
            disabled=True,
            n_clicks=0,
        ),
        html.Div(id="roigbiv-finetune-deploy-feedback", className="mt-1"),
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
        try:
            rows = list_fovs()
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
        try:
            bundle = load_cross_session_bundle(fov_id)
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
        try:
            labels = _session_labels(fov_id, checked) if fov_id else {}
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
        try:
            bundle = load_cross_session_bundle(viewer_state["fov_id"])
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
        Input("roigbiv-review-state", "data"),
        Input("roigbiv-review-session-check", "value"),
        Input("roigbiv-review-active-session", "value"),
        Input("roigbiv-review-color", "value"),
        Input("roigbiv-review-overlay", "value"),
        Input("roigbiv-theme", "data"),
    )
    def _render_canvas(viewer_state, selected_ids, active_session,
                       color_mode, overlay_on, theme):
        if not viewer_state or "fov_id" not in viewer_state:
            return (html.Em("Select a FOV to load sessions.",
                            className="text-muted"), "Review")
        if viewer_state.get("error"):
            return (user_error(RuntimeError(viewer_state["error"]),
                               "Loading cross-session bundle"),
                    "Review")
        try:
            bundle = load_cross_session_bundle(viewer_state["fov_id"])
        except Exception as exc:  # noqa: BLE001
            return (user_error(exc, "Rendering canvas"), "Review")
        if not bundle.sessions:
            return (html.Em("No sessions on this FOV yet.",
                            className="text-muted"), "Review")

        sel_set = set(selected_ids or [])
        if not sel_set:
            sel_set = {bundle.sessions[0].session_id}

        cards = []
        graph_config = {
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d",
                                       "autoScale2d",
                                       "toggleSpikelines"],
            "scrollZoom": True,
            "responsive": True,
        }
        for s in bundle.sessions:
            if s.session_id not in sel_set:
                continue
            fb = bundle.bundles[s.session_id]
            date_str = (s.session_date.isoformat()
                        if s.session_date else s.session_id[:8])
            fig = build_roi_figure(
                fb.mean_M, fb.rois,
                color_mode=color_mode or "stage",
                show_overlay=bool(overlay_on),
                title=None,
                theme=theme,
            )
            fig.update_layout(dragmode="pan")
            is_active = (s.session_id == active_session)
            cards.append(_session_card(s, fb, date_str, fig, graph_config,
                                        is_active))

        # Always stack vertically at full main-col width — shrinking the grid
        # as session count grew made per-ROI detail unreadable.
        title = _compose_title(bundle, len(cards))
        return (dbc.Row([dbc.Col(c, md=12, className="mb-3")
                         for c in cards]),
                title)

    @app.callback(
        Output("roigbiv-review-selected-roi", "data"),
        Input({"type": "roigbiv-session-graph", "session_id": ALL},
              "clickData"),
        State("roigbiv-review-state", "data"),
        prevent_initial_call=True,
    )
    def _on_roi_click(click_datas, viewer_state):
        if not click_datas or not viewer_state:
            return no_update
        triggered = [(i, cd) for i, cd in enumerate(click_datas) if cd]
        if not triggered:
            return no_update
        _, cd = triggered[-1]
        points = (cd or {}).get("points") or []
        if not points:
            return no_update
        label_id = _extract_label_id(points[0])
        if label_id is None:
            return no_update
        triggered_id = dash.callback_context.triggered_id
        session_id = (triggered_id.get("session_id")
                      if isinstance(triggered_id, dict) else None)
        if not session_id:
            return no_update
        # click_counter increments on every click so the drawer re-opens
        # even if the user clicks the same ROI twice.
        return {"session_id": session_id,
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
        try:
            bundle = load_cross_session_bundle(viewer_state["fov_id"])
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
        kind = kind or "dff"
        fov_id = viewer_state["fov_id"]
        fov_meta = _lookup_fov_meta(fov_id)
        sel_set = set(selected_ids or [])
        try:
            all_sessions = collect_sessions_for_fov(fov_id, kind=kind)
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
        kind = kind or "dff"
        local_label_id = int(selected["local_label_id"])
        fov_meta = _lookup_fov_meta(fov_id)
        try:
            bundle = load_cross_session_bundle(fov_id)
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
                pairs = collect_cross_session_traces(fov_id, gcid, kind=kind)
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
        return build_roi_across_sessions(fov_meta, pairs, session_id,
                                         theme=theme)

    @app.callback(
        Output("roigbiv-review-output-path", "children"),
        Input("roigbiv-review-output-dir", "data"),
    )
    def _show_output_path(output_dir):
        if not output_dir:
            return "(select an active session to populate)"
        return output_dir

    # Launch the user's external TIFF editor (Fiji / ImageJ / GIMP) on the
    # active session's mask. Server-side because the editor is spawned on
    # whichever machine runs ``roigbiv-ui`` — for ``--host 0.0.0.0`` use,
    # the editor still opens on the server, not the lab member's browser
    # machine. See docs/external-editing.md for the round-trip workflow.
    @app.callback(
        Output("roigbiv-review-open-folder-feedback", "children"),
        Input("roigbiv-review-open-folder", "n_clicks"),
        State("roigbiv-review-output-dir", "data"),
        prevent_initial_call=True,
    )
    def _launch_external_editor(n_clicks, output_dir):
        if not (n_clicks and output_dir):
            return no_update
        try:
            target = resolve_mask_target(Path(output_dir))
            editor = launch_editor(target)
        except EditorNotFoundError as exc:
            return user_error(exc, "No TIFF editor found",
                              include_traceback=False)
        except FileNotFoundError as exc:
            return user_error(exc, "No mask file to edit",
                              include_traceback=False)
        except Exception as exc:  # noqa: BLE001
            return user_error(exc, "Launching external editor failed")
        return dbc.Alert(
            f"Opened {target.name} in {editor.name}.",
            color="success", className="mb-0 small",
        )

    # ── Fine-tuning callbacks ─────────────────────────────────────────────────

    @app.callback(
        Output("roigbiv-finetune-annotated-dir", "value"),
        Output("roigbiv-finetune-masks-dir", "value"),
        Input("roigbiv-review-output-dir", "data"),
    )
    def _fill_finetune_paths(output_dir):
        if not output_dir:
            return "", ""
        base = Path(output_dir)
        return (
            str(base / "hitl_staging" / "images"),
            str(base / "hitl_staging" / "masks"),
        )

    @app.callback(
        Output("roigbiv-finetune-gui-feedback", "children"),
        Input("roigbiv-finetune-launch-gui", "n_clicks"),
        State("roigbiv-review-output-dir", "data"),
        prevent_initial_call=True,
    )
    def _launch_cellpose_gui(n_clicks, output_dir):
        if not n_clicks:
            return no_update
        if not output_dir:
            return dbc.Alert(
                "Select an active session first.",
                color="warning", className="mb-0 small",
            )
        try:
            staging_path = get_trainer().launch_gui(Path(output_dir))
        except CellposeNotFoundError as exc:
            return user_error(exc, "Cellpose not found", include_traceback=False)
        except FileNotFoundError as exc:
            return user_error(exc, "Staging materials missing",
                              include_traceback=False)
        except Exception as exc:  # noqa: BLE001
            return user_error(exc, "Launching Cellpose GUI failed")
        return dbc.Alert(
            [
                html.Div("Cellpose GUI launched — staging images and masks pre-loaded.",
                         className="fw-semibold"),
                html.Small(
                    f"Staging dir: {staging_path}",
                    className="font-monospace",
                ),
            ],
            color="success", className="mb-0 small",
        )

    @app.callback(
        Output("roigbiv-finetune-poll", "disabled", allow_duplicate=True),
        Output("roigbiv-finetune-ingest-feedback", "children"),
        Input("roigbiv-finetune-ingest", "n_clicks"),
        State("roigbiv-finetune-annotated-dir", "value"),
        State("roigbiv-finetune-masks-dir", "value"),
        prevent_initial_call=True,
    )
    def _start_ingest(n_clicks, annotated_dir, masks_dir):
        if not n_clicks:
            return no_update, no_update
        if not annotated_dir or not masks_dir:
            return no_update, dbc.Alert(
                "Set annotation and masks directories first.",
                color="warning", className="mb-0 small",
            )
        try:
            ok = get_trainer().start_ingest(
                Path(annotated_dir), Path(masks_dir),
            )
        except Exception as exc:  # noqa: BLE001
            return no_update, user_error(exc, "Starting ingest failed")
        if not ok:
            return no_update, dbc.Alert(
                "A job is already running. Wait for it to finish.",
                color="warning", className="mb-0 small",
            )
        return False, dbc.Alert(
            "Ingesting corrections…", color="info", className="mb-0 small",
        )

    @app.callback(
        Output("roigbiv-finetune-poll", "disabled", allow_duplicate=True),
        Output("roigbiv-finetune-status", "children"),
        Input("roigbiv-finetune-start", "n_clicks"),
        State("roigbiv-finetune-run-id", "value"),
        State("roigbiv-finetune-epochs", "value"),
        State("roigbiv-finetune-lr", "value"),
        State("roigbiv-finetune-annotated-dir", "value"),
        State("roigbiv-finetune-masks-dir", "value"),
        prevent_initial_call=True,
    )
    def _start_training(n_clicks, run_id, epochs, lr, data_dir, masks_dir):
        if not n_clicks:
            return no_update, no_update
        missing = [f for f, v in [("run_id", run_id), ("data_dir", data_dir),
                                   ("masks_dir", masks_dir)] if not v]
        if missing:
            return no_update, dbc.Alert(
                f"Missing required fields: {', '.join(missing)}",
                color="warning", className="mb-0 small",
            )
        try:
            ok = get_trainer().start_training(
                run_id=str(run_id),
                data_dir=Path(data_dir),
                masks_dir=Path(masks_dir),
                epochs=int(epochs or 200),
                lr=float(lr or 0.05),
            )
        except Exception as exc:  # noqa: BLE001
            return no_update, user_error(exc, "Starting training failed")
        if not ok:
            return no_update, dbc.Alert(
                "A job is already running. Wait for it to finish.",
                color="warning", className="mb-0 small",
            )
        return False, _trainer_status_badge("training")

    @app.callback(
        Output("roigbiv-finetune-poll", "disabled", allow_duplicate=True),
        Output("roigbiv-finetune-status", "children", allow_duplicate=True),
        Output("roigbiv-finetune-log", "children"),
        Output("roigbiv-finetune-deploy", "disabled"),
        Input("roigbiv-finetune-poll", "n_intervals"),
        prevent_initial_call=True,
    )
    def _poll_trainer(n_intervals):
        snap = get_trainer().snapshot()
        poll_disabled = snap.state in ("idle", "done", "error")
        deploy_disabled = snap.state != "done"
        log_div = (
            log_stream(snap.logs, empty_hint="Waiting for activity…")
            if snap.state != "idle"
            else html.Div()
        )
        return (
            poll_disabled,
            _trainer_status_badge(snap.state, snap.error),
            log_div,
            deploy_disabled,
        )

    @app.callback(
        Output("roigbiv-finetune-deploy-feedback", "children"),
        Input("roigbiv-finetune-deploy", "n_clicks"),
        State("roigbiv-finetune-run-id", "value"),
        prevent_initial_call=True,
    )
    def _deploy_model(n_clicks, run_id):
        if not (n_clicks and run_id):
            return no_update
        try:
            backup = get_trainer().deploy(str(run_id))
        except FileNotFoundError as exc:
            return user_error(exc, "Checkpoint not found", include_traceback=False)
        except Exception as exc:  # noqa: BLE001
            return user_error(exc, "Deploying model failed")
        msg = f"Deployed {run_id} → current_model."
        if backup:
            msg += f" Previous model backed up to {backup.name}."
        return dbc.Alert(msg, color="success", className="mb-0 small")


# ── helpers ────────────────────────────────────────────────────────────────


_TRAINER_BADGE_COLOR = {
    "idle": "secondary",
    "ingesting": "info",
    "training": "primary",
    "done": "success",
    "error": "danger",
}


def _trainer_status_badge(state: str, error: Optional[str] = None) -> html.Div:
    color = _TRAINER_BADGE_COLOR.get(state, "secondary")
    label = state.capitalize()
    children: list = [dbc.Badge(label, color=color, className="me-2")]
    if state == "training":
        children.insert(0, dbc.Spinner(size="sm", color="primary",
                                       className="me-1"))
    if error and state == "error":
        children.append(html.Small(error, className="text-danger"))
    return html.Div(children, className="d-flex align-items-center small")


_click_counter_state: dict[str, int] = {"n": 0}


def _click_counter_inc() -> int:
    _click_counter_state["n"] += 1
    return _click_counter_state["n"]


def _extract_label_id(pt: dict) -> Optional[int]:
    cd = pt.get("customdata")
    if cd:
        try:
            return int(cd[0]) if isinstance(cd, list) else int(cd)
        except (TypeError, ValueError):
            return None
    text = pt.get("text")
    if text is not None:
        try:
            return int(text)
        except (TypeError, ValueError):
            return None
    return None


def _compose_title(bundle: CrossSessionBundle, n_selected: int) -> str:
    animal = bundle.animal_id or "—"
    region = bundle.region or "—"
    return (f"FOV {bundle.fov_id[:8]} · {animal} / {region}  "
            f"· viewing {n_selected} / {len(bundle.sessions)} session(s)")


def _session_card(sref, fb: FOVBundle, date_str: str, fig,
                  graph_config: dict, is_active: bool):
    card_class = "roigbiv-active-session" if is_active else ""
    header_suffix = " · active" if is_active else ""
    return dbc.Card(dbc.CardBody([
        html.Div([
            html.Strong(f"Session {date_str}{header_suffix}"),
            html.Span(f"  · {len(fb.rois)} ROIs",
                      className="text-muted ms-2"),
        ], className="mb-2"),
        dcc.Graph(
            id={"type": "roigbiv-session-graph",
                "session_id": sref.session_id},
            figure=fig,
            config=graph_config,
            # vh-based height lets the 1:1-locked plot fill most of a wide
            # stacked card; minHeight keeps small viewports sensible.
            style={"height": "85vh", "minHeight": "520px"},
        ),
    ]), className=card_class)


def _session_labels(fov_id: Optional[str], session_ids: list[str]) -> dict:
    if not fov_id:
        return {}
    bundle = load_cross_session_bundle(fov_id)
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


def _lookup_fov_meta(fov_id: str) -> dict:
    try:
        rows = list_fovs()
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


