"""Traces page — visualise extracted per-ROI fluorescence traces.

Four view modes:

``mean_single``   mean-FOV trace for one session
``mean_multi``    mean-FOV trace across every session on a FOV
``roi_single``    one ROI (``local_label_id``) for one session
``roi_multi``     one persistent cell (``global_cell_id``) across sessions

URL params: ``/traces?fov_id=<uuid>`` preselects a FOV so the Registry or
Viewer pages can deep-link here for the same FOV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from roigbiv.ui.components.sidebar import segmented, workspace_summary_card
from roigbiv.ui.components.trace_figure import (
    build_mean_multi,
    build_mean_single,
    build_roi_multi,
    build_roi_single,
)
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.registry_service import list_fovs, list_sessions_for_fov
from roigbiv.ui.services.trace_viz import (
    SignalKind,
    collect_cross_session_traces,
    collect_sessions_for_fov,
    list_global_cells_for_fov,
    list_local_rois_for_session,
    load_session_traces,
)


VIEW_MEAN_SINGLE = "mean_single"
VIEW_MEAN_MULTI = "mean_multi"
VIEW_ROI_SINGLE = "roi_single"
VIEW_ROI_MULTI = "roi_multi"

VIEW_OPTIONS = [
    (VIEW_MEAN_SINGLE, "Mean · one session"),
    (VIEW_MEAN_MULTI, "Mean · across sessions"),
    (VIEW_ROI_SINGLE, "ROI · one session"),
    (VIEW_ROI_MULTI, "ROI · across sessions"),
]
KIND_OPTIONS = [("dff", "dF/F"), ("f", "F corrected")]


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    state = get_app_state()
    return html.Div([
        dcc.Store(id="roigbiv-traces-state", storage_type="memory"),
        dbc.Row([
            dbc.Col([
                workspace_summary_card(state.workspace),
                _controls_card(),
            ], md=4, className="pe-md-4"),
            dbc.Col([
                html.H4(id="roigbiv-traces-title", children="Traces",
                        className="mb-2"),
                html.Div(id="roigbiv-traces-meta",
                         className="text-muted small mb-2"),
                dcc.Graph(
                    id="roigbiv-traces-graph",
                    figure={"data": [], "layout": {"height": 420}},
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d",
                                                    "autoScale2d",
                                                    "toggleSpikelines"],
                        "scrollZoom": True,
                    },
                ),
            ], md=8),
        ], className="g-3"),
    ])


def _controls_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("FOV", className="mb-2"),
        dbc.Select(id="roigbiv-traces-fov-select",
                   options=[], value=None, className="mb-2"),
        dbc.Button("Refresh FOV list", id="roigbiv-traces-refresh",
                   size="sm", outline=True, color="secondary",
                   className="mb-3", n_clicks=0),

        html.H6("View", className="mb-2"),
        segmented("roigbiv-traces-view", VIEW_OPTIONS, value=VIEW_MEAN_SINGLE),

        html.H6("Signal", className="mt-3 mb-2"),
        segmented("roigbiv-traces-kind", KIND_OPTIONS, value="dff"),

        html.Div(id="roigbiv-traces-session-wrap", className="mt-3", children=[
            html.H6("Session", className="mb-2"),
            dbc.Select(id="roigbiv-traces-session-select",
                       options=[], value=None),
        ]),

        html.Div(id="roigbiv-traces-roi-wrap", className="mt-3", children=[
            html.H6("ROI (this session)", className="mb-2"),
            dbc.Select(id="roigbiv-traces-roi-select",
                       options=[], value=None),
        ]),

        html.Div(id="roigbiv-traces-gcell-wrap", className="mt-3", children=[
            html.H6("Cell (cross-session)", className="mb-2"),
            dbc.Select(id="roigbiv-traces-gcell-select",
                       options=[], value=None),
            html.Small(
                "Only cells seen in 2+ sessions are listed.",
                className="text-muted d-block mt-1",
            ),
        ]),
    ]))


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:

    @app.callback(
        Output("roigbiv-traces-fov-select", "options"),
        Output("roigbiv-traces-fov-select", "value"),
        Input("roigbiv-traces-refresh", "n_clicks"),
        Input("roigbiv-url", "search"),
        State("roigbiv-traces-fov-select", "value"),
    )
    def _load_fov_options(_n, search, current):
        try:
            rows = list_fovs()
        except Exception:  # noqa: BLE001
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

    # Show/hide conditional selectors based on the view mode.
    @app.callback(
        Output("roigbiv-traces-session-wrap", "style"),
        Output("roigbiv-traces-roi-wrap", "style"),
        Output("roigbiv-traces-gcell-wrap", "style"),
        Input("roigbiv-traces-view", "value"),
    )
    def _toggle_selectors(view: str):
        show = {}
        hide = {"display": "none"}
        needs_session = view in (VIEW_MEAN_SINGLE, VIEW_ROI_SINGLE)
        needs_roi = view == VIEW_ROI_SINGLE
        needs_gcell = view == VIEW_ROI_MULTI
        return (
            show if needs_session else hide,
            show if needs_roi else hide,
            show if needs_gcell else hide,
        )

    # Sessions dropdown follows FOV.
    @app.callback(
        Output("roigbiv-traces-session-select", "options"),
        Output("roigbiv-traces-session-select", "value"),
        Input("roigbiv-traces-fov-select", "value"),
        State("roigbiv-traces-session-select", "value"),
    )
    def _load_sessions(fov_id: Optional[str], current):
        if not fov_id:
            return [], None
        try:
            sessions = list_sessions_for_fov(fov_id)
        except Exception:  # noqa: BLE001
            return [], None
        options = [
            {"label": (f"{s.session_date or '—'} · {s.session_id[:8]}…"),
             "value": s.session_id}
            for s in sessions
        ]
        preselect = current if any(o["value"] == current for o in options) else None
        if preselect is None and options:
            preselect = options[0]["value"]
        return options, preselect

    # ROI dropdown follows (FOV, session). Looks up output_dir via registry.
    @app.callback(
        Output("roigbiv-traces-roi-select", "options"),
        Output("roigbiv-traces-roi-select", "value"),
        Input("roigbiv-traces-fov-select", "value"),
        Input("roigbiv-traces-session-select", "value"),
        State("roigbiv-traces-roi-select", "value"),
    )
    def _load_rois(fov_id: Optional[str], session_id: Optional[str], current):
        if not fov_id or not session_id:
            return [], None
        try:
            sessions = list_sessions_for_fov(fov_id)
        except Exception:  # noqa: BLE001
            return [], None
        row = next((s for s in sessions if s.session_id == session_id), None)
        if row is None:
            return [], None
        rois = list_local_rois_for_session(Path(row.output_dir))
        options = [
            {"label": (f"local {r['local_label_id']}"
                       + (f" · gcid {r['global_cell_id'][:8]}…"
                          if r.get("global_cell_id") else "")),
             "value": int(r["local_label_id"])}
            for r in rois
        ]
        preselect = current if any(o["value"] == current for o in options) else None
        if preselect is None and options:
            preselect = options[0]["value"]
        return options, preselect

    # Cross-session cell dropdown follows FOV.
    @app.callback(
        Output("roigbiv-traces-gcell-select", "options"),
        Output("roigbiv-traces-gcell-select", "value"),
        Input("roigbiv-traces-fov-select", "value"),
        State("roigbiv-traces-gcell-select", "value"),
    )
    def _load_gcells(fov_id: Optional[str], current):
        if not fov_id:
            return [], None
        try:
            rows = list_global_cells_for_fov(fov_id)
        except Exception:  # noqa: BLE001
            return [], None
        options = [
            {"label": f"{r.global_cell_id[:8]}… · seen in {r.n_sessions} sessions",
             "value": r.global_cell_id}
            for r in rows
        ]
        preselect = current if any(o["value"] == current for o in options) else None
        if preselect is None and options:
            preselect = options[0]["value"]
        return options, preselect

    # Main renderer.
    @app.callback(
        Output("roigbiv-traces-graph", "figure"),
        Output("roigbiv-traces-title", "children"),
        Output("roigbiv-traces-meta", "children"),
        Input("roigbiv-traces-fov-select", "value"),
        Input("roigbiv-traces-view", "value"),
        Input("roigbiv-traces-kind", "value"),
        Input("roigbiv-traces-session-select", "value"),
        Input("roigbiv-traces-roi-select", "value"),
        Input("roigbiv-traces-gcell-select", "value"),
    )
    def _render(fov_id, view, kind, session_id, local_label_id, global_cell_id):
        kind = kind or "dff"
        if not fov_id:
            return (
                _placeholder_fig("Select a FOV to load traces."),
                "Traces",
                "",
            )
        fov_meta = _lookup_fov_meta(fov_id)
        title_suffix = f"FOV {fov_id[:8]}… · {fov_meta.get('animal_id') or '—'} / {fov_meta.get('region') or '—'}"

        try:
            if view == VIEW_MEAN_SINGLE:
                sess = _load_session_by_id(fov_id, session_id, kind)
                if sess is None:
                    return _placeholder_fig("Pick a session."), title_suffix, ""
                fig = build_mean_single(fov_meta, sess)
                meta = _meta_line(sess=sess)
                return fig, title_suffix, meta

            if view == VIEW_MEAN_MULTI:
                sessions = collect_sessions_for_fov(fov_id, kind=kind)
                fig = build_mean_multi(fov_meta, sessions)
                meta = _meta_line(n_sessions=len(sessions))
                return fig, title_suffix, meta

            if view == VIEW_ROI_SINGLE:
                sess = _load_session_by_id(fov_id, session_id, kind)
                if sess is None:
                    return _placeholder_fig("Pick a session."), title_suffix, ""
                if local_label_id is None:
                    return (_placeholder_fig("Pick an ROI."),
                            title_suffix,
                            _meta_line(sess=sess))
                fig = build_roi_single(fov_meta, sess, int(local_label_id))
                meta = _meta_line(sess=sess, local_label_id=int(local_label_id))
                return fig, title_suffix, meta

            if view == VIEW_ROI_MULTI:
                if not global_cell_id:
                    return (_placeholder_fig("Pick a cross-session cell."),
                            title_suffix,
                            "")
                bundle = collect_cross_session_traces(fov_id, global_cell_id,
                                                      kind=kind)
                fig = build_roi_multi(fov_meta, global_cell_id, bundle)
                meta = _meta_line(global_cell_id=global_cell_id,
                                  n_sessions=len(bundle))
                return fig, title_suffix, meta
        except Exception as exc:  # noqa: BLE001
            return (
                _placeholder_fig(f"Error: {type(exc).__name__}: {exc}"),
                title_suffix,
                "",
            )

        return _placeholder_fig("Unknown view."), title_suffix, ""


# ── helpers ────────────────────────────────────────────────────────────────


def _placeholder_fig(message: str):
    return {
        "data": [],
        "layout": {
            "height": 420,
            "template": "plotly_white",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "annotations": [{
                "text": message,
                "showarrow": False,
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5,
                "font": {"size": 14, "color": "#888"},
            }],
        },
    }


def _lookup_fov_meta(fov_id: str) -> dict:
    try:
        rows = list_fovs()
    except Exception:  # noqa: BLE001
        return {"fov_id": fov_id}
    for r in rows:
        if r.fov_id == fov_id:
            return {"fov_id": fov_id, "animal_id": r.animal_id, "region": r.region}
    return {"fov_id": fov_id}


def _load_session_by_id(
    fov_id: str, session_id: Optional[str], kind: SignalKind,
):
    if not session_id:
        return None
    sessions = list_sessions_for_fov(fov_id)
    row = next((s for s in sessions if s.session_id == session_id), None)
    if row is None:
        return None
    # The registry gives us an ISO string; pass None for session_date and let
    # the sidecar/empty state drive the labelling if needed.
    return load_session_traces(
        Path(row.output_dir),
        kind=kind,
        session_date=None,
        session_id=row.session_id,
    )


def _meta_line(
    *,
    sess=None,
    n_sessions: Optional[int] = None,
    local_label_id: Optional[int] = None,
    global_cell_id: Optional[str] = None,
) -> str:
    bits: list[str] = []
    if sess is not None:
        if sess.session_id:
            bits.append(f"session_id={sess.session_id}")
        if sess.fs:
            bits.append(f"fs={sess.fs:g} Hz")
        if sess.n_frames:
            bits.append(f"n_frames={sess.n_frames}")
        if sess.source_label:
            bits.append(f"source={sess.source_label}")
        if sess.note:
            bits.append(f"note={sess.note}")
    if n_sessions is not None:
        bits.append(f"n_sessions={n_sessions}")
    if local_label_id is not None:
        bits.append(f"local_label_id={local_label_id}")
    if global_cell_id is not None:
        bits.append(f"global_cell_id={global_cell_id}")
    return " · ".join(bits)


def _preselect_from_search(search: Optional[str]) -> Optional[str]:
    if not search:
        return None
    parsed = urlparse(search if search.startswith("?") else "?" + search)
    params = parse_qs(parsed.query or "")
    values = params.get("fov_id")
    if values and values[0]:
        return values[0]
    return None
