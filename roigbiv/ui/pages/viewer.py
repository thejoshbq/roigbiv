"""Viewer page — per-FOV ROI canvas + FOV-level and ROI-level trace panels.

The page is organised top-to-bottom in the right column:

1. **Viewer canvas** — the mean projection + ROI overlay, one card per
   selected session, rendered by :func:`build_roi_figure`.
2. **FOV-level trace** — the per-session mean-of-ROIs line plot. One line
   for a single selected session, one line per session otherwise.
3. **ROI-level trace** — the signal trace(s) of whichever ROI the user last
   clicked in the canvas. If the ROI carries a ``global_cell_id`` it is drawn
   across every session on the FOV, with the source session highlighted.
4. **ROI metadata** — the :func:`roi_panel` summary of the clicked ROI.

Modes (orthogonal):

``color``     single  | stage | feature | gcid   (gcid = cross-session ID)
``kind``      dff     | f                          (signal for trace panels)

ROIs are always rendered as outlines.

URL params: ``/viewer?fov_id=<uuid>`` preselects the FOV to show.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from roigbiv.ui.components.figure import build_roi_figure
from roigbiv.ui.components.roi_panel import roi_panel
from roigbiv.ui.components.sidebar import segmented, workspace_summary_card
from roigbiv.ui.components.trace_figure import (
    build_mean_multi,
    build_mean_single,
    build_roi_across_sessions,
)
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.loaders import (
    CrossSessionBundle,
    FOVBundle,
    load_cross_session_bundle,
)
from roigbiv.ui.services.registry_service import list_fovs
from roigbiv.ui.services.trace_viz import (
    collect_cross_session_traces,
    collect_sessions_for_fov,
    load_session_traces,
)


KIND_OPTIONS = [("dff", "dF/F"), ("f", "F corrected")]


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    state = get_app_state()
    return html.Div([
        dcc.Store(id="roigbiv-viewer-state", storage_type="memory"),
        dcc.Store(id="roigbiv-viewer-selected-roi", storage_type="memory"),
        dbc.Row([
            dbc.Col([
                workspace_summary_card(state.workspace),
                _controls_card(),
            ], md=4, className="pe-md-4"),
            dbc.Col([
                html.H4(id="roigbiv-viewer-title", children="Viewer",
                        className="mb-2"),
                html.Div(id="roigbiv-viewer-canvas"),
                html.Hr(),
                html.H5("FOV signal — per-session mean",
                        className="mb-2 text-muted"),
                dcc.Graph(
                    id="roigbiv-viewer-fov-trace",
                    figure=_placeholder_fig("Select a FOV to load traces."),
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d",
                                                    "autoScale2d",
                                                    "toggleSpikelines"],
                        "scrollZoom": True,
                    },
                ),
                html.H5("ROI signal — across sessions",
                        className="mt-4 mb-2 text-muted"),
                dcc.Graph(
                    id="roigbiv-viewer-roi-trace",
                    figure=_placeholder_fig(
                        "Click an ROI in a session above to load traces."
                    ),
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["select2d", "lasso2d",
                                                    "autoScale2d",
                                                    "toggleSpikelines"],
                        "scrollZoom": True,
                    },
                ),
                html.Hr(),
                html.Div(id="roigbiv-viewer-roi-detail",
                         children=roi_panel(None, None)),
            ], md=8),
        ], className="g-3"),
    ])


def _controls_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("FOV", className="mb-2"),
        dbc.Select(id="roigbiv-viewer-fov-select",
                   options=[], value=None, className="mb-3"),
        dbc.Button("Refresh FOV list", id="roigbiv-viewer-refresh",
                   size="sm", outline=True, color="secondary",
                   className="mb-3", n_clicks=0),
        html.H6("Sessions", className="mb-2"),
        dbc.Checklist(id="roigbiv-viewer-session-check",
                      options=[], value=[], switch=False,
                      className="mb-3"),
        html.H6("Signal", className="mb-2"),
        segmented("roigbiv-viewer-kind", KIND_OPTIONS, value="dff"),
        html.H6("Color", className="mt-3 mb-2"),
        segmented(
            "roigbiv-viewer-color",
            [("single", "Single"),
             ("stage", "Stage"),
             ("feature", "Feature"),
             ("gcid", "Cross-session")],
            value="stage",
        ),
        html.Small(
            "Cross-session mode uses each ROI's global_cell_id — the same "
            "cell is the same color across days.",
            className="text-muted d-block mt-2",
        ),
    ]))


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("roigbiv-viewer-fov-select", "options"),
        Output("roigbiv-viewer-fov-select", "value"),
        Input("roigbiv-viewer-refresh", "n_clicks"),
        Input("roigbiv-url", "search"),
        State("roigbiv-viewer-fov-select", "value"),
    )
    def _load_fov_options(_n, search, current):
        try:
            rows = list_fovs()
        except Exception:  # noqa: BLE001
            return [], None
        options = [
            {"label": f"{r.animal_id or '—'} · {r.region or '—'} · "
                      f"{r.fov_id[:8]}…  "
                      f"({r.n_sessions} sess.)",
             "value": r.fov_id}
            for r in rows
        ]
        preselect = _preselect_from_search(search) or current
        if preselect and preselect not in {o["value"] for o in options}:
            preselect = None
        return options, preselect or (options[0]["value"] if options else None)

    @app.callback(
        Output("roigbiv-viewer-session-check", "options"),
        Output("roigbiv-viewer-session-check", "value"),
        Output("roigbiv-viewer-state", "data"),
        Input("roigbiv-viewer-fov-select", "value"),
    )
    def _load_cross_session(fov_id: Optional[str]):
        if not fov_id:
            return [], [], {}
        try:
            bundle = load_cross_session_bundle(fov_id)
        except Exception as exc:  # noqa: BLE001
            return [], [], {"error": f"{type(exc).__name__}: {exc}"}
        state = get_app_state()
        state.fov_cache(Path(bundle.sessions[0].output_dir)
                        if bundle.sessions else Path("."))  # prime cache keys
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
        Output("roigbiv-viewer-canvas", "children"),
        Output("roigbiv-viewer-title", "children"),
        Input("roigbiv-viewer-state", "data"),
        Input("roigbiv-viewer-session-check", "value"),
        Input("roigbiv-viewer-color", "value"),
    )
    def _render_canvas(viewer_state, selected_ids, color_mode):
        if not viewer_state or "fov_id" not in viewer_state:
            return html.Em("Select a FOV to load sessions.",
                           className="text-muted"), "Viewer"
        if viewer_state.get("error"):
            return dbc.Alert(viewer_state["error"], color="danger"), "Viewer"
        try:
            bundle = load_cross_session_bundle(viewer_state["fov_id"])
        except Exception as exc:  # noqa: BLE001
            return dbc.Alert(str(exc), color="danger"), "Viewer"
        if not bundle.sessions:
            return html.Em("No sessions on this FOV yet.",
                           className="text-muted"), "Viewer"
        sel_set = set(selected_ids or [])
        if not sel_set:
            sel_set = {bundle.sessions[0].session_id}

        title = _compose_title(bundle, len(sel_set))
        cards = []
        for s in bundle.sessions:
            if s.session_id not in sel_set:
                continue
            fb: FOVBundle = bundle.bundles[s.session_id]
            date_str = s.session_date.isoformat() if s.session_date else s.session_id[:8]
            fig = build_roi_figure(
                fb.mean_M, fb.rois,
                color_mode=color_mode or "stage",
                title=None,
            )
            cards.append(_session_card(s, fb, date_str, fig))

        md_per_card = 12 if len(cards) == 1 else (6 if len(cards) == 2 else 4)
        return (
            dbc.Row([dbc.Col(c, md=md_per_card, className="mb-3") for c in cards]),
            title,
        )

    # Store the last-clicked ROI (session_id + local_label_id) so the metadata
    # panel and the ROI-level trace figure can react independently.
    @app.callback(
        Output("roigbiv-viewer-selected-roi", "data"),
        Input({"type": "roigbiv-session-graph", "session_id": dash.ALL},
              "clickData"),
        State("roigbiv-viewer-state", "data"),
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
        label_id: Optional[int] = None
        cd_data = points[0].get("customdata")
        if cd_data:
            label_id = int(cd_data[0]) if isinstance(cd_data, list) else int(cd_data)
        elif "text" in points[0]:
            try:
                label_id = int(points[0]["text"])
            except (TypeError, ValueError):
                label_id = None
        if label_id is None:
            return no_update

        triggered_id = dash.callback_context.triggered_id
        session_id = (
            triggered_id.get("session_id")
            if isinstance(triggered_id, dict) else None
        )
        if not session_id:
            return no_update
        return {"session_id": session_id, "local_label_id": int(label_id)}

    @app.callback(
        Output("roigbiv-viewer-roi-detail", "children"),
        Input("roigbiv-viewer-selected-roi", "data"),
        State("roigbiv-viewer-state", "data"),
    )
    def _render_metadata(selected_roi, viewer_state):
        if not (selected_roi and viewer_state and viewer_state.get("fov_id")):
            return roi_panel(None, None)
        session_id = selected_roi.get("session_id")
        if session_id not in (viewer_state.get("session_ids") or []):
            return roi_panel(None, None)
        try:
            bundle = load_cross_session_bundle(viewer_state["fov_id"])
        except Exception:  # noqa: BLE001
            return roi_panel(None, None)
        fb = bundle.bundles.get(session_id)
        return roi_panel(fb, int(selected_roi["local_label_id"]))

    @app.callback(
        Output("roigbiv-viewer-fov-trace", "figure"),
        Input("roigbiv-viewer-state", "data"),
        Input("roigbiv-viewer-session-check", "value"),
        Input("roigbiv-viewer-kind", "value"),
    )
    def _render_fov_trace(viewer_state, selected_ids, kind):
        if not (viewer_state and viewer_state.get("fov_id")):
            return _placeholder_fig("Select a FOV to load traces.")
        kind = kind or "dff"
        fov_id = viewer_state["fov_id"]
        fov_meta = _lookup_fov_meta(fov_id)
        sel_set = set(selected_ids or [])
        try:
            all_sessions = collect_sessions_for_fov(fov_id, kind=kind)
        except Exception as exc:  # noqa: BLE001
            return _placeholder_fig(f"Error: {type(exc).__name__}: {exc}")
        if not all_sessions:
            return _placeholder_fig("No sessions on this FOV yet.")
        chosen = [s for s in all_sessions if s.session_id in sel_set]
        if not chosen:
            chosen = all_sessions[:1]

        try:
            if len(chosen) == 1:
                return build_mean_single(fov_meta, chosen[0])
            return build_mean_multi(fov_meta, chosen)
        except Exception as exc:  # noqa: BLE001
            return _placeholder_fig(f"Error: {type(exc).__name__}: {exc}")

    @app.callback(
        Output("roigbiv-viewer-roi-trace", "figure"),
        Input("roigbiv-viewer-selected-roi", "data"),
        Input("roigbiv-viewer-kind", "value"),
        State("roigbiv-viewer-state", "data"),
    )
    def _render_roi_trace(selected_roi, kind, viewer_state):
        if not (selected_roi and viewer_state and viewer_state.get("fov_id")):
            return _placeholder_fig(
                "Click an ROI in a session above to load traces."
            )
        session_id = selected_roi.get("session_id")
        if session_id not in (viewer_state.get("session_ids") or []):
            return _placeholder_fig(
                "Click an ROI in a session above to load traces."
            )
        fov_id = viewer_state["fov_id"]
        kind = kind or "dff"
        local_label_id = int(selected_roi["local_label_id"])
        fov_meta = _lookup_fov_meta(fov_id)

        try:
            bundle = load_cross_session_bundle(fov_id)
        except Exception as exc:  # noqa: BLE001
            return _placeholder_fig(f"Error: {type(exc).__name__}: {exc}")
        fb = bundle.bundles.get(session_id)
        if fb is None:
            return _placeholder_fig("Selected session is no longer available.")

        gcid = _lookup_global_cell_id(fb, local_label_id)
        try:
            if gcid:
                pairs = collect_cross_session_traces(fov_id, gcid, kind=kind)
            else:
                ref = next((s for s in bundle.sessions
                            if s.session_id == session_id), None)
                if ref is None:
                    return _placeholder_fig("Selected session not found.")
                sess = load_session_traces(
                    Path(ref.output_dir),
                    kind=kind,
                    session_date=ref.session_date,
                    session_id=ref.session_id,
                )
                row = sess.row_for_local_label(local_label_id)
                pairs = [(sess, row)] if row is not None else []
        except Exception as exc:  # noqa: BLE001
            return _placeholder_fig(f"Error: {type(exc).__name__}: {exc}")

        return build_roi_across_sessions(fov_meta, pairs, session_id)


# ── helpers ────────────────────────────────────────────────────────────────


def _compose_title(bundle: CrossSessionBundle, n_selected: int) -> str:
    animal = bundle.animal_id or "—"
    region = bundle.region or "—"
    return (f"FOV {bundle.fov_id[:8]} · {animal} / {region}  "
            f"· viewing {n_selected} / {len(bundle.sessions)} session(s)")


def _session_card(sess, bundle: FOVBundle, date_str: str, fig: go.Figure):
    return dbc.Card(dbc.CardBody([
        html.Div([
            html.Strong(f"Session {date_str}"),
            html.Span(f"  · {len(bundle.rois)} ROIs",
                      className="text-muted ms-2"),
        ], className="mb-2"),
        dcc.Graph(
            id={"type": "roigbiv-session-graph",
                "session_id": sess.session_id},
            figure=fig,
            config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["select2d", "autoScale2d",
                                            "toggleSpikelines"],
                "scrollZoom": True,
            },
            style={"height": "520px"},
        ),
    ]))


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


def _placeholder_fig(message: str) -> dict:
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
