"""Review page — HITL drawable corrections.

Built on the same Plotly image figure as the Viewer, with two additions:

1. Plotly draw-mode toolbar buttons (polygon / freehand / eraser).
2. A session-scoped corrections log with undo / redo / commit / re-register.

Corrections are additive — pipeline outputs are never rewritten. Each user
action writes one line to ``{output_dir}/corrections/corrections.jsonl`` at
commit time; materialized ``corrected_masks.tif`` + ``corrected_metadata.json``
are written alongside so the registry re-match can pick them up.

Draw-mode interactions round-trip through ``relayoutData`` — Plotly reports
the drawn shape in pixel coordinates, we translate that to a
:class:`CorrectionOp`, queue it, and re-render.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from roigbiv.pipeline.corrections import CorrectionOp
from roigbiv.ui.components.figure import build_roi_figure
from roigbiv.ui.components.sidebar import segmented, workspace_summary_card
from roigbiv.ui.services.app_state import get_app_state
from roigbiv.ui.services.corrections_session import (
    get_corrections_session,
    reregister_corrected_session,
    reset_corrections_session,
)
from roigbiv.ui.services.loaders import load_fov_bundle
from roigbiv.ui.services.registry_service import list_fovs, list_sessions_for_fov


# ── layout ─────────────────────────────────────────────────────────────────


def layout() -> html.Div:
    state = get_app_state()
    return html.Div([
        dcc.Store(id="roigbiv-review-output-dir", storage_type="memory"),
        dcc.Store(id="roigbiv-review-selected-roi", storage_type="memory"),
        dbc.Row([
            dbc.Col([
                workspace_summary_card(state.workspace),
                _selector_card(),
                _tools_card(),
                _commit_card(),
            ], md=4, className="pe-md-4"),
            dbc.Col([
                html.H4(id="roigbiv-review-title", children="Review",
                        className="mb-2"),
                html.Div(id="roigbiv-review-banner"),
                html.Div(id="roigbiv-review-canvas"),
                html.Hr(),
                html.H5("Pending corrections", className="mb-2"),
                html.Div(id="roigbiv-review-ops-list"),
            ], md=8),
        ], className="g-3"),
    ])


def _selector_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("FOV", className="mb-2"),
        dbc.Select(id="roigbiv-review-fov-select",
                   options=[], value=None, className="mb-3"),
        html.H6("Session", className="mb-2"),
        dbc.Select(id="roigbiv-review-session-select",
                   options=[], value=None, className="mb-3"),
        dbc.Button("Refresh", id="roigbiv-review-refresh",
                   size="sm", outline=True, color="secondary", n_clicks=0),
    ]))


def _tools_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("Drawing tool", className="mb-2"),
        segmented(
            "roigbiv-review-tool",
            [("pan", "Pan / select"),
             ("polygon", "Polygon"),
             ("freehand", "Freehand"),
             ("eraser", "Eraser")],
            value="pan",
        ),
        html.Div([
            html.Small(
                "Draw a closed shape to add a new ROI, or pick a pipeline ROI "
                "by clicking and use the action buttons below.",
                className="text-muted d-block mt-2",
            ),
        ]),
        html.H6("Actions", className="mt-3 mb-2"),
        dbc.ButtonGroup([
            dbc.Button("Delete", id="roigbiv-review-delete",
                       color="danger", outline=True, size="sm"),
            dbc.Button("Relabel tonic", id="roigbiv-review-relabel-tonic",
                       color="warning", outline=True, size="sm"),
            dbc.Button("Relabel phasic", id="roigbiv-review-relabel-phasic",
                       color="info", outline=True, size="sm"),
        ], className="mb-2"),
        dbc.ButtonGroup([
            dbc.Button("Undo", id="roigbiv-review-undo",
                       color="secondary", outline=True, size="sm"),
            dbc.Button("Redo", id="roigbiv-review-redo",
                       color="secondary", outline=True, size="sm"),
            dbc.Button("Discard pending",
                       id="roigbiv-review-discard",
                       color="secondary", outline=True, size="sm"),
        ]),
    ]))


def _commit_card() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("Persist", className="mb-2"),
        html.P(
            "Commit writes corrections.jsonl + corrected_masks.tif alongside "
            "the pipeline outputs. Re-register runs the cross-session matcher "
            "against the corrected artifacts.",
            className="text-muted small",
        ),
        dbc.ButtonGroup([
            dbc.Button("Commit corrections",
                       id="roigbiv-review-commit",
                       color="success", size="sm"),
            dbc.Button("Re-register", id="roigbiv-review-reregister",
                       color="primary", size="sm"),
        ]),
        html.Div(id="roigbiv-review-commit-output", className="mt-2"),
    ]))


# ── callbacks ──────────────────────────────────────────────────────────────


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("roigbiv-review-fov-select", "options"),
        Output("roigbiv-review-fov-select", "value"),
        Input("roigbiv-review-refresh", "n_clicks"),
        State("roigbiv-review-fov-select", "value"),
    )
    def _load_fovs(_n, current):
        try:
            rows = list_fovs()
        except Exception:  # noqa: BLE001
            return [], None
        options = [
            {"label": f"{r.animal_id or '—'} · {r.region or '—'} · "
                      f"{r.fov_id[:8]}…",
             "value": r.fov_id}
            for r in rows
        ]
        if current and current not in {o["value"] for o in options}:
            current = None
        return options, current or (options[0]["value"] if options else None)

    @app.callback(
        Output("roigbiv-review-session-select", "options"),
        Output("roigbiv-review-session-select", "value"),
        Input("roigbiv-review-fov-select", "value"),
    )
    def _load_sessions(fov_id):
        if not fov_id:
            return [], None
        try:
            rows = list_sessions_for_fov(fov_id)
        except Exception:  # noqa: BLE001
            return [], None
        options = [
            {"label": (s.session_date or s.session_id[:8]),
             "value": s.output_dir}
            for s in rows
        ]
        return options, options[0]["value"] if options else None

    @app.callback(
        Output("roigbiv-review-output-dir", "data"),
        Input("roigbiv-review-session-select", "value"),
    )
    def _update_output_dir(output_dir):
        return output_dir

    @app.callback(
        Output("roigbiv-review-canvas", "children"),
        Output("roigbiv-review-ops-list", "children"),
        Output("roigbiv-review-title", "children"),
        Output("roigbiv-review-banner", "children"),
        Input("roigbiv-review-output-dir", "data"),
        Input("roigbiv-review-tool", "value"),
        prevent_initial_call=False,
    )
    def _render(output_dir, tool):
        if not output_dir:
            return (html.Em("Select a session to review.",
                            className="text-muted"),
                    _render_pending_ops([]),
                    "Review",
                    None)
        sess = get_corrections_session(Path(output_dir))
        rendered_rois = _rois_for_display(sess)
        bundle = load_fov_bundle(Path(output_dir))
        fig = build_roi_figure(
            bundle.mean_M, rendered_rois,
            geometry="outline", color_mode="stage",
            title=None,
        )
        fig.update_layout(dragmode=_dragmode_for(tool),
                          newshape=dict(line=dict(color="rgba(231,76,60,0.95)",
                                                   width=2)))
        graph = dcc.Graph(
            id="roigbiv-review-graph",
            figure=fig,
            config={
                "displayModeBar": True,
                "modeBarButtonsToAdd": [
                    "drawclosedpath", "drawopenpath", "eraseshape",
                ],
                "modeBarButtonsToRemove": ["select2d", "autoScale2d"],
                "scrollZoom": True,
            },
            style={"height": "640px"},
        )
        summary = sess.summary()
        banner = _status_banner(summary)
        ops_list = _render_pending_ops(sess.pending)
        title = f"Reviewing  {Path(output_dir).name}"
        return graph, ops_list, title, banner

    @app.callback(
        Output("roigbiv-review-commit-output", "children"),
        Output("roigbiv-review-ops-list", "children",
               allow_duplicate=True),
        Output("roigbiv-review-banner", "children",
               allow_duplicate=True),
        Input("roigbiv-review-commit", "n_clicks"),
        Input("roigbiv-review-reregister", "n_clicks"),
        Input("roigbiv-review-undo", "n_clicks"),
        Input("roigbiv-review-redo", "n_clicks"),
        Input("roigbiv-review-discard", "n_clicks"),
        Input("roigbiv-review-delete", "n_clicks"),
        Input("roigbiv-review-relabel-tonic", "n_clicks"),
        Input("roigbiv-review-relabel-phasic", "n_clicks"),
        State("roigbiv-review-output-dir", "data"),
        State("roigbiv-review-selected-roi", "data"),
        prevent_initial_call=True,
    )
    def _actions(*args):
        output_dir = args[-2]
        selected_label = args[-1]
        if not output_dir:
            return no_update, no_update, dbc.Alert(
                "Select a session first.", color="warning",
            )
        sess = get_corrections_session(Path(output_dir))
        trig = dash.callback_context.triggered_id

        if trig == "roigbiv-review-undo":
            sess.undo()
        elif trig == "roigbiv-review-redo":
            sess.redo()
        elif trig == "roigbiv-review-discard":
            sess.discard_pending()
        elif trig == "roigbiv-review-delete" and selected_label is not None:
            sess.add(CorrectionOp.delete(int(selected_label),
                                         notes="review-page delete"))
        elif trig == "roigbiv-review-relabel-tonic" and selected_label is not None:
            sess.add(CorrectionOp.relabel(int(selected_label),
                                          activity_type="tonic"))
        elif trig == "roigbiv-review-relabel-phasic" and selected_label is not None:
            sess.add(CorrectionOp.relabel(int(selected_label),
                                          activity_type="phasic"))
        elif trig == "roigbiv-review-commit":
            result = sess.commit()
            reset_corrections_session(Path(output_dir))
            state = get_app_state()
            state.invalidate_fov(Path(output_dir))
            msg = (f"Committed {result.n_ops} op(s). "
                   f"Wrote {result.masks_path.name} / {result.metadata_path.name}.")
            return (dbc.Alert(msg, color="success"),
                    _render_pending_ops([]),
                    no_update)
        elif trig == "roigbiv-review-reregister":
            try:
                report = reregister_corrected_session(Path(output_dir), sess)
            except Exception as exc:  # noqa: BLE001
                return (dbc.Alert(f"Re-register failed: {exc}",
                                  color="danger"),
                        no_update, no_update)
            if not report:
                return (dbc.Alert("Nothing to re-register (no corrections?).",
                                  color="warning"),
                        no_update, no_update)
            get_app_state().invalidate_fov(Path(output_dir))
            return (dbc.Alert(
                f"Re-registered: decision={report.get('decision')} "
                f"fov_id={report.get('fov_id')}",
                color="success"),
                    no_update, no_update)

        summary = sess.summary()
        return (no_update, _render_pending_ops(sess.pending),
                _status_banner(summary))

    @app.callback(
        Output("roigbiv-review-selected-roi", "data"),
        Output("roigbiv-review-banner", "children",
               allow_duplicate=True),
        Output("roigbiv-review-ops-list", "children",
               allow_duplicate=True),
        Input("roigbiv-review-graph", "clickData"),
        Input("roigbiv-review-graph", "relayoutData"),
        State("roigbiv-review-output-dir", "data"),
        State("roigbiv-review-selected-roi", "data"),
        prevent_initial_call=True,
    )
    def _on_graph_event(click_data, relayout, output_dir, current_selection):
        if not output_dir:
            return no_update, no_update, no_update

        trig = dash.callback_context.triggered_id
        sess = get_corrections_session(Path(output_dir))

        # New shape drawn?
        if trig == "roigbiv-review-graph" and relayout:
            op = _relayout_to_op(relayout)
            if op is not None:
                sess.add(op)
                return (current_selection,
                        _status_banner(sess.summary()),
                        _render_pending_ops(sess.pending))

        # Click-selected an ROI contour?
        if click_data and click_data.get("points"):
            pt = click_data["points"][0]
            label_id = None
            cd = pt.get("customdata")
            if cd:
                label_id = int(cd[0]) if isinstance(cd, list) else int(cd)
            elif pt.get("text"):
                try:
                    label_id = int(pt["text"])
                except (TypeError, ValueError):
                    label_id = None
            if label_id is not None:
                return (label_id,
                        dbc.Alert(f"Selected ROI #{label_id}",
                                  color="secondary", className="py-2 mb-2"),
                        no_update)
        return no_update, no_update, no_update


# ── helpers ────────────────────────────────────────────────────────────────


def _dragmode_for(tool: str) -> str:
    return {
        "polygon":  "drawclosedpath",
        "freehand": "drawopenpath",
        "eraser":   "eraseshape",
    }.get(tool or "pan", "pan")


def _rois_for_display(sess) -> list:
    """Return the corrected ROI list as ``ROIRender`` objects for the figure."""
    from roigbiv.ui.services.loaders import render_roi

    rois_corrected = sess.corrected_rois()
    return [render_roi(r) for r in rois_corrected]


def _relayout_to_op(relayout: dict[str, Any]) -> Optional[CorrectionOp]:
    """Translate Plotly's ``relayoutData`` from a new shape into a CorrectionOp.

    Plotly emits something like ``{"shapes": [{"type": "path", "path": "M y,x L ..."}]}``
    after ``drawclosedpath``. We parse only the newest shape — earlier shapes
    have already been turned into ops in previous ticks.
    """
    shapes = relayout.get("shapes") if isinstance(relayout, dict) else None
    if not shapes:
        # ``shapes[-1].path`` style is keyed like "shapes[-1].path" by Plotly.
        path_str = None
        for k, v in (relayout or {}).items():
            if k.endswith(".path") and isinstance(v, str):
                path_str = v
        if path_str is None:
            return None
        polygon = _svg_path_to_polygon(path_str)
        if not polygon:
            return None
        return CorrectionOp.add(polygon=polygon, notes="ui draw")

    last = shapes[-1]
    if last.get("type") != "path":
        return None
    polygon = _svg_path_to_polygon(last.get("path", ""))
    if not polygon:
        return None
    return CorrectionOp.add(polygon=polygon, notes="ui draw")


def _svg_path_to_polygon(path: str) -> list[list[float]]:
    """Parse Plotly's ``M x,y L x,y L ... Z`` path into ``[[y, x], ...]``."""
    if not path:
        return []
    out: list[list[float]] = []
    token = ""
    for ch in path:
        if ch in ("M", "L", "Z"):
            if token.strip():
                parts = token.strip().split(",")
                if len(parts) == 2:
                    try:
                        x = float(parts[0])
                        y = float(parts[1])
                        out.append([y, x])   # swap → (row, col)
                    except ValueError:
                        pass
            token = ""
        else:
            token += ch
    if token.strip():
        parts = token.strip().split(",")
        if len(parts) == 2:
            try:
                out.append([float(parts[1]), float(parts[0])])
            except ValueError:
                pass
    return out


def _render_pending_ops(ops: list[CorrectionOp]) -> html.Div:
    if not ops:
        return html.Em("No pending corrections.", className="text-muted")
    rows = []
    for op in ops:
        summary = _op_summary(op)
        rows.append(html.Tr([
            html.Td(op.op),
            html.Td(summary),
            html.Td(op.ts[:19]),
        ]))
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("op"), html.Th("summary"),
                             html.Th("timestamp")])),
         html.Tbody(rows)],
        size="sm", striped=True, borderless=True,
    )


def _op_summary(op: CorrectionOp) -> str:
    if op.op == "add":
        n = len(op.polygon or [])
        return f"add polygon ({n} vertices)"
    if op.op == "delete":
        return f"delete label {op.label_id}"
    if op.op == "edit":
        n = len(op.polygon or [])
        return f"edit label {op.label_id} ({n} vertices)"
    if op.op == "relabel":
        return f"relabel label {op.label_id} → {op.activity_type}"
    if op.op == "merge":
        return f"merge {op.label_ids}"
    if op.op == "split":
        return f"split label {op.label_id} into {len(op.polygons or [])}"
    return str(op.op)


def _status_banner(summary: dict) -> Any:
    color = "info" if summary.get("n_pending") else "secondary"
    text = (f"{summary.get('n_persisted', 0)} persisted · "
            f"{summary.get('n_pending', 0)} pending · "
            f"undo {'✓' if summary.get('can_undo') else '—'} / "
            f"redo {'✓' if summary.get('can_redo') else '—'}")
    return dbc.Alert(text, color=color, className="py-1 mb-2")
