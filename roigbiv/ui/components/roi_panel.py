"""Right-side detail panel for a clicked ROI.

Structured as hero (always visible) + collapsible details. The drawer hosting
this component (see ``pages/review.py``) owns the collapse-toggle callback.
"""
from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import dcc, html

from roigbiv.ui.services.colors import FEATURE_LABELS, STAGE_LABELS
from roigbiv.ui.services.loaders import FOVBundle, ROIRender


DETAILS_COLLAPSE_ID = "roigbiv-roi-details-collapse"
DETAILS_TOGGLE_ID = "roigbiv-roi-details-toggle"


def roi_panel(bundle: Optional[FOVBundle], label_id: Optional[int]) -> html.Div:
    """Render the panel contents for the currently-selected ROI."""
    if bundle is None:
        return _empty("Select a FOV to begin.")
    if label_id is None:
        return _empty("Click an ROI in the viewer to inspect it.")
    render = bundle.roi_by_label(int(label_id))
    if render is None:
        return _empty(f"ROI {label_id} not found in current bundle.")
    return _render_roi(render)


def _empty(msg: str) -> html.Div:
    return html.Div(
        [html.P(msg, className="text-muted mb-0")],
        className="p-1",
    )


def _render_roi(r: ROIRender) -> html.Div:
    return html.Div([
        _render_hero(r),
        dbc.Button(
            "Details",
            id=DETAILS_TOGGLE_ID,
            size="sm", outline=True, color="secondary",
            className="mt-2 mb-1",
            n_clicks=0,
        ),
        dbc.Collapse(_render_details(r), id=DETAILS_COLLAPSE_ID, is_open=False),
    ], className="roigbiv-drawer-hero")


def _render_hero(r: ROIRender) -> html.Div:
    stage_label = STAGE_LABELS.get(r.source_stage, f"Stage {r.source_stage}")
    feature_label = FEATURE_LABELS.get(r.activity_type or "", r.activity_type or "—")
    badge_color = _badge_color(r.gate_outcome)
    badges = [
        dbc.Badge(stage_label, color="secondary", className="me-2"),
        dbc.Badge(feature_label, color="info", className="me-2"),
        dbc.Badge(r.gate_outcome, color=badge_color, className="me-2"),
    ]
    if r.is_user:
        badges.append(dbc.Badge("USER", color="danger"))
    hero_rows = [
        html.Div([html.Span("Area: ", className="text-muted"),
                  html.Span(f"{r.area} px")], className="small"),
        html.Div([html.Span("Centroid: ", className="text-muted"),
                  html.Span(f"{r.centroid_yx[0]:.1f}, "
                            f"{r.centroid_yx[1]:.1f}")],
                 className="small"),
    ]
    return html.Div([
        html.H5(f"ROI #{r.label_id}", className="mb-1"),
        html.Div(badges, className="mb-2"),
        html.Div(hero_rows),
    ])


def _render_details(r: ROIRender) -> html.Div:
    rows: list[tuple[str, str]] = []
    if r.global_cell_id:
        rows.append(("Cross-session ID", r.global_cell_id))
    for k, v in sorted(r.features.items()):
        rows.append((k, str(v)))
    if not rows:
        return html.P("No additional features recorded.",
                      className="text-muted small mt-2")
    table = dbc.Table(
        [html.Tbody(
            [html.Tr([html.Td(k), html.Td(v)]) for k, v in rows]
        )],
        borderless=True, size="sm", className="roigbiv-tooltip-table mb-0 mt-2",
    )
    return html.Div(table)


def _badge_color(gate_outcome: str) -> str:
    return {
        "accept": "success",
        "flag": "warning",
        "reject": "dark",
    }.get(gate_outcome, "secondary")


def trace_section(traces_available: bool) -> html.Div:
    """Placeholder for per-ROI trace viewer (loaded lazily on click)."""
    if not traces_available:
        return html.Div()
    return html.Div(
        dcc.Graph(id="roigbiv-trace-graph", config={"displayModeBar": False}),
        className="px-3 pb-3",
    )
