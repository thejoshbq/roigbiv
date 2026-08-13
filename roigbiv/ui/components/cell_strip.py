"""One cell, cropped out of every session it could have appeared in.

The contact sheet answers *where* a tracked cell is; this answers *whether the
match is believable*. Same box, same scale, one frame per session, side by
side — the comparison a researcher would make by hand, and the one figure in
the cross-session literature that actually settles the question.

A session where the cell was never detected still gets a crop, taken at the
position the cell last held. That empty box is the evidence for a dropout;
omitting the panel would hide exactly the thing worth looking at.
"""
from __future__ import annotations

import math
from typing import Optional

import dash_bootstrap_components as dbc
import numpy as np
from dash import dcc, html

from roigbiv.ui.components.figure import build_roi_figure
from roigbiv.ui.services.colors import color_for_match_status
from roigbiv.ui.services.loaders import ROIRender
from roigbiv.ui.services.tracked_cells import TrackedCell, TrackedFOV

# Crop width in units of the canonical ROI radius. Four leaves the soma filling
# the middle half of the box with enough surround to judge whether the outline
# sits on the same structure in every session.
_BOX_RADII = 4.0
_FALLBACK_RADIUS = 8.0          # matches PipelineConfig.roi_stamp_radius
_CROP_PX = 150                  # rendered size of one crop


def cell_strip(
    fov: TrackedFOV,
    cell: Optional[TrackedCell],
    *,
    theme: Optional[str] = None,
) -> html.Div:
    """A row of fixed-scale crops of *cell*, one per session in timeline order."""
    if cell is None:
        return html.Div(
            "Click a cell in any session above to compare it across sessions.",
            className="text-muted small")

    half = _half_width(fov, cell)
    centers = _crop_centers(cell)
    cards = [
        _crop_card(fov, cell, i, centers[i], half, theme)
        for i in range(len(fov.sessions))
    ]
    return html.Div(cards, className="d-flex flex-wrap gap-2")


# ── internals ──────────────────────────────────────────────────────────────


def _half_width(fov: TrackedFOV, cell: TrackedCell) -> float:
    """Half the crop box, in pixels — identical for every session of a cell.

    Derived from the drawn footprint rather than read back out of
    ``calibration.json``: every ROI the registry sees is a canonical disk
    (ADR-0003), so its area gives the radius directly, and a crop sized from
    what is actually on screen cannot disagree with it.
    """
    areas = [
        roi.area
        for i, session in enumerate(fov.sessions)
        for roi in session.rois
        if roi.area > 0 and roi.label_id == cell.label_in(i)
    ]
    radius = math.sqrt(np.median(areas) / math.pi) if areas else _FALLBACK_RADIUS
    return max(radius, _FALLBACK_RADIUS) * _BOX_RADII / 2.0


def _crop_centers(cell: TrackedCell) -> list[tuple[float, float]]:
    """Where to center each session's crop, carrying positions into gaps.

    A session that never saw the cell has no position of its own, so it borrows
    the most recent one before it — or, for a cell that arrives late, the first
    one after. Both are the honest "look here" for an absence.
    """
    known = [c for c in cell.centroids if c is not None]
    if not known:
        return [(0.0, 0.0)] * len(cell.centroids)

    out: list[tuple[float, float]] = []
    last = known[0]
    for centroid in cell.centroids:
        if centroid is not None:
            last = centroid
        out.append(last)
    return out


def _crop_card(fov, cell, i, center, half, theme) -> dbc.Card:
    session = fov.sessions[i]
    present = cell.present[i]
    status = _status(cell, i)
    roi = _roi_for(session, cell, i)

    body = [
        dcc.Graph(
            figure=_crop_figure(session.mean_M, roi, center, half, theme),
            config={"displayModeBar": False, "staticPlot": True},
            style={"height": f"{_CROP_PX}px", "width": f"{_CROP_PX}px"},
        ),
        html.Div(f"{i + 1}. {session.short_label}",
                 className="font-monospace small text-truncate",
                 style={"maxWidth": f"{_CROP_PX}px"}),
        html.Div(
            "detected" if present else "not detected",
            className="small " + ("text-muted" if present else "text-danger"),
        ),
    ]
    return dbc.Card(
        dbc.CardBody(body, className="p-2"),
        className="roigbiv-cell-crop",
        style={"borderColor": color_for_match_status(status)},
    )


def _status(cell: TrackedCell, i: int) -> str:
    if not cell.present[i]:
        return "lost"
    return "new" if not any(cell.present[:i]) else "matched"


def _roi_for(session, cell: TrackedCell, i: int) -> Optional[ROIRender]:
    """This session's outline for *cell* — its own, or the carried-forward ghost."""
    label_id = cell.label_in(i)
    if label_id is not None:
        return next((r for r in session.rois if r.label_id == label_id), None)
    return next((r for r in session.rois
                 if r.match_status == "lost"
                 and r.global_cell_id == cell.global_cell_id), None)


def _crop_figure(mean, roi, center, half, theme):
    """A ``build_roi_figure`` over a window of *mean*, in window coordinates."""
    cy, cx = center
    if mean is None:
        return build_roi_figure(mean=None, rois=[], color_mode="status",
                                theme=theme)

    H, W = mean.shape[:2]
    y0 = int(max(0, round(cy - half)))
    x0 = int(max(0, round(cx - half)))
    y1 = int(min(H, round(cy + half)))
    x1 = int(min(W, round(cx + half)))
    if y1 <= y0 or x1 <= x0:        # centre outside the frame entirely
        return build_roi_figure(mean=None, rois=[], color_mode="status",
                                theme=theme)

    fig = build_roi_figure(
        mean=np.asarray(mean)[y0:y1, x0:x1],
        rois=[_shifted(roi, y0, x0)] if roi is not None else [],
        color_mode="status",
        theme=theme,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig


def _shifted(roi: ROIRender, y0: int, x0: int) -> ROIRender:
    """*roi* translated from frame coordinates into crop coordinates."""
    cy, cx = roi.centroid_yx
    return ROIRender(
        label_id=roi.label_id,
        source_stage=roi.source_stage,
        gate_outcome=roi.gate_outcome,
        activity_type=roi.activity_type,
        area=roi.area,
        centroid_yx=(cy - y0, cx - x0),
        contours=[([y - y0 for y in ys], [x - x0 for x in xs])
                  for ys, xs in roi.contours],
        global_cell_id=roi.global_cell_id,
        match_status=roi.match_status,
    )
