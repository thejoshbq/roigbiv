"""Plotly figure builder for ROI overlays on a mean-projection image.

One entry point — :func:`build_roi_figure` — parameterised by two
orthogonal view-mode axes:

* ``geometry ∈ {"outline", "fill"}`` — render contours vs. tinted mask fills.
* ``color_mode ∈ {"single", "stage", "feature", "gcid"}`` — what drives the
  hue of each ROI. See :mod:`roigbiv.ui.services.colors`.

The figure uses a ``heatmap`` trace for the mean projection (fast, supports
zoom + pan natively) and one ``scatter`` trace per ROI for contours, plus a
single composited ``Image`` trace for fills (drawn on the same axis).

Drawing mode (``drawmode`` / ``dragmode``) is injected by the Review page via
``update_layout`` — this module is drawing-tool-agnostic.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from roigbiv.ui.services.colors import (
    SINGLE_COLOR,
    color_for_feature,
    color_for_gcid,
    color_for_stage,
)
from roigbiv.ui.services.loaders import ROIRender


GeometryMode = str        # "outline" | "fill"
ColorMode = str           # "single" | "stage" | "feature" | "gcid"


def build_roi_figure(
    mean: Optional[np.ndarray],
    rois: list[ROIRender],
    *,
    geometry: GeometryMode = "outline",
    color_mode: ColorMode = "stage",
    hide_rejected: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """Compose the mean projection with ROI overlays.

    Parameters
    ----------
    mean :
        ``(H, W)`` float mean projection. ``None`` renders an empty canvas.
    rois :
        ROIRender objects — contours already in pixel coordinates.
    geometry :
        ``"outline"`` draws ring scatters; ``"fill"`` draws an RGBA-composited
        image on top of the mean.
    color_mode :
        Drives per-ROI hue — see module docstring.
    hide_rejected :
        Drops ROIs with ``gate_outcome == "reject"`` from the overlay (they
        live in the pipeline output for auditing but add noise to viewers).
    title :
        Optional figure title — kept small so the plot dominates.
    """
    fig = go.Figure()
    if mean is not None:
        fig.add_trace(
            go.Heatmap(
                z=mean,
                colorscale="Greys",
                reversescale=True,
                showscale=False,
                hoverinfo="skip",
                name="mean_M",
            )
        )

    visible = [
        r for r in rois
        if not (hide_rejected and r.gate_outcome == "reject")
    ]

    if geometry == "fill" and mean is not None:
        overlay = _build_fill_overlay(mean.shape, visible, color_mode)
        if overlay is not None:
            fig.add_trace(overlay)

    # Outlines are drawn in both modes — they're the click target and give
    # fills a clean edge. For "outline" mode, this is the only ROI glyph.
    for render in visible:
        color = _pick_color(render, color_mode)
        for ys, xs in render.contours:
            if not ys:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs + [xs[0]],
                    y=ys + [ys[0]],
                    mode="lines",
                    line=dict(color=color, width=1.6),
                    hovertemplate=_hover_text(render),
                    name=str(render.label_id),
                    customdata=[[render.label_id]] * (len(xs) + 1),
                    showlegend=False,
                )
            )

    H, W = (mean.shape if mean is not None else (1, 1))
    fig.update_layout(
        title=dict(text=title or "", x=0.01, xanchor="left",
                   font=dict(size=13)),
        margin=dict(l=0, r=0, t=30 if title else 6, b=0),
        xaxis=dict(
            visible=False, range=[0, W - 1],
            constrain="domain", scaleanchor="y",
        ),
        yaxis=dict(visible=False, range=[H - 1, 0]),   # invert → row 0 at top
        dragmode="pan",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hoverlabel=dict(bgcolor="rgba(44,62,80,0.92)", font_color="white"),
    )
    return fig


# ── internals ──────────────────────────────────────────────────────────────


def _pick_color(render: ROIRender, color_mode: ColorMode) -> str:
    if color_mode == "single":
        return SINGLE_COLOR
    if color_mode == "stage":
        return color_for_stage(render.source_stage)
    if color_mode == "feature":
        return color_for_feature(render.activity_type)
    if color_mode == "gcid":
        return color_for_gcid(render.global_cell_id)
    return SINGLE_COLOR


def _hover_text(render: ROIRender) -> str:
    lines = [
        f"<b>label</b>: {render.label_id}",
        f"<b>stage</b>: {render.source_stage}",
        f"<b>gate</b>: {render.gate_outcome}",
    ]
    if render.activity_type:
        lines.append(f"<b>activity</b>: {render.activity_type}")
    if render.global_cell_id:
        lines.append(f"<b>gcid</b>: {render.global_cell_id[:8]}")
    if render.area:
        lines.append(f"<b>area</b>: {render.area} px")
    return "<br>".join(lines) + "<extra></extra>"


def _build_fill_overlay(
    shape: tuple[int, int],
    rois: list[ROIRender],
    color_mode: ColorMode,
):
    """Rasterise per-ROI fills into a single RGBA Image trace.

    Returns ``None`` if the environment can't rasterise (missing
    ``skimage.draw.polygon``), letting the caller render outlines only.
    """
    try:
        from skimage.draw import polygon as sk_polygon
    except ImportError:
        return None

    H, W = int(shape[0]), int(shape[1])
    rgba = np.zeros((H, W, 4), dtype=np.uint8)

    for render in rois:
        color = _pick_color(render, color_mode)
        rgba_tuple = _parse_rgba(color)
        for ys, xs in render.contours:
            if not ys:
                continue
            rr, cc = sk_polygon(
                np.asarray(ys, dtype=float),
                np.asarray(xs, dtype=float),
                shape=(H, W),
            )
            if rr.size == 0:
                continue
            rgba[rr, cc] = rgba_tuple

    return go.Image(z=rgba, hoverinfo="skip")


def _parse_rgba(text: str) -> tuple[int, int, int, int]:
    """Parse ``"rgba(r, g, b, a)"`` into ``(r, g, b, a_uint8)``."""
    inner = text.strip()
    if inner.startswith("rgba"):
        inner = inner[inner.find("(") + 1: inner.rfind(")")]
    parts = [p.strip() for p in inner.split(",")]
    try:
        r = int(float(parts[0]))
        g = int(float(parts[1]))
        b = int(float(parts[2]))
        a = float(parts[3]) if len(parts) > 3 else 0.85
    except (IndexError, ValueError):
        return (60, 60, 60, 160)
    return (r, g, b, int(round(a * 255)))
