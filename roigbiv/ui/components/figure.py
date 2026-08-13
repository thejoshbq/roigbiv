"""Plotly figure builder for ROI overlays on a mean-projection image.

One entry point — :func:`build_roi_figure` — parameterised by two
orthogonal view-mode axes:

* ``geometry ∈ {"outline", "fill"}`` — render contours vs. tinted mask fills.
* ``color_mode ∈ {"single", "stage", "feature", "gcid", "status"}`` — what
  drives the hue of each ROI. See :mod:`roigbiv.ui.services.colors`.

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
    color_for_match_status,
    color_for_stage,
)
from roigbiv.ui.services.loaders import ROIRender
from roigbiv.ui.services.theme import (
    figure_paper_bg,
    heatmap_colorscale,
    heatmap_reverse,
    plotly_template,
)


GeometryMode = str        # "outline" | "fill"
ColorMode = str           # "single" | "stage" | "feature" | "gcid" | "status"

_OUTLINE_WIDTH = 3.2
_HIGHLIGHT_WIDTH = 5.5


def build_roi_figure(
    mean: Optional[np.ndarray],
    rois: list[ROIRender],
    *,
    geometry: GeometryMode = "outline",
    color_mode: ColorMode = "stage",
    hide_rejected: bool = True,
    show_overlay: bool = True,
    title: Optional[str] = None,
    theme: Optional[str] = None,
    highlight_labels: Optional[dict[int, str]] = None,
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
    show_overlay :
        If ``False``, skip all ROI glyphs so the raw mean projection is
        visible. The figure keeps ROI-less click targets — useful for
        inspecting registration or motion-correction quality.
    title :
        Optional figure title — kept small so the plot dominates.
    highlight_labels :
        ``{label_id: badge_text}``. Those ROIs draw thicker and get their badge
        printed at the centroid. Used by the /cells page to make one cell stand
        out simultaneously across several session panels; the badge carries the
        cell's display number, which this module has no other way to know.
    """
    fig = go.Figure()
    if mean is not None:
        p1 = float(np.percentile(mean, 1))
        p99 = float(np.percentile(mean, 99.5))
        if p99 <= p1:
            p99 = p1 + 1.0
        fig.add_trace(
            go.Heatmap(
                z=mean,
                colorscale=heatmap_colorscale(theme),
                reversescale=heatmap_reverse(theme),
                zmin=p1,
                zmax=p99,
                showscale=False,
                # "skip" (unlike "none") also swallows plotly_click, which
                # /cells needs on empty background to place a new centroid in
                # edit mode. Tooltip stays off either way.
                hoverinfo="none",
                name="mean_M",
            )
        )

    if show_overlay:
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
        highlight = highlight_labels or {}
        for render in visible:
            color = _pick_color(render, color_mode)
            lit = int(render.label_id) in highlight
            for ys, xs in render.contours:
                if not ys:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=xs + [xs[0]],
                        y=ys + [ys[0]],
                        mode="lines",
                        line=dict(
                            color=color,
                            width=_HIGHLIGHT_WIDTH if lit else _OUTLINE_WIDTH,
                            dash=_line_dash(render, color_mode),
                        ),
                        hovertemplate=_hover_text(render),
                        name=str(render.label_id),
                        # ``meta`` is what trace_index_map reads back; unlike
                        # ``name`` it stays an int and is never rendered.
                        meta=int(render.label_id),
                        customdata=[[render.label_id]] * (len(xs) + 1),
                        showlegend=False,
                    )
                )
            if lit:
                fig.add_trace(_badge_trace(render, color, highlight))

    H, W = (mean.shape if mean is not None else (1, 1))
    fig.update_layout(
        template=plotly_template(theme),
        title=dict(text=title or "", x=0.01, xanchor="left",
                   font=dict(size=13)),
        margin=dict(l=0, r=0, t=30 if title else 6, b=0),
        xaxis=dict(
            visible=False, range=[0, W - 1],
            constrain="domain", scaleanchor="y",
        ),
        yaxis=dict(visible=False, range=[H - 1, 0]),   # invert → row 0 at top
        dragmode="pan",
        # Emit click events but never enter select-state: otherwise Plotly
        # dims all non-clicked ROI scatters, breaking cross-session tracking.
        clickmode="event",
        plot_bgcolor=figure_paper_bg(theme),
        paper_bgcolor=figure_paper_bg(theme),
        hoverlabel=dict(bgcolor="#0A1818", bordercolor="#00E5FF", font_color="#C8E8E8"),
    )
    return fig


def trace_index_map(fig: go.Figure) -> dict[int, list[int]]:
    """``{label_id: [trace indices]}`` for the ROI outlines in *fig*.

    Read back off the built figure rather than recomputed from the ROI list, so
    it cannot drift from what was actually drawn. Callers use it to restyle a
    selection through ``dash.Patch`` — repainting one outline instead of
    shipping a fresh multi-megabyte heatmap to the browser on every click.
    """
    out: dict[int, list[int]] = {}
    for i, trace in enumerate(fig.data):
        meta = getattr(trace, "meta", None)
        if isinstance(meta, int):
            out.setdefault(meta, []).append(i)
    return out


# ── internals ──────────────────────────────────────────────────────────────


def _badge_trace(render: ROIRender, color: str, highlight: dict[int, str]):
    """The ``#N`` label pinned at a highlighted ROI's centroid."""
    cy, cx = render.centroid_yx
    return go.Scatter(
        x=[cx], y=[cy],
        mode="text",
        text=[str(highlight[int(render.label_id)])],
        textfont=dict(color=color, size=13, family="monospace"),
        hoverinfo="skip",
        showlegend=False,
    )


def _line_dash(render: ROIRender, color_mode: ColorMode) -> str:
    """Dot a cell that isn't really here.

    A "lost" ROI is drawn at its last known position in a session where it was
    never detected. A solid outline would claim a detection that does not
    exist, so only this one case departs from a solid line.
    """
    if color_mode == "status" and render.match_status == "lost":
        return "dot"
    return "solid"


def _pick_color(render: ROIRender, color_mode: ColorMode) -> str:
    if color_mode == "single":
        return SINGLE_COLOR
    if color_mode == "stage":
        return color_for_stage(render.source_stage)
    if color_mode == "feature":
        return color_for_feature(render.activity_type)
    if color_mode == "gcid":
        return color_for_gcid(render.global_cell_id)
    if color_mode == "status":
        return color_for_match_status(render.match_status)
    return SINGLE_COLOR


def _hover_text(render: ROIRender) -> str:
    lines = [
        f"<b>label</b>: {render.label_id}",
        f"<b>stage</b>: {render.source_stage}",
        f"<b>gate</b>: {render.gate_outcome}",
    ]
    if render.activity_type:
        lines.append(f"<b>activity</b>: {render.activity_type}")
    if render.match_status:
        lines.append(f"<b>across sessions</b>: {render.match_status}")
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
