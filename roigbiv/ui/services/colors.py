"""Color palettes for ROI rendering.

Four view modes drive the color assignment:

``single``
    All ROIs render in one accent color. Good for morphology-only inspection.
``stage``
    Hue keyed to ``ROI.source_stage``. User-corrected ROIs (sentinel 99) get
    a distinct color so HITL edits are visually separable from pipeline output.
``feature``
    Hue keyed to ``ROI.activity_type``. ``phasic``, ``sparse``, ``tonic``,
    ``silent``, ``ambiguous`` each have a dedicated color.
``gcid``
    Hue deterministically hashed from the cross-session ``global_cell_id``
    via :func:`roigbiv.pipeline.cross_session_viewer._rgba_for_global_cell_id`,
    so the same cell is the same color across sessions. Unmatched (no gcid)
    ROIs fall back to a neutral gray.
"""
from __future__ import annotations

from typing import Optional

from roigbiv.pipeline.corrections import USER_STAGE_SENTINEL

SINGLE_COLOR: str = "rgba(46, 204, 113, 0.85)"    # emerald
UNTRACKED_COLOR: str = "rgba(140, 140, 140, 0.55)"

STAGE_PALETTE: dict[int, str] = {
    1: "rgba(52, 152, 219, 0.85)",    # Cellpose — blue
    2: "rgba(230, 126, 34, 0.85)",    # Suite2p  — orange
    3: "rgba(155, 89, 182, 0.85)",    # Template — purple
    4: "rgba(241, 196, 15, 0.90)",    # Tonic    — gold
    USER_STAGE_SENTINEL: "rgba(231, 76, 60, 0.90)",   # User     — red
}

FEATURE_PALETTE: dict[str, str] = {
    "phasic":    "rgba(41, 128, 185, 0.85)",      # deep blue
    "sparse":    "rgba(46, 204, 113, 0.85)",      # emerald
    "tonic":     "rgba(243, 156, 18, 0.90)",      # amber
    "silent":    "rgba(127, 140, 141, 0.70)",     # slate
    "ambiguous": "rgba(192, 57, 43, 0.85)",       # brick red
}


def color_for_stage(source_stage: int) -> str:
    return STAGE_PALETTE.get(int(source_stage), SINGLE_COLOR)


def color_for_feature(activity_type: Optional[str]) -> str:
    if not activity_type:
        return UNTRACKED_COLOR
    return FEATURE_PALETTE.get(activity_type, UNTRACKED_COLOR)


def color_for_gcid(gcid: Optional[str]) -> str:
    if not gcid:
        return UNTRACKED_COLOR
    from roigbiv.pipeline.cross_session_viewer import _rgba_for_global_cell_id

    r, g, b, a = _rgba_for_global_cell_id(gcid)
    return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a:.2f})"


def session_colors(n: int, *, alpha: float = 0.65) -> list[str]:
    """Return ``n`` rgba strings sampled from viridis in chronological order.

    Samples are evenly spaced in ``[0.15, 0.85]`` so both extremes stay
    readable on a white background. For ``n <= 1`` the single color is the
    midpoint of the palette.
    """
    if n <= 0:
        return []
    from plotly.colors import sample_colorscale

    if n == 1:
        positions = [0.5]
    else:
        step = (0.85 - 0.15) / (n - 1)
        positions = [0.15 + i * step for i in range(n)]

    out: list[str] = []
    for raw in sample_colorscale("Viridis", positions):
        # sample_colorscale returns 'rgb(r, g, b)' strings.
        inner = raw[raw.index("(") + 1 : raw.index(")")]
        r, g, b = (int(round(float(p.strip()))) for p in inner.split(","))
        out.append(f"rgba({r}, {g}, {b}, {alpha:.2f})")
    return out


STAGE_LABELS: dict[int, str] = {
    1: "Stage 1 — Cellpose",
    2: "Stage 2 — Suite2p",
    3: "Stage 3 — Template",
    4: "Stage 4 — Tonic",
    USER_STAGE_SENTINEL: "User correction",
}

FEATURE_LABELS: dict[str, str] = {
    "phasic":    "Phasic",
    "sparse":    "Sparse",
    "tonic":     "Tonic",
    "silent":    "Silent",
    "ambiguous": "Ambiguous",
}
