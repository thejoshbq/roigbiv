"""Flat PNG overlay rendering for pipeline results.

Renders `fov.mean_M` (contrast-stretched) as a grayscale base with ROI contours
drawn on top. Used by `roigbiv.pipeline.run` after a pipeline run to produce an image
suitable for email attachment and quick visual inspection. Pure rendering —
no pipeline calls, no disk I/O besides the final PNG write.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import find_contours

from roigbiv.pipeline.types import FOVData

_ACCEPT_COLOR = "#00FF00"   # green
_FLAG_COLOR = "#FFA500"     # orange
_REJECT_COLOR = "#FF3030"   # red
_COLOR_FOR_OUTCOME = {
    "accept": _ACCEPT_COLOR,
    "flag": _FLAG_COLOR,
    "reject": _REJECT_COLOR,
}
ALL_OUTCOMES: tuple[str, ...] = ("accept", "flag", "reject")
_TEXT_COLOR = (255, 255, 255, 255)
_ANNOTATION_BG = (0, 0, 0, 170)


def render_overlay(
    fov: FOVData,
    output_dir: Path,
    model_name: str,
    fov_stem: str,
    *,
    timestamp: Optional[_dt.datetime] = None,
    outcomes: Iterable[str] = ALL_OUTCOMES,
) -> Path:
    """Render ROI contours on the FOV mean projection and save as PNG.

    Parameters
    ----------
    fov:
        Populated ``FOVData`` with ``mean_M`` (H, W) float32 and ``rois``.
        ``roi.gate_outcome`` must be one of ``"accept" | "flag" | "reject"``.
    output_dir:
        Directory under which an ``overlay/`` subdir will be created.
    model_name:
        Short name (not full path) of the Cellpose model used — written into
        the annotation.
    fov_stem:
        Filename stem (no ``_mc`` suffix, no extension) — used in the output
        filename and annotation.
    timestamp:
        Override the annotation timestamp (for tests). Defaults to ``now()``.
    outcomes:
        Iterable of ``"accept" | "flag" | "reject"`` selecting which ROI gate
        outcomes to draw. Defaults to all three. ROIs whose ``gate_outcome``
        is not in this set are still counted in the annotation but not drawn.

    Returns
    -------
    Path
        Absolute path to the written PNG.

    Raises
    ------
    ValueError
        If ``fov.mean_M`` is ``None``.
    """
    if fov.mean_M is None:
        raise ValueError("fov.mean_M is None; cannot render overlay")

    outcomes_tuple = tuple(outcomes)
    draw_set = frozenset(outcomes_tuple)

    base = _stretch_to_uint8(fov.mean_M, 0.5, 99.5)
    img = Image.fromarray(base).convert("RGB")
    draw = ImageDraw.Draw(img)

    counts = {"accept": 0, "flag": 0, "reject": 0}
    for roi in fov.rois:
        out = roi.gate_outcome
        if out in counts:
            counts[out] += 1
        if out not in draw_set:
            continue
        color = _COLOR_FOR_OUTCOME.get(out)
        if color is None:
            continue
        for contour in find_contours(roi.mask.astype(np.float32), 0.5):
            pts = [(int(round(c)), int(round(r))) for r, c in contour]
            if len(pts) >= 3:
                draw.line(pts + [pts[0]], fill=color, width=1)

    ts = timestamp or _dt.datetime.now()
    lines = [
        fov_stem,
        f"ROIs: {counts['accept']} accept  |  {counts['flag']} flag  |  "
        f"{counts['reject']} reject",
    ]
    if outcomes_tuple != ALL_OUTCOMES:
        lines.append(f"drawn: {','.join(outcomes_tuple)}")
    lines += [
        f"Model: {model_name}",
        ts.isoformat(timespec="seconds"),
    ]
    img = _draw_annotation(img, lines)

    overlay_dir = Path(output_dir) / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{fov_stem}_overlay_{ts.strftime('%Y%m%dT%H%M%S')}.png"
    out_path = overlay_dir / filename
    img.save(out_path, format="PNG")
    return out_path


def render_overlay_from_disk(
    output_dir: Path,
    model_name: str,
    fov_stem: str,
    *,
    timestamp: Optional[_dt.datetime] = None,
    outcomes: Iterable[str] = ALL_OUTCOMES,
) -> Path:
    """Fallback used when the live ``FOVData`` isn't available.

    Reconstitutes a minimal ``FOVData`` from ``output_dir`` via
    ``roigbiv.pipeline.loaders.load_fov_from_output_dir`` and hands off to
    ``render_overlay``. Raises ``FileNotFoundError`` if the output directory
    lacks the expected ``merged_masks.tif`` / ``roi_metadata.json`` layout.
    """
    from roigbiv.pipeline.loaders import load_fov_from_output_dir

    fov, _review_queue = load_fov_from_output_dir(Path(output_dir))
    return render_overlay(
        fov, output_dir, model_name, fov_stem,
        timestamp=timestamp, outcomes=outcomes,
    )


def _stretch_to_uint8(img: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _draw_annotation(img: Image.Image, lines: list[str]) -> Image.Image:
    font = _load_font()
    pad = 6
    line_h = font.getbbox("Mg")[3] + 2
    block_w = max(font.getlength(s) for s in lines) + 2 * pad
    block_h = line_h * len(lines) + 2 * pad

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([(0, 0), (int(block_w), int(block_h))], fill=_ANNOTATION_BG)
    for i, line in enumerate(lines):
        odraw.text((pad, pad + i * line_h), line, font=font, fill=_TEXT_COLOR)

    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _load_font() -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=13)
        except OSError:
            continue
    return ImageFont.load_default()
