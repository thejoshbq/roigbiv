"""Ingest ROI corrections submitted by the browser-based editor.

Converts W3C Web Annotation (Annotorious) polygon output to the same
:class:`CorrectionOp` pipeline used by :mod:`reingest`.  The diff logic
reuses :func:`~roigbiv.pipeline.reingest.greedy_match` and
:func:`~roigbiv.pipeline.reingest.iou` directly — no disk round-trip.

Coordinate conventions
----------------------
* Annotorious SVG polygon points: ``"x,y x,y ..."`` (image pixel space,
  x = column, y = row).
* CorrectionOp polygon / contour: ``[[y, x], [y, x], ...]`` (row, col).
* Internal rasterisation: (row, col) = (y, x).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from roigbiv.pipeline.corrections import (
    CorrectionOp,
    append_correction,
    apply_corrections,
    load_corrections,
    materialize,
)
from roigbiv.pipeline.loaders import load_fov_from_output_dir
from roigbiv.pipeline.reingest import (
    EDIT_IOU,
    PRESERVE_IOU,
    ReingestResult,
    greedy_match,
)
from roigbiv.pipeline.types import ROI


# ── public helpers ─────────────────────────────────────────────────────────────


_POINTS_RE = re.compile(r'\bpoints\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


def parse_svg_polygon(svg_value: str) -> list[tuple[float, float]]:
    """Parse an Annotorious SvgSelector value → ``[(x, y), ...]`` pixel coords.

    Input format: ``<svg><polygon points="x1,y1 x2,y2 ..."></svg>``
    Output: list of (x, y) tuples in image pixel space.

    Intentionally avoids XML parsing (stdlib ET is vulnerable to XXE).
    The Annotorious SVG format is simple enough for a targeted regex.
    """
    svg_value = svg_value.strip()
    if not svg_value:
        return []

    m = _POINTS_RE.search(svg_value)
    if not m:
        return []

    points_str = m.group(1).strip()
    pairs: list[tuple[float, float]] = []
    for token in points_str.split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                pairs.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return pairs


def annotations_to_label_image(
    annotations: list[dict],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterise a list of W3C Annotorious annotations → uint16 label image.

    Each annotation that contains a valid SvgSelector polygon becomes one
    label in the output image (label = 1-based annotation index).  Overlapping
    polygons are resolved last-write-wins.
    """
    label_img = np.zeros(shape, dtype=np.uint16)
    for idx, ann in enumerate(annotations, start=1):
        xy_pairs = _parse_annotation_polygon(ann)
        if len(xy_pairs) < 3:
            continue
        mask = _polygon_to_mask([(y, x) for x, y in xy_pairs], shape)
        label_img[mask] = idx
    return label_img


def reingest_from_annotations(
    output_dir: Path,
    annotations: list[dict],
    *,
    notes: str = "web editor",
    dry_run: bool = False,
    preserve_iou: float = PRESERVE_IOU,
    edit_iou: float = EDIT_IOU,
) -> ReingestResult:
    """Diff browser annotations against the current ROI state and emit ops.

    Parameters
    ----------
    output_dir
        FOV pipeline output directory.
    annotations
        List of W3C Annotorious annotation dicts from ``anno.getAnnotations()``.
    notes
        Free-text note recorded on every emitted op.
    dry_run
        If ``True``, return the diff without writing corrections or materialising.
    preserve_iou, edit_iou
        IoU thresholds (see :mod:`reingest` module docstring).
    """
    output_dir = Path(output_dir)

    fov, _ = load_fov_from_output_dir(output_dir)
    base_rois: list[ROI] = list(fov.rois)
    if base_rois and base_rois[0].mask is not None:
        H, W = base_rois[0].mask.shape
    elif fov.mean_M is not None:
        H, W = fov.mean_M.shape
    else:
        H, W = tifffile.imread(str(output_dir / "merged_masks.tif")).shape

    existing_ops = load_corrections(output_dir)
    current_rois = apply_corrections(base_rois, existing_ops, (H, W))

    current_masks: dict[int, np.ndarray] = {
        int(r.label_id): r.mask
        for r in current_rois
        if r.mask is not None and r.mask.any()
    }

    # Rasterise submitted annotations into per-ROI boolean masks
    new_masks: dict[int, np.ndarray] = {}
    for idx, ann in enumerate(annotations, start=1):
        xy_pairs = _parse_annotation_polygon(ann)
        if len(xy_pairs) < 3:
            continue
        mask = _polygon_to_mask([(y, x) for x, y in xy_pairs], (H, W))
        if mask.any():
            new_masks[idx] = mask

    matches, unmatched_current, unmatched_new = greedy_match(
        current_masks, new_masks, edit_iou,
    )

    ops: list[CorrectionOp] = []
    n_unchanged = 0
    n_edited = 0

    for current_id, new_id, iou_val in matches:
        if iou_val >= preserve_iou:
            n_unchanged += 1
            continue
        polygon = _mask_to_polygon_yx(new_masks[new_id])
        if not polygon:
            continue
        ops.append(CorrectionOp.edit(label_id=current_id, polygon=polygon, notes=notes))
        n_edited += 1

    for new_id in unmatched_new:
        polygon = _mask_to_polygon_yx(new_masks[new_id])
        if not polygon:
            continue
        ops.append(CorrectionOp.add(polygon=polygon, notes=notes))

    for current_id in unmatched_current:
        ops.append(CorrectionOp.delete(label_id=current_id, notes=notes))

    n_added = len(unmatched_new)
    n_deleted = len(unmatched_current)

    result = ReingestResult(
        ops=ops,
        n_unchanged=n_unchanged,
        n_edited=n_edited,
        n_added=n_added,
        n_deleted=n_deleted,
    )

    if dry_run or not ops:
        return result

    for op in ops:
        append_correction(output_dir, op)

    all_ops = load_corrections(output_dir)
    rois_corrected = apply_corrections(base_rois, all_ops, (H, W))
    materialize(rois_corrected, output_dir, (H, W))

    return result


# ── internals ──────────────────────────────────────────────────────────────────


def _parse_annotation_polygon(annotation: dict) -> list[tuple[float, float]]:
    """Extract ``(x, y)`` pixel pairs from a W3C Annotorious annotation dict."""
    target = annotation.get("target", {})
    selector = target.get("selector")
    if isinstance(selector, list):
        selector = next(
            (s for s in selector if s.get("type") == "SvgSelector"), None
        )
    if not selector or selector.get("type") != "SvgSelector":
        return []
    return parse_svg_polygon(selector.get("value", ""))


def _polygon_to_mask(yx_pairs: list[tuple[float, float]], shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a ``(y, x)`` polygon → boolean mask of the given shape."""
    from skimage.draw import polygon as sk_polygon

    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if len(yx_pairs) < 3:
        return mask
    ys = np.clip(np.array([p[0] for p in yx_pairs]), 0, H - 1)
    xs = np.clip(np.array([p[1] for p in yx_pairs]), 0, W - 1)
    rr, cc = sk_polygon(ys, xs, shape=(H, W))
    mask[rr, cc] = True
    return mask


def _mask_to_polygon_yx(mask: np.ndarray) -> list[list[float]]:
    """Extract exterior contour as ``[[y, x], ...]`` for use in CorrectionOp."""
    from skimage.measure import find_contours

    if mask is None or not mask.any():
        return []
    contours = find_contours(mask.astype(np.uint8), level=0.5)
    if not contours:
        return []
    largest = max(contours, key=len)
    return [[float(y), float(x)] for y, x in largest]
