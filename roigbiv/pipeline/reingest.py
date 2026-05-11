"""Ingest an externally-edited ROI mask back into the corrections log.

Researchers edit ROIs in Fiji/ImageJ (the in-app editor was retired) and
save a new label TIFF. :func:`reingest_mask` diffs that TIFF against the
current corrected state, emits :class:`CorrectionOp` entries, appends them
to ``corrections.jsonl``, and re-materialises the corrected artifacts.

Diff semantics
--------------
Per-ROI continuity is recovered by IoU matching, not by the user's label
ids in the edited TIFF (Fiji typically renumbers). For each pair of
(current ROI, new ROI):

* IoU >= 0.95              — preserved, no op
* 0.50 <= IoU < 0.95       — ``edit`` (keeps the original ``label_id``)
* unmatched new ROI        — ``add``
* unmatched current ROI    — ``delete``

The thresholds are tunable. The emitted ops replay deterministically via
:func:`roigbiv.pipeline.corrections.apply_corrections`, so the same
``corrected_masks.tif`` would be regenerated from a clean log.
"""
from __future__ import annotations

from dataclasses import dataclass
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
from roigbiv.pipeline.types import ROI


PRESERVE_IOU = 0.95   # IoU at/above which we treat ROIs as identical (no-op)
EDIT_IOU = 0.50       # IoU at/above which we treat the change as an `edit`


@dataclass
class ReingestResult:
    """Diff outcome — populated whether we wrote anything or not."""
    ops: list[CorrectionOp]
    n_unchanged: int
    n_edited: int
    n_added: int
    n_deleted: int

    def summary(self) -> str:
        parts = [
            f"{self.n_added} added",
            f"{self.n_edited} edited",
            f"{self.n_deleted} deleted",
            f"{self.n_unchanged} unchanged",
        ]
        return ", ".join(parts)


def reingest_mask(
    output_dir: Path,
    new_mask_path: Path,
    *,
    notes: Optional[str] = None,
    dry_run: bool = False,
    preserve_iou: float = PRESERVE_IOU,
    edit_iou: float = EDIT_IOU,
) -> ReingestResult:
    """Diff ``new_mask_path`` against the current ROIs and emit ops.

    Parameters
    ----------
    output_dir
        FOV pipeline output dir (e.g. ``inference/pipeline/<stem>/``).
    new_mask_path
        Externally-edited label TIFF (uint16). Background must be 0.
    notes
        Free-text note recorded on every emitted op (typically identifies
        the editor / tool / date).
    dry_run
        If ``True``, return the diff without writing the JSONL log or
        materialising ``corrected_masks.tif``.
    preserve_iou, edit_iou
        Thresholds for the IoU matching policy described in the module
        docstring.
    """
    output_dir = Path(output_dir)
    new_mask_path = Path(new_mask_path)

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

    new_labels = _load_label_image(new_mask_path, expected_shape=(H, W))

    current_masks = {
        int(r.label_id): r.mask
        for r in current_rois
        if r.mask is not None and r.mask.any()
    }
    new_masks = _split_label_image(new_labels)

    matches, unmatched_current, unmatched_new = _greedy_match(
        current_masks, new_masks, edit_iou,
    )

    notes = notes or "external edit (Fiji)"
    ops: list[CorrectionOp] = []
    n_unchanged = 0
    n_edited = 0

    for current_id, new_id, iou in matches:
        if iou >= preserve_iou:
            n_unchanged += 1
            continue
        polygon = _mask_to_polygon(new_masks[new_id])
        if not polygon:
            continue
        ops.append(CorrectionOp.edit(
            label_id=current_id, polygon=polygon, notes=notes,
        ))
        n_edited += 1

    for new_id in unmatched_new:
        polygon = _mask_to_polygon(new_masks[new_id])
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


# ── internals ──────────────────────────────────────────────────────────────


def _load_label_image(path: Path, *, expected_shape: tuple[int, int]) -> np.ndarray:
    img = tifffile.imread(str(path))
    if img.ndim != 2:
        raise ValueError(
            f"expected a 2-D label image at {path}; got shape {img.shape}",
        )
    if expected_shape and img.shape != expected_shape:
        raise ValueError(
            f"shape mismatch: edited mask {img.shape} != FOV {expected_shape}",
        )
    return img.astype(np.int32, copy=False)


def _split_label_image(labels: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for lid in np.unique(labels):
        if int(lid) <= 0:
            continue
        mask = (labels == lid)
        if mask.any():
            out[int(lid)] = mask
    return out


def _greedy_match(
    current: dict[int, np.ndarray],
    new: dict[int, np.ndarray],
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """Greedy IoU matching between two label sets.

    Returns ``(matches, unmatched_current, unmatched_new)`` where ``matches``
    is a list of ``(current_id, new_id, iou)`` tuples sorted by IoU desc.
    """
    pairs: list[tuple[int, int, float]] = []
    for cid, cmask in current.items():
        for nid, nmask in new.items():
            iou = _iou(cmask, nmask)
            if iou >= iou_threshold:
                pairs.append((cid, nid, iou))
    pairs.sort(key=lambda t: t[2], reverse=True)

    used_current: set[int] = set()
    used_new: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for cid, nid, iou in pairs:
        if cid in used_current or nid in used_new:
            continue
        matches.append((cid, nid, iou))
        used_current.add(cid)
        used_new.add(nid)

    unmatched_current = [c for c in current if c not in used_current]
    unmatched_new = [n for n in new if n not in used_new]
    return matches, unmatched_current, unmatched_new


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 0.0
    return inter / union


def _mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """Extract the largest exterior contour as ``[[y, x], ...]``.

    Uses :func:`skimage.measure.find_contours` at the 0.5 isolevel of the
    boolean mask. Multi-region masks return only the largest component's
    contour; downstream replay rasterizes it back to a single ROI.
    """
    from skimage.measure import find_contours

    if mask is None or not mask.any():
        return []
    contours = find_contours(mask.astype(np.uint8), level=0.5)
    if not contours:
        return []
    largest = max(contours, key=len)
    return [[float(y), float(x)] for y, x in largest]
