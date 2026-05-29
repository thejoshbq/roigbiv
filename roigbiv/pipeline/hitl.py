"""
ROI G. Biv pipeline — HITL Review Queue + Export Package (spec §14).

Does NOT build a GUI. Produces the data and visualization layers that make
a human reviewer's manual pass (in the Cellpose GUI) effective and
counteract phasic confirmation bias (Blindspot 13).

Priority tiers:
  1. Stage 4 candidates (confidence="requires_review")
     — sorted by corr_contrast ASCending (most uncertain first)
     — bandpass trace is PRIMARY evidence (NOT raw)
  2. Flagged ROIs from any stage (confidence="moderate")
     — sorted by source_stage DESCending
  3. Stage 3 ROIs with event_count==1 (confidence="low")
  4. Everything else (informational — for false-negative search)

Package layout:
  {output_dir}/
    review_queue.json
    merged_masks.tif              (uint16; label IDs preserved)
    hitl/
      stage4/{label_id}/
        bandpass_trace.npy        ← PRIMARY evidence
        corr_contrast_crop.npy
        info.json
      stage3/{label_id}/
        event_frame_indices.json
    hitl_staging/                 (Cellpose GUI training layout)
      images/{stem}.tif           (copy of mean_S)
      masks/{stem}_seg.tif        (copy of merged_masks)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline.types import FOVData, ROI


# ─────────────────────────────────────────────────────────────────────────
# Review queue
# ─────────────────────────────────────────────────────────────────────────

def _reason_for(roi: ROI, priority: int) -> str:
    if priority == 1:
        return "Stage 4 tonic candidate — review with BANDPASS trace (not raw)."
    if priority == 2:
        return f"Flagged by Gate {roi.source_stage} (moderate confidence)."
    if priority == 3:
        return "Stage 3 single-event detection — confirm event is real."
    return "Informational — scan for false negatives."


def build_review_queue(rois: list[ROI]) -> list[dict]:
    """Assemble the prioritized review queue.

    Returns
    -------
    List of dicts, one per non-rejected ROI:
      {roi_index, label_id, priority, reason, activity_type, source_stage,
       confidence, corr_contrast, event_count}
    Ordered by priority (1 first), then by priority-specific secondary key.
    """
    # Partition into tiers
    p1, p2, p3, p4 = [], [], [], []
    for idx, roi in enumerate(rois):
        if roi.gate_outcome == "reject":
            continue
        if roi.source_stage == 4 and roi.confidence == "requires_review":
            p1.append(idx)
        elif roi.confidence == "moderate":
            p2.append(idx)
        elif (roi.source_stage == 3
              and (roi.event_count == 1 or roi.confidence == "low")):
            p3.append(idx)
        else:
            p4.append(idx)

    # Sort within tiers
    p1.sort(key=lambda i: (rois[i].corr_contrast or float("inf")))  # ascending
    p2.sort(key=lambda i: -int(rois[i].source_stage))               # descending
    p3.sort(key=lambda i: int(rois[i].label_id))
    p4.sort(key=lambda i: int(rois[i].label_id))

    queue = []
    for tier, indices in enumerate([p1, p2, p3, p4], start=1):
        for idx in indices:
            roi = rois[idx]
            queue.append({
                "roi_index": int(idx),
                "label_id": int(roi.label_id),
                "priority": int(tier),
                "reason": _reason_for(roi, tier),
                "activity_type": roi.activity_type,
                "source_stage": int(roi.source_stage),
                "confidence": roi.confidence,
                "corr_contrast": (None if roi.corr_contrast is None
                                  else float(roi.corr_contrast)),
                "event_count": (None if roi.event_count is None
                                else int(roi.event_count)),
            })
    return queue


# ─────────────────────────────────────────────────────────────────────────
# Export
# ─────────────────────────────────────────────────────────────────────────

def _crop_around_centroid(image: np.ndarray, mask: np.ndarray,
                          pad: int = 30) -> np.ndarray:
    """Return a (≤2pad+1, ≤2pad+1) crop of `image` centered on mask centroid."""
    H, W = image.shape
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((1, 1), dtype=image.dtype)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0, y1 = max(0, cy - pad), min(H, cy + pad + 1)
    x0, x1 = max(0, cx - pad), min(W, cx + pad + 1)
    return image[y0:y1, x0:x1].copy()


def _to_jsonable(obj):
    """JSON-safe conversion for numpy scalars/arrays in info blobs."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def export_hitl_package(
    fov: FOVData,
    rois: list[ROI],
    review_queue: list[dict],
    output_dir: Path,
) -> None:
    """Write all files the reviewer needs to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. review_queue.json
    (output_dir / "review_queue.json").write_text(
        json.dumps(review_queue, indent=2)
    )

    # 2. merged_masks.tif — uint16 label image of every non-rejected ROI
    H, W = fov.shape[1], fov.shape[2]
    merged = np.zeros((H, W), dtype=np.uint16)
    for roi in rois:
        if roi.gate_outcome != "reject":
            merged[roi.mask] = int(roi.label_id)
    tifffile.imwrite(str(output_dir / "merged_masks.tif"), merged)

    # 3. Priority-1 Stage 4 materials
    hitl_dir = output_dir / "hitl"
    (hitl_dir / "stage4").mkdir(parents=True, exist_ok=True)
    (hitl_dir / "stage3").mkdir(parents=True, exist_ok=True)

    # Pick a correlation contrast map to crop from — prefer the window with
    # the highest score on the ROI, fall back to "medium".
    def _best_cmap_for(roi):
        if not fov.corr_contrast_maps:
            return None, None
        best_key, best_val = None, -np.inf
        for key, cmap in fov.corr_contrast_maps.items():
            val = float(cmap[roi.mask].mean()) if roi.mask.any() else 0.0
            if val > best_val:
                best_val, best_key = val, key
        return best_key, fov.corr_contrast_maps.get(best_key)

    for entry in review_queue:
        if entry["priority"] != 1:
            continue
        roi = rois[entry["roi_index"]]
        roi_dir = hitl_dir / "stage4" / str(roi.label_id)
        roi_dir.mkdir(parents=True, exist_ok=True)

        # PRIMARY evidence: bandpass trace (Blindspot 13)
        bandpass = roi.features.get("trace_bandpass")
        if bandpass is not None:
            np.save(str(roi_dir / "bandpass_trace.npy"),
                    np.asarray(bandpass, dtype=np.float32))

        # Correlation contrast crop around centroid
        window_name, cmap = _best_cmap_for(roi)
        if cmap is not None:
            crop = _crop_around_centroid(cmap, roi.mask, pad=30)
            np.save(str(roi_dir / "corr_contrast_crop.npy"),
                    crop.astype(np.float32))

        (roi_dir / "info.json").write_text(json.dumps(_to_jsonable({
            "label_id": int(roi.label_id),
            "corr_contrast": (None if roi.corr_contrast is None
                              else float(roi.corr_contrast)),
            "best_contrast_window": window_name,
            "activity_type": roi.activity_type,
            "area": int(roi.area),
            "centroid": [roi.features.get("centroid_y", 0.0),
                         roi.features.get("centroid_x", 0.0)],
            "notes": "Review using bandpass_trace.npy (raw traces mislead).",
        }), indent=2))

    # 4. Priority-3 Stage 3 single-event materials
    T = fov.shape[0]
    for entry in review_queue:
        if entry["priority"] != 3:
            continue
        roi = rois[entry["roi_index"]]
        roi_dir = hitl_dir / "stage3" / str(roi.label_id)
        roi_dir.mkdir(parents=True, exist_ok=True)

        events = roi.features.get("picked_events") or roi.features.get("events") or []
        # events may be a list of (t, score) tuples or ints; accept both
        frames = []
        for ev in events:
            if isinstance(ev, (list, tuple)) and len(ev) > 0:
                frames.append(int(ev[0]))
            else:
                try:
                    frames.append(int(ev))
                except Exception:
                    pass
        windows = [[max(0, f - 10), min(T - 1, f + 10)] for f in frames]

        (roi_dir / "event_frame_indices.json").write_text(json.dumps({
            "label_id": int(roi.label_id),
            "event_frames": frames,
            "windows": windows,
        }, indent=2))

    # 5. Cellpose-compatible training staging
    stem = fov.raw_path.stem.replace("_mc", "")
    staging = output_dir / "hitl_staging"
    (staging / "images").mkdir(parents=True, exist_ok=True)
    (staging / "masks").mkdir(parents=True, exist_ok=True)

    # Image source: prefer mean_M (morphological), fall back to mean_S
    img = fov.mean_M if fov.mean_M is not None else fov.mean_S
    if img is not None:
        tifffile.imwrite(str(staging / "images" / f"{stem}.tif"),
                         img.astype(np.float32))
    tifffile.imwrite(str(staging / "masks" / f"{stem}_seg.tif"), merged)
