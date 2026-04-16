"""
ROI G. Biv pipeline — Final Output Assembly (spec §17.1).

Persists everything downstream analysis depends on:
  F.npy, Fneu.npy, F_corrected.npy, dFF.npy, spks.npy, F_bandpass.npy,
  F_bandpass_index.npy, roi_metadata.json, pipeline_log.json.

ROI ordering contract: row K of every trace array corresponds to the ROI
at position K in `rois_sorted` (sorted by label_id). The same ROI's label
appears at the matching integer value in merged_masks.tif.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from roigbiv.pipeline.types import FOVData, ROI, PipelineConfig


def _jsonable(obj):
    """Recursively coerce numpy types, arrays, and Paths to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_pipeline_outputs(
    fov: FOVData,
    rois_sorted: list[ROI],
    F_raw: np.ndarray,
    F_neu: np.ndarray,
    F_corrected: np.ndarray,
    dFF: np.ndarray,
    spks: np.ndarray,
    review_queue: list[dict],
    overlap_groups: list[list[int]],
    output_dir: Path,
    cfg: PipelineConfig,
    stage_timings: dict,
    warnings: list,
    subtraction_summary: dict,
) -> None:
    """Write every final output file.

    Parameters
    ----------
    rois_sorted : ROI list already sorted by label_id. This ordering defines
                  the row contract for every trace array.
    overlap_groups : result of find_overlap_groups, for the pipeline log.
    subtraction_summary : per-stage subtraction pass/fail counts (already
                  collected in run.py; passed through).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Trace arrays ─────────────────────────────────────────────────────
    np.save(str(output_dir / "F.npy"), F_raw.astype(np.float32))
    np.save(str(output_dir / "Fneu.npy"), F_neu.astype(np.float32))
    np.save(str(output_dir / "F_corrected.npy"), F_corrected.astype(np.float32))
    np.save(str(output_dir / "dFF.npy"), dFF.astype(np.float32))
    np.save(str(output_dir / "spks.npy"), spks.astype(np.float32))

    # ── F_bandpass (tonic only) + index mapping ──────────────────────────
    tonic_rows = []
    tonic_labels = []
    for i, roi in enumerate(rois_sorted):
        if roi.activity_type == "tonic":
            bp = roi.features.get("trace_bandpass")
            if bp is not None:
                tonic_rows.append(np.asarray(bp, dtype=np.float32))
                tonic_labels.append(int(roi.label_id))
    if tonic_rows:
        F_bandpass = np.stack(tonic_rows)
    else:
        T = F_raw.shape[1] if F_raw.ndim == 2 else 0
        F_bandpass = np.zeros((0, T), dtype=np.float32)
    np.save(str(output_dir / "F_bandpass.npy"), F_bandpass)
    np.save(str(output_dir / "F_bandpass_index.npy"),
            np.array(tonic_labels, dtype=np.int32))

    # ── roi_metadata.json ────────────────────────────────────────────────
    priority_map = {int(entry["label_id"]): int(entry["priority"])
                    for entry in review_queue}
    meta_list = []
    for i, roi in enumerate(rois_sorted):
        d = roi.to_serializable()
        d["row_index"] = int(i)
        d["review_priority"] = priority_map.get(int(roi.label_id))
        meta_list.append(d)
    (output_dir / "roi_metadata.json").write_text(
        json.dumps(_jsonable(meta_list), indent=2)
    )

    # ── pipeline_log.json ────────────────────────────────────────────────
    activity_counts = {k: 0 for k in
                       ("phasic", "sparse", "tonic", "silent", "ambiguous")}
    for roi in rois_sorted:
        k = roi.activity_type or "ambiguous"
        activity_counts[k] = activity_counts.get(k, 0) + 1

    review_counts = {"priority_1": 0, "priority_2": 0,
                     "priority_3": 0, "priority_4": 0}
    for entry in review_queue:
        review_counts[f"priority_{entry['priority']}"] += 1

    log = {
        "input": str(fov.raw_path),
        "output_dir": str(output_dir),
        "fov_name": fov.raw_path.stem.replace("_mc", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shape": list(fov.shape),
        "k_background": int(fov.k_background),
        "config": cfg.summary_for_log(),
        "stage_counts": fov.stage_counts,
        "subtraction": subtraction_summary,
        "activity_type_counts": activity_counts,
        "overlap_groups": {
            "n_groups": int(len(overlap_groups)),
            "group_sizes": [int(len(g)) for g in overlap_groups],
            "groups": [[int(i) for i in g] for g in overlap_groups],
        },
        "total_rois": int(len(rois_sorted)),
        "review_queue_summary": {
            "total": int(len(review_queue)),
            **review_counts,
        },
        "timings_s": stage_timings,
        "warnings": list(warnings),
    }
    (output_dir / "pipeline_log.json").write_text(
        json.dumps(_jsonable(log), indent=2)
    )


def print_final_summary(
    fov: FOVData,
    review_queue: list[dict],
    output_dir: Path,
) -> None:
    """Print the end-of-pipeline summary block."""
    counts = {k: 0 for k in
              ("phasic", "sparse", "tonic", "silent", "ambiguous")}
    for roi in fov.rois:
        k = roi.activity_type or "ambiguous"
        counts[k] = counts.get(k, 0) + 1
    total = len(fov.rois)

    p1 = sum(1 for e in review_queue if e["priority"] == 1)
    p2 = sum(1 for e in review_queue if e["priority"] == 2)
    p3 = sum(1 for e in review_queue if e["priority"] == 3)
    p_actionable = p1 + p2 + p3

    stem = fov.raw_path.stem.replace("_mc", "")
    print("\n=== ROI G. Biv — Pipeline Complete ===", flush=True)
    print(f"FOV: {stem}", flush=True)
    print(
        f"Total ROIs: {total} "
        f"({counts['phasic']} phasic, {counts['sparse']} sparse, "
        f"{counts['tonic']} tonic, {counts['silent']} silent, "
        f"{counts['ambiguous']} ambiguous)", flush=True,
    )
    print(f"Review queue: {p_actionable} ROIs flagged for HITL review",
          flush=True)
    print(f"  Priority 1 (tonic review): {p1}", flush=True)
    print(f"  Priority 2 (flagged):      {p2}", flush=True)
    print(f"  Priority 3 (single-event): {p3}", flush=True)
    print(f"Outputs saved to: {output_dir}", flush=True)
    print(f"HITL staging ready at: {output_dir}/hitl_staging/", flush=True)
    print("\nNext step: Open merged_masks.tif in Cellpose GUI for review.",
          flush=True)
