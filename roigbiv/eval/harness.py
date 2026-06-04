"""Eval harness CLI — score a pipeline output directory against a GT mask.

Usage
-----
python -m roigbiv.eval.harness \\
    --pipeline-dir inference/pipeline/{stem}/ \\
    --gt-masks data/JOSH/ROIGBIV-DATA/cellpose_ready/masks/{stem}_mc_masks.tif \\
    --output experiments/runs/{stem}_{solver}_metrics.json

For external baselines that emit only a mask TIFF (no roi_metadata.json):
python -m roigbiv.eval.harness \\
    --pred-masks experiments/runs/{stem}_cnmfe_masks.tif \\
    --gt-masks ... \\
    --output ...
Activity-type stratification is skipped when roi_metadata.json is absent.

Batch mode — score all FOVs in heldout_fovs.txt:
python -m roigbiv.eval.harness \\
    --batch experiments/harness/heldout_fovs.txt \\
    --pipeline-root experiments/runs/ \\
    --solver ridge \\
    --output experiments/runs/batch_ridge_metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from .diagnostics import load_all_stage_reports
from .match import iou_match
from .metrics import stratified_metrics


def _load_roi_metadata(pipeline_dir: Path) -> dict[int, dict] | None:
    """Return {label_id: metadata_dict} from roi_metadata.json, or None if absent."""
    meta_path = pipeline_dir / "roi_metadata.json"
    if not meta_path.exists():
        return None
    entries: list[dict] = json.loads(meta_path.read_text())
    return {int(e["label_id"]): e for e in entries}


def score_one(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    pred_metadata: dict[int, dict] | None,
    stage_reports: dict[str, dict],
    min_iou: float = 0.3,
) -> dict:
    """Score one FOV. Returns the full metrics dict for that FOV."""
    match = iou_match(gt_masks, pred_masks, min_iou=min_iou)
    detection = stratified_metrics(match, pred_metadata or {})
    detection["iou_threshold"] = min_iou
    detection["activity_stratification"] = pred_metadata is not None
    if pred_metadata is None:
        detection["warnings"].append(
            "No roi_metadata.json — activity-type stratification skipped."
        )
    return {
        "detection": detection,
        "artifact": stage_reports,
        "counts": {
            "gt_rois": match.n_tp + match.n_fn,
            "pred_rois": match.n_tp + match.n_fp,
            "tp": match.n_tp,
            "fp": match.n_fp,
            "fn": match.n_fn,
        },
    }


def _run_single(args: argparse.Namespace) -> dict:
    if args.pred_masks:
        pred_img = tifffile.imread(str(args.pred_masks)).astype(np.uint16)
        pipeline_dir = None
        pred_metadata = None
        stage_reports: dict = {}
    else:
        pipeline_dir = Path(args.pipeline_dir)
        pred_img = tifffile.imread(str(pipeline_dir / "merged_masks.tif")).astype(np.uint16)
        pred_metadata = _load_roi_metadata(pipeline_dir)
        stage_reports = load_all_stage_reports(pipeline_dir)

    gt_img = tifffile.imread(str(args.gt_masks)).astype(np.uint16)
    result = score_one(pred_img, gt_img, pred_metadata, stage_reports)
    result["stem"] = Path(args.gt_masks).stem.replace("_mc_masks", "")
    return result


def _run_batch(args: argparse.Namespace) -> list[dict]:
    fovs_file = Path(args.batch)
    pipeline_root = Path(args.pipeline_root)
    solver = args.solver or "ridge"
    results = []
    for line in fovs_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        stem = parts[0]
        gt_masks_path = Path(parts[2]) if len(parts) >= 3 else None

        pipeline_dir = pipeline_root / f"{stem}_{solver}"
        if not pipeline_dir.exists():
            print(f"[SKIP] {stem}: {pipeline_dir} not found", file=sys.stderr)
            continue
        if gt_masks_path is None or not gt_masks_path.exists():
            print(f"[SKIP] {stem}: GT masks not found", file=sys.stderr)
            continue

        pred_img = tifffile.imread(str(pipeline_dir / "merged_masks.tif")).astype(np.uint16)
        pred_metadata = _load_roi_metadata(pipeline_dir)
        stage_reports = load_all_stage_reports(pipeline_dir)
        gt_img = tifffile.imread(str(gt_masks_path)).astype(np.uint16)
        result = score_one(pred_img, gt_img, pred_metadata, stage_reports)
        result["stem"] = stem
        results.append(result)
        print(f"[OK] {stem}: overall recall={result['detection']['overall']['recall']:.3f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="roigbiv eval harness")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pipeline-dir", metavar="DIR",
                      help="Pipeline output directory (contains merged_masks.tif)")
    mode.add_argument("--pred-masks", metavar="TIF",
                      help="Predicted mask TIFF (for external baselines)")
    mode.add_argument("--batch", metavar="TXT",
                      help="Batch mode: path to heldout_fovs.txt")

    ap.add_argument("--gt-masks", metavar="TIF",
                    help="GT label TIFF (required for single-FOV modes)")
    ap.add_argument("--pipeline-root", metavar="DIR", default="experiments/runs",
                    help="Root directory for batch runs (default: experiments/runs)")
    ap.add_argument("--solver", default="ridge",
                    help="Solver tag for batch mode directory lookup (default: ridge)")
    ap.add_argument("--min-iou", type=float, default=0.3,
                    help="IoU threshold for matching (default: 0.3)")
    ap.add_argument("--output", required=True, metavar="JSON",
                    help="Output JSON path")

    args = ap.parse_args()

    if args.batch:
        results = _run_batch(args)
        payload = {"results": results, "n_fovs": len(results)}
    else:
        if not args.gt_masks:
            ap.error("--gt-masks is required for single-FOV modes")
        payload = _run_single(args)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
