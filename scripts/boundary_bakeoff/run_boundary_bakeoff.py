"""Boundary bake-off CLI — free Cellpose vs. disk stamps vs. seeded boundaries.

Answers the question ADR-0003 left open and this change reopens: does
conditioning mask formation on confirmed centroids actually draw better
boundaries than either the model alone or the fixed disk that replaced it?

Ground truth is the same hand-drawn ImageJ RoiSets the centroid bake-off uses
(``scripts/centroid_bakeoff/ground_truth.py``), rasterized to label images
instead of reduced to points.

Seeds default to the GT centroids themselves. That is the *ceiling* case — it
measures how good the boundary can be when the centroid is right, which is the
question this change is actually about; centroid localization already has its
own bake-off and is not re-measured here. Pass ``--seeds detected`` to seed off
Cellpose's own centroids instead, for the end-to-end number.

Examples
--------
Default (GT-seeded ceiling)::

    conda run -n roigbiv python scripts/boundary_bakeoff/run_boundary_bakeoff.py \\
        --real-fov-dir data/BEGINNER_ROIS/LM_RoiSets/LM_RoiSets/TDT4_ENSURESA

End-to-end, seeded by the detector's own centroids::

    conda run -n roigbiv python scripts/boundary_bakeoff/run_boundary_bakeoff.py \\
        --real-fov-dir data/... --seeds detected

Outputs a JSON report under experiments/runs/boundary_bakeoff/. Nothing is
written to inference/.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.boundary_bakeoff import arms as arms_mod          # noqa: E402
from scripts.boundary_bakeoff.ground_truth import (            # noqa: E402
    discover_pairs,
    imagej_roiset_to_labels,
    label_centroids,
)
from scripts.boundary_bakeoff.score import score_arm           # noqa: E402

log = logging.getLogger("boundary_bakeoff")

ARM_NAMES = ("free_cellpose", "disk_stamps", "seeded")


def _mean_projection(mc_tif: Path, max_frames: int = 2000) -> np.ndarray:
    """Anatomical mean of a motion-corrected stack — centroids.py's convention."""
    import tifffile

    with tifffile.TiffFile(str(mc_tif)) as tif:
        n = len(tif.pages)
        if n == 1:
            return np.asarray(tif.pages[0].asarray(), dtype=np.float32)
        idx = np.unique(np.linspace(0, n - 1, min(n, max_frames)).astype(int))
        total = np.zeros(tif.pages[0].shape, dtype=np.float64)
        for i in idx:
            total += np.asarray(tif.pages[int(i)].asarray(), dtype=np.float64)
    return (total / len(idx)).astype(np.float32)


def _detection_cfg(args):
    """Centroid discovery's pinned substrate — see centroids.py for why."""
    from roigbiv.pipeline.types import PipelineConfig

    return PipelineConfig(
        cellpose_model=args.model,
        diameter=int(args.diameter), diameter_auto=False,
        channels=(0, 0), tile_norm_blocksize=0, use_denoise=False,
        cellprob_threshold=float(args.cellprob_threshold),
        flow_threshold=float(args.flow_threshold),
        force_cpu=bool(args.cpu),
        boundary_min_area=int(args.min_area),
        boundary_max_area=args.max_area,
    )


def _detected_seeds(flows) -> dict[int, tuple[float, float]]:
    """Centroids of Cellpose's own masks, the same way centroids.py takes them."""
    from scipy.ndimage import center_of_mass

    labels = np.asarray(flows.label_image)
    ids = [int(i) for i in np.unique(labels) if i != 0]
    if not ids:
        return {}
    coms = center_of_mass(labels > 0, labels, ids)
    return {n: (float(c[0]), float(c[1])) for n, c in enumerate(coms, start=1)}


def run_fov(mc_tif: Path, roi_zip: Path, stem: str, args) -> list[dict]:
    morph = _mean_projection(mc_tif)
    gt_labels, gt_names = imagej_roiset_to_labels(roi_zip, morph.shape)
    n_gt = len(gt_names)
    if n_gt == 0:
        log.warning("%s: no usable ground-truth boundaries; skipping", stem)
        return []

    cfg = _detection_cfg(args)
    t0 = time.time()
    flows = arms_mod._detect(morph, cfg)
    log.info("%s: %d GT cell(s), inference in %.1fs", stem, n_gt, time.time() - t0)

    seeds = (label_centroids(gt_labels) if args.seeds == "gt"
             else _detected_seeds(flows))
    if not seeds:
        log.warning("%s: no seeds under --seeds=%s; skipping", stem, args.seeds)
        return []

    capture = float(args.capture_px if args.capture_px is not None
                    else args.diameter / 2.0)
    radius = int(args.stamp_radius)

    built = [
        arms_mod.free_cellpose(flows),
        arms_mod.disk_stamps(seeds, morph.shape, radius),
        arms_mod.seeded(seeds, morph.shape, flows, cfg,
                        capture_px=capture, fallback_radius=radius),
    ]
    rows = []
    for arm in built:
        if arm.name not in args.arms:
            continue
        score = score_arm(arm, gt_labels, stem, seeds, min_iou=args.min_iou)
        rows.append(score.to_dict())
        log.info("  %-14s tp=%-3d fp=%-3d fn=%-3d meanIoU=%.3f seededIoU=%s",
                 arm.name, score.n_tp, score.n_fp, score.n_fn, score.mean_iou,
                 "n/a" if score.mean_iou_seeded is None
                 else f"{score.mean_iou_seeded:.3f}")
    return rows


def summarize(rows: list[dict]) -> dict:
    """Per-arm aggregate across FOVs. Means are over FOVs, not over cells."""
    out: dict = {}
    for name in ARM_NAMES:
        arm_rows = [r for r in rows if r["arm"] == name]
        if not arm_rows:
            continue
        seeded = [r["mean_iou_seeded"] for r in arm_rows
                  if r["mean_iou_seeded"] is not None]
        flow = [r["mean_iou_flow"] for r in arm_rows
                if r["mean_iou_flow"] is not None]
        out[name] = {
            "n_fovs": len(arm_rows),
            "n_gt": sum(r["n_gt"] for r in arm_rows),
            "n_tp": sum(r["n_tp"] for r in arm_rows),
            "n_fp": sum(r["n_fp"] for r in arm_rows),
            "n_fn": sum(r["n_fn"] for r in arm_rows),
            "mean_iou": round(float(np.mean([r["mean_iou"] for r in arm_rows])), 4),
            "mean_iou_seeded": (round(float(np.mean(seeded)), 4) if seeded else None),
            "mean_iou_flow": (round(float(np.mean(flow)), 4) if flow else None),
            "n_flow": sum(r["n_flow"] for r in arm_rows),
            "n_disk_fallback": sum(r["n_disk_fallback"] for r in arm_rows),
        }
        tp, fp, fn = out[name]["n_tp"], out[name]["n_fp"], out[name]["n_fn"]
        out[name]["precision"] = round(tp / (tp + fp), 4) if tp + fp else None
        out[name]["recall"] = round(tp / (tp + fn), 4) if tp + fn else None
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-fov-dir", action="append", type=Path, required=True,
                   help="directory to search for *_mc.tif + *_RoiSet*.zip pairs")
    p.add_argument("--exclude", action="append", default=[],
                   help="substring to exclude from discovered paths")
    p.add_argument("--arms", nargs="+", default=list(ARM_NAMES), choices=ARM_NAMES)
    p.add_argument("--seeds", choices=("gt", "detected"), default="gt",
                   help="gt = ceiling case (default); detected = end-to-end")
    p.add_argument("--model", default="cyto3")
    p.add_argument("--diameter", type=float, default=30.0)
    p.add_argument("--cellprob-threshold", type=float, default=-2.0)
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--capture-px", type=float, default=None,
                   help="default: diameter/2")
    p.add_argument("--stamp-radius", type=int, default=8,
                   help="disk radius for the stamps arm and the seeded fallback")
    p.add_argument("--min-area", type=int, default=0)
    p.add_argument("--max-area", type=int, default=None)
    p.add_argument("--min-iou", type=float, default=0.3,
                   help="IoU for a GT/pred pair to count as matched")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-dir", type=Path,
                   default=_REPO / "experiments" / "runs" / "boundary_bakeoff")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pairs = discover_pairs(args.real_fov_dir, args.exclude)
    if not pairs:
        log.error("no *_mc.tif + RoiSet pairs found under %s", args.real_fov_dir)
        return 1
    log.info("%d FOV(s) discovered", len(pairs))

    rows: list[dict] = []
    for mc_tif, roi_zip, stem in pairs:
        try:
            rows.extend(run_fov(mc_tif, roi_zip, stem, args))
        except Exception as exc:  # noqa: BLE001 — one bad FOV must not end the run
            log.error("%s: %s: %s", stem, type(exc).__name__, exc)

    if not rows:
        log.error("no FOV produced a score")
        return 1

    summary = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = args.out_dir / "boundary_bakeoff.json"
    report.write_text(json.dumps({
        "seeds": args.seeds,
        "min_iou": args.min_iou,
        "params": {
            "model": args.model, "diameter": args.diameter,
            "cellprob_threshold": args.cellprob_threshold,
            "flow_threshold": args.flow_threshold,
            "capture_px": args.capture_px, "stamp_radius": args.stamp_radius,
        },
        "summary": summary,
        "per_fov": rows,
    }, indent=2))

    log.info("\n%-14s %8s %8s %9s %10s %9s", "arm", "prec", "recall", "meanIoU",
             "flowIoU", "n_flow")
    for name, s in summary.items():
        log.info("%-14s %8s %8s %9.3f %10s %9s", name,
                 "n/a" if s["precision"] is None else f"{s['precision']:.3f}",
                 "n/a" if s["recall"] is None else f"{s['recall']:.3f}",
                 s["mean_iou"],
                 "n/a" if s["mean_iou_flow"] is None
                 else f"{s['mean_iou_flow']:.3f}",
                 f"{s['n_flow']}/{s['n_gt']}" if s["n_flow"] else "-")
    log.info("\nflowIoU is the only column describing a seeded boundary; the "
             "rest of the seeded arm is the disk fallback.")
    log.info("\nreport: %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
