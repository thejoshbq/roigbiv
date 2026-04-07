#!/usr/bin/env python3
"""
ROI G. Biv --- Step 8: Three-branch ROI merge.

Merges masks from Branch A (Cellpose), Branch B (Suite2p), and Branch C
(tonic detection) into a unified ROI set via pairwise IoU matching.

Each merged ROI is assigned a confidence tier based on which branches
detected it (ABC, AB, AC, BC, A, B, or C) and a provenance label.

Outputs per FOV:
  {stem}_merged_masks.tif   — uint16 labeled merged mask
  {stem}_merge_records.csv  — per-ROI: tier, source branches, IoU scores

Batch output:
  merge_summary.csv         — all FOVs combined

Usage:
  python merge_rois.py --s2p_dir suite2p_workspace/output
  python merge_rois.py --iou_threshold 0.4 --s2p_min_prob 0.3
  python merge_rois.py --dry-run
"""

import argparse
import logging
from pathlib import Path

import sys
from config import BASE_DIR, load_config
sys.path.insert(0, str(BASE_DIR))
from roigbiv.merge import merge_batch

S2P_OUT = BASE_DIR / "suite2p_workspace" / "output"
BRANCH_A_OUT = BASE_DIR / "inference" / "output"
BRANCH_C_OUT = BASE_DIR / "inference" / "tonic"
MERGED_OUT = BASE_DIR / "inference" / "merged"
LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("merge_rois")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to console and logs/merge.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "merge.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)
    # Also configure the library logger
    lib_log = logging.getLogger("merge")
    lib_log.setLevel(logging.INFO)
    lib_log.addHandler(fh)
    lib_log.addHandler(ch)


# ---------------------------------------------------------------------------
# FOV discovery
# ---------------------------------------------------------------------------

def discover_stems(
    branch_a_dir: Path,
    s2p_dir: Path,
    branch_c_dir: Path,
) -> list[dict]:
    """Discover FOV stems across all three branches.

    A FOV is included if it appears in at least one branch.
    Returns list of dicts with keys: stem, has_a, has_b, has_c.
    """
    stems_a = set()
    if branch_a_dir.exists():
        for p in branch_a_dir.glob("*_masks.tif"):
            stems_a.add(p.stem.replace("_masks", ""))

    stems_b = set()
    if s2p_dir.exists():
        for d in s2p_dir.iterdir():
            if d.is_dir():
                plane = d / "suite2p" / "plane0"
                if (plane / "stat.npy").exists():
                    stems_b.add(d.name)

    stems_c = set()
    if branch_c_dir.exists():
        for p in branch_c_dir.glob("*_tonic_masks.tif"):
            stems_c.add(p.stem.replace("_tonic_masks", ""))

    all_stems = sorted(stems_a | stems_b | stems_c)
    results = []
    for stem in all_stems:
        results.append({
            "stem": stem,
            "has_a": stem in stems_a,
            "has_b": stem in stems_b,
            "has_c": stem in stems_c,
        })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Step 8: Three-branch ROI merge (Cellpose + Suite2p + tonic).")

    ap.add_argument("--s2p_dir", default=None,
                    help=f"Suite2p output directory (default: {S2P_OUT})")
    ap.add_argument("--branch_a_dir", default=None,
                    help=f"Branch A (Cellpose) mask directory (default: {BRANCH_A_OUT})")
    ap.add_argument("--branch_c_dir", default=None,
                    help=f"Branch C (tonic) mask directory (default: {BRANCH_C_OUT})")
    ap.add_argument("--output_dir", default=None,
                    help=f"Output directory for merged masks (default: {MERGED_OUT})")

    ap.add_argument("--iou_threshold", type=float, default=None,
                    help="IoU threshold for matching (default: 0.3 from config)")
    ap.add_argument("--s2p_min_prob", type=float, default=None,
                    help="Minimum Suite2p iscell probability for Branch B (default: 0.0)")
    ap.add_argument("--mean_image", default=None,
                    help="Directory of per-FOV mean images ({stem}_mean.tif); "
                         "falls back to Suite2p ops['meanImg'] if omitted")
    ap.add_argument("--no_require_spatial_support", action="store_false",
                    dest="require_spatial_support",
                    help="Disable mean-projection intensity gate for Branch C-only ROIs")
    ap.set_defaults(require_spatial_support=True)

    ap.add_argument("--config", default=None,
                    help="Path to pipeline YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Discover FOVs and print plan without processing")

    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    mc = cfg.get("merge", {})
    paths_cfg = cfg.get("paths", {})

    # Resolve directories
    s2p_dir = Path(args.s2p_dir) if args.s2p_dir else \
        BASE_DIR / paths_cfg.get("s2p_output", "suite2p_workspace/output")
    branch_a_dir = Path(args.branch_a_dir) if args.branch_a_dir else \
        BASE_DIR / paths_cfg.get("inference_output", "inference/output")
    branch_c_dir = Path(args.branch_c_dir) if args.branch_c_dir else \
        BASE_DIR / paths_cfg.get("tonic_output", "inference/tonic")
    out_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / paths_cfg.get("merged_output", "inference/merged")

    # Resolve parameters: CLI > config > defaults
    iou_threshold = args.iou_threshold if args.iou_threshold is not None \
        else mc.get("iou_threshold", 0.3)
    s2p_min_prob = args.s2p_min_prob if args.s2p_min_prob is not None \
        else mc.get("s2p_min_prob", 0.0)
    mean_image_dir = Path(args.mean_image) if args.mean_image else None
    require_spatial_support = args.require_spatial_support

    # Print plan
    log.info("Step 8 — Three-branch ROI merge")
    log.info("  Branch A (Cellpose): %s", branch_a_dir)
    log.info("  Branch B (Suite2p):  %s", s2p_dir)
    log.info("  Branch C (tonic):    %s", branch_c_dir)
    log.info("  Output:              %s", out_dir)
    log.info("  IoU threshold:       %.2f", iou_threshold)
    log.info("  S2p min prob:        %.2f", s2p_min_prob)
    log.info("  Spatial support:     %s", require_spatial_support)

    # Discover FOVs
    fov_info = discover_stems(branch_a_dir, s2p_dir, branch_c_dir)
    n_total = len(fov_info)
    n_a = sum(1 for f in fov_info if f["has_a"])
    n_b = sum(1 for f in fov_info if f["has_b"])
    n_c = sum(1 for f in fov_info if f["has_c"])
    log.info("  FOVs discovered: %d total (A=%d, B=%d, C=%d)", n_total, n_a, n_b, n_c)

    if args.dry_run:
        for f in fov_info:
            branches = []
            if f["has_a"]:
                branches.append("A")
            if f["has_b"]:
                branches.append("B")
            if f["has_c"]:
                branches.append("C")
            log.info("    %s: %s", f["stem"], "+".join(branches))
        log.info("DRY RUN — exiting without processing")
        return

    if n_total == 0:
        log.error("No FOVs found across any branch")
        return

    stems = [f["stem"] for f in fov_info]
    merge_batch(
        stems, branch_a_dir, s2p_dir, branch_c_dir, out_dir,
        iou_threshold=iou_threshold,
        s2p_min_prob=s2p_min_prob,
        mean_image_dir=mean_image_dir,
        require_spatial_support=require_spatial_support,
    )


if __name__ == "__main__":
    main()
