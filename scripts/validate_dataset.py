#!/usr/bin/env python3
"""
ROI G. Biv — Dataset validation.

Two modes:
  1. Suite2p output validation (default):
     Checks that Suite2p ran successfully and projections were extracted.

  2. Training data validation (--training):
     Checks matched _mean.tif / _masks.tif pairs for Cellpose training.

Usage:
  python validate_dataset.py --s2p_dir suite2p_workspace/output
  python validate_dataset.py --training
  python validate_dataset.py --training --annotated_dir data/annotated --masks_dir data/masks
  python validate_dataset.py --s2p_dir suite2p_workspace/output --dry-run
"""

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import tifffile

from config import BASE_DIR, load_config

LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("validate_dataset")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to console and logs/validate.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "validate.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)


# ---------------------------------------------------------------------------
# Suite2p output validation
# ---------------------------------------------------------------------------

EXPECTED_S2P_FILES = ["stat.npy", "iscell.npy", "F.npy", "Fneu.npy", "ops.npy"]
EXPECTED_OPS_KEYS = ["meanImg", "Vcorr", "max_proj"]


def validate_suite2p_outputs(
    s2p_dir: Path,
    annotated_dir: Path,
    max_proj_dir: Path,
) -> list[dict]:
    """Validate Suite2p outputs exist and contain expected keys.

    Returns list of per-FOV status dicts with keys:
      stem, ok, n_rois, missing_files, missing_ops_keys, missing_projections,
      has_data_bin, mean_diameter
    """
    results = []
    fov_dirs = sorted(d for d in s2p_dir.iterdir() if d.is_dir())

    if not fov_dirs:
        log.warning("No FOV directories found in %s", s2p_dir)
        return results

    for fov_dir in fov_dirs:
        plane_dir = fov_dir / "suite2p" / "plane0"
        stem = fov_dir.name
        entry = {
            "stem": stem,
            "ok": True,
            "n_rois": 0,
            "missing_files": [],
            "missing_ops_keys": [],
            "missing_projections": [],
            "has_data_bin": False,
            "mean_diameter": 0.0,
        }

        # Check expected Suite2p output files
        for fname in EXPECTED_S2P_FILES:
            if not (plane_dir / fname).exists():
                entry["missing_files"].append(fname)
                entry["ok"] = False

        # Check data.bin
        data_bin = plane_dir / "data.bin"
        entry["has_data_bin"] = data_bin.exists()
        if not data_bin.exists():
            entry["missing_files"].append("data.bin")
            entry["ok"] = False

        # Check ops keys
        ops_path = plane_dir / "ops.npy"
        if ops_path.exists():
            ops = np.load(str(ops_path), allow_pickle=True).item()
            for key in EXPECTED_OPS_KEYS:
                if key not in ops or ops[key] is None:
                    entry["missing_ops_keys"].append(key)
                    entry["ok"] = False

        # Check extracted projections
        proj_checks = [
            (annotated_dir / f"{stem}_mean.tif", "mean"),
            (annotated_dir / f"{stem}_vcorr.tif", "vcorr"),
            (max_proj_dir / f"{stem}_max.tif", "max"),
        ]
        for path, name in proj_checks:
            if not path.exists():
                entry["missing_projections"].append(name)

        # Count ROIs and compute mean diameter from stat
        stat_path = plane_dir / "stat.npy"
        if stat_path.exists():
            stat = np.load(str(stat_path), allow_pickle=True)
            entry["n_rois"] = len(stat)
            if len(stat) > 0:
                areas = [len(s["ypix"]) for s in stat]
                diameters = [2.0 * math.sqrt(a / math.pi) for a in areas]
                entry["mean_diameter"] = sum(diameters) / len(diameters)

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# Training data validation
# ---------------------------------------------------------------------------

def validate_training_data(
    annotated_dir: Path,
    masks_dir: Path,
) -> list[dict]:
    """Validate matched _mean.tif / _masks.tif pairs for Cellpose training.

    Returns list of per-FOV status dicts with keys:
      stem, ok, n_neurons, mean_diameter, issues
    """
    results = []
    mask_files = sorted(masks_dir.glob("*_masks.tif"))

    if not mask_files:
        log.warning("No *_masks.tif files found in %s", masks_dir)
        return results

    seen_stems = set()
    for mask_path in mask_files:
        stem = mask_path.stem.replace("_masks", "")
        seen_stems.add(stem)
        entry = {
            "stem": stem,
            "ok": True,
            "n_neurons": 0,
            "mean_diameter": 0.0,
            "issues": [],
        }

        # Check matching mean image
        mean_path = annotated_dir / f"{stem}_mean.tif"
        if not mean_path.exists():
            entry["issues"].append(f"Missing image: {stem}_mean.tif in {annotated_dir}")
            entry["ok"] = False
            results.append(entry)
            continue

        # Read and validate
        mask = tifffile.imread(str(mask_path))
        img = tifffile.imread(str(mean_path))

        # Shape check
        if img.shape[-2:] != mask.shape[-2:]:
            entry["issues"].append(
                f"Shape mismatch: image {img.shape} vs mask {mask.shape}")
            entry["ok"] = False

        # Empty mask check
        if mask.max() == 0:
            entry["issues"].append("Empty mask (0 neurons)")
            entry["ok"] = False
        else:
            roi_ids = np.unique(mask[mask > 0])
            entry["n_neurons"] = len(roi_ids)
            areas = [(mask == roi_id).sum() for roi_id in roi_ids]
            diameters = [2.0 * math.sqrt(a / math.pi) for a in areas]
            entry["mean_diameter"] = sum(diameters) / len(diameters)

        results.append(entry)

    # Check for orphaned mean images (no matching mask)
    for mean_path in sorted(annotated_dir.glob("*_mean.tif")):
        stem = mean_path.stem.replace("_mean", "")
        if stem not in seen_stems:
            log.info("  Note: %s_mean.tif has no matching mask (not an error)", stem)

    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_suite2p_report(results: list[dict]) -> bool:
    """Print Suite2p validation report. Returns True if all OK."""
    all_ok = True
    total_rois = 0
    diameters = []

    for r in results:
        status = "OK" if r["ok"] else "ISSUES"
        log.info("  %-30s %s  %4d ROIs  diam=%.1f px  data.bin=%s",
                 r["stem"], status, r["n_rois"], r["mean_diameter"],
                 "yes" if r["has_data_bin"] else "MISSING")
        total_rois += r["n_rois"]
        if r["mean_diameter"] > 0:
            diameters.append(r["mean_diameter"])

        if r["missing_files"]:
            log.warning("    Missing files: %s", ", ".join(r["missing_files"]))
        if r["missing_ops_keys"]:
            log.warning("    Missing ops keys: %s", ", ".join(r["missing_ops_keys"]))
        if r["missing_projections"]:
            log.info("    Missing projections: %s", ", ".join(r["missing_projections"]))

        if not r["ok"]:
            all_ok = False

    log.info("")
    log.info("Summary: %d FOVs, %d total ROIs", len(results), total_rois)
    if diameters:
        log.info("  Mean diameter across FOVs: %.1f px", sum(diameters) / len(diameters))

    n_ok = sum(1 for r in results if r["ok"])
    n_issues = len(results) - n_ok
    if n_issues:
        log.warning("  %d FOV(s) with issues", n_issues)
    else:
        log.info("  All FOVs validated successfully")

    return all_ok


def print_training_report(results: list[dict]) -> bool:
    """Print training data validation report. Returns True if all OK."""
    all_ok = True
    total_neurons = 0
    diameters = []

    for r in results:
        status = "OK" if r["ok"] else "ISSUES"
        log.info("  %-30s %s  %4d neurons  diam=%.1f px",
                 r["stem"], status, r["n_neurons"], r["mean_diameter"])
        total_neurons += r["n_neurons"]
        if r["mean_diameter"] > 0:
            diameters.append(r["mean_diameter"])

        for issue in r["issues"]:
            log.warning("    %s", issue)

        if not r["ok"]:
            all_ok = False

    log.info("")
    log.info("Summary: %d FOVs, %d total neurons", len(results), total_neurons)
    if diameters:
        log.info("  Mean diameter across FOVs: %.1f px", sum(diameters) / len(diameters))

    n_ok = sum(1 for r in results if r["ok"])
    n_issues = len(results) - n_ok
    if n_issues:
        log.warning("  %d FOV(s) with issues", n_issues)
    else:
        log.info("  All pairs validated successfully")

    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Validate Suite2p outputs or training data for the pipeline.")

    ap.add_argument("--training", action="store_true",
                    help="Validate training data (mean/mask pairs) instead of Suite2p output")

    # Suite2p validation args
    ap.add_argument("--s2p_dir", default=None,
                    help="Suite2p output directory (default: suite2p_workspace/output)")

    # Training validation args
    ap.add_argument("--annotated_dir", default=None,
                    help="Directory with *_mean.tif images (default: data/annotated)")
    ap.add_argument("--masks_dir", default=None,
                    help="Directory with *_masks.tif files (default: data/masks)")
    ap.add_argument("--max_proj_dir", default=None,
                    help="Directory with *_max.tif files (default: data/max_projections)")

    ap.add_argument("--config", default=None,
                    help="Path to pipeline YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be validated without reading files")

    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})

    # Resolve directories from args > config > defaults
    annotated_dir = Path(args.annotated_dir) if args.annotated_dir else \
        BASE_DIR / paths_cfg.get("annotated_dir", "data/annotated")
    masks_dir = Path(args.masks_dir) if args.masks_dir else \
        BASE_DIR / paths_cfg.get("masks_dir", "data/masks")
    max_proj_dir = Path(args.max_proj_dir) if args.max_proj_dir else \
        BASE_DIR / paths_cfg.get("max_projections_dir", "data/max_projections")
    s2p_dir = Path(args.s2p_dir) if args.s2p_dir else \
        BASE_DIR / paths_cfg.get("s2p_output", "suite2p_workspace/output")

    if args.training:
        log.info("Training data validation")
        log.info("  Annotated: %s", annotated_dir)
        log.info("  Masks:     %s", masks_dir)

        if not annotated_dir.exists():
            log.error("Annotated directory not found: %s", annotated_dir)
            return
        if not masks_dir.exists():
            log.error("Masks directory not found: %s", masks_dir)
            return

        if args.dry_run:
            n_masks = len(list(masks_dir.glob("*_masks.tif")))
            n_means = len(list(annotated_dir.glob("*_mean.tif")))
            log.info("DRY RUN — would validate %d masks against %d mean images",
                     n_masks, n_means)
            return

        results = validate_training_data(annotated_dir, masks_dir)
        print_training_report(results)

    else:
        log.info("Suite2p output validation")
        log.info("  Suite2p:      %s", s2p_dir)
        log.info("  Annotated:    %s", annotated_dir)
        log.info("  Max proj:     %s", max_proj_dir)

        if not s2p_dir.exists():
            log.error("Suite2p output directory not found: %s", s2p_dir)
            return

        if args.dry_run:
            fov_dirs = [d for d in s2p_dir.iterdir() if d.is_dir()]
            log.info("DRY RUN — would validate %d FOV directories", len(fov_dirs))
            return

        results = validate_suite2p_outputs(s2p_dir, annotated_dir, max_proj_dir)
        print_suite2p_report(results)


if __name__ == "__main__":
    main()
