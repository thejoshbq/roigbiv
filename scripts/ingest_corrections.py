#!/usr/bin/env python3
"""
ROI G. Biv --- Ingest Cellpose GUI corrections into training dataset.

Converts *_seg.npy files (saved by Cellpose GUI after manual review) into
*_masks.tif files for retraining. Run after each HITL review session.

Workflow:
  1. Open mean projections + masks in Cellpose GUI
  2. Delete false positives, draw missed ROIs, adjust boundaries
  3. Save (creates *_seg.npy alongside the mean image)
  4. Run this script to update masks directory
  5. Retrain from cyto3 base: python train.py --run_id runXXX

Usage:
  python ingest_corrections.py --annotated_dir data/annotated
  python ingest_corrections.py --annotated_dir data/annotated --masks_dir data/masks
  python ingest_corrections.py --annotated_dir data/annotated --dry-run
"""

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import tifffile

from config import BASE_DIR, load_config

ANNOTATED_DIR = BASE_DIR / "data" / "annotated"
MASKS_DIR = BASE_DIR / "data" / "masks"
LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("ingest_corrections")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to console and logs/ingest.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "ingest.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest_corrections(
    annotated_dir: Path,
    masks_dir: Path,
    backup: bool = True,
    dry_run: bool = False,
) -> list[dict]:
    """Convert *_seg.npy files to *_masks.tif in masks_dir.

    Parameters
    ----------
    annotated_dir : directory containing *_seg.npy from Cellpose GUI
    masks_dir     : output directory for *_masks.tif
    backup        : backup existing mask before overwriting
    dry_run       : report what would happen without writing

    Returns
    -------
    list of dicts with keys: stem, n_rois, action (created/updated/skipped)
    """
    seg_files = sorted(annotated_dir.glob("*_seg.npy"))
    if not seg_files:
        log.warning("No *_seg.npy files found in %s", annotated_dir)
        return []

    if not dry_run:
        masks_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seg_file in seg_files:
        # Extract stem: foo_mean_seg.npy -> foo, foo_seg.npy -> foo
        stem = seg_file.stem
        for suffix in ("_mean_seg", "_max_seg", "_vcorr_seg", "_seg"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break

        data = np.load(str(seg_file), allow_pickle=True).item()
        masks = data.get("masks", None)

        if masks is None or masks.max() == 0:
            log.warning("  %s: empty or no masks, skipping", seg_file.name)
            results.append({"stem": stem, "n_rois": 0, "action": "skipped"})
            continue

        n_rois = int(masks.max())
        out_path = masks_dir / f"{stem}_masks.tif"
        existing = out_path.exists()
        action = "updated" if existing else "created"

        if dry_run:
            log.info("  %s: %d ROIs -> %s (%s)", stem, n_rois, out_path.name, action)
            results.append({"stem": stem, "n_rois": n_rois, "action": action})
            continue

        # Backup existing mask
        if existing and backup:
            backup_path = masks_dir / f"{stem}_masks.tif.bak"
            shutil.copy2(str(out_path), str(backup_path))
            log.info("  %s: backed up existing mask", stem)

        tifffile.imwrite(str(out_path), masks.astype(np.uint16))
        log.info("  %s: %d ROIs -> %s (%s)", stem, n_rois, out_path.name, action)
        results.append({"stem": stem, "n_rois": n_rois, "action": action})

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Ingest Cellpose GUI corrections (*_seg.npy) into training masks.")

    ap.add_argument("--annotated_dir", default=None,
                    help=f"Directory with *_seg.npy files (default: {ANNOTATED_DIR})")
    ap.add_argument("--masks_dir", default=None,
                    help=f"Output directory for *_masks.tif (default: {MASKS_DIR})")
    ap.add_argument("--no_backup", action="store_true",
                    help="Skip backup of existing masks before overwriting")
    ap.add_argument("--config", default=None,
                    help="Path to pipeline YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be ingested without writing files")

    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})

    annotated_dir = Path(args.annotated_dir) if args.annotated_dir else \
        BASE_DIR / paths_cfg.get("annotated_dir", "data/annotated")
    masks_dir = Path(args.masks_dir) if args.masks_dir else \
        BASE_DIR / paths_cfg.get("masks_dir", "data/masks")

    log.info("Ingest Cellpose GUI corrections")
    log.info("  Annotated: %s", annotated_dir)
    log.info("  Masks:     %s", masks_dir)

    if not annotated_dir.exists():
        log.error("Annotated directory not found: %s", annotated_dir)
        return

    results = ingest_corrections(
        annotated_dir, masks_dir,
        backup=not args.no_backup,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        log.info("DRY RUN — no files written")

    # Summary
    n_created = sum(1 for r in results if r["action"] == "created")
    n_updated = sum(1 for r in results if r["action"] == "updated")
    n_skipped = sum(1 for r in results if r["action"] == "skipped")
    total_rois = sum(r["n_rois"] for r in results)

    parts = []
    if n_created:
        parts.append(f"{n_created} created")
    if n_updated:
        parts.append(f"{n_updated} updated")
    if n_skipped:
        parts.append(f"{n_skipped} skipped")
    log.info("Complete: %s, %d total ROIs", ", ".join(parts) or "nothing to ingest", total_rois)


if __name__ == "__main__":
    main()
