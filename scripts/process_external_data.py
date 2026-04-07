#!/usr/bin/env python3
"""
ROI G. Biv — Process external ROIGBIV-DATA for Cellpose training.

Discovers *_mc.tif + *_RoiSet.zip pairs on the external drive, runs Suite2p
(no registration) to extract meanImg and Vcorr, converts ImageJ ROIs to uint16
label masks, and writes Cellpose-ready output to cellpose_ready/.

Usage:
  python process_external_data.py                          # dry-run (discovery only)
  python process_external_data.py --execute                # process all sessions
  python process_external_data.py --execute --skip_existing # resume after interruption
"""

import argparse
import csv
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import roifile
import tifffile
from skimage.draw import polygon

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from roigbiv.suite2p import run_suite2p_fov

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/mnt/external/ROIGBIV-DATA")
SEARCH_ROOTS = [
    DATA_ROOT / "PrL-NAc-G6-5M",
    DATA_ROOT / "PrL-NAc-G6-6M",
    DATA_ROOT / "PrL-NAc-G6-6F",
    DATA_ROOT / "PrL-NAc-G6-7F",
    DATA_ROOT / "040226" / "PrL-NAc-G6-10M",
    DATA_ROOT / "040226" / "PrL-NAc-G6-11M",
]
EXCLUDE_PATTERNS = ["040226_transfer"]

log = logging.getLogger("process")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_pairs(
    search_roots: list[Path],
    exclude_patterns: list[str] | None = None,
) -> list[tuple[Path, Path, str]]:
    """Find *_mc.tif + *_RoiSet.zip pairs, deduplicated by mc_stem.

    Returns list of (mc_tif_path, roi_zip_path, mc_stem) tuples.
    """
    exclude = exclude_patterns or []
    seen_stems: set[str] = set()
    pairs: list[tuple[Path, Path, str]] = []
    unpaired_mc: list[Path] = []

    for root in search_roots:
        if not root.exists():
            log.warning("Search root does not exist: %s", root)
            continue

        for mc_tif in sorted(root.rglob("*_mc.tif")):
            # Skip excluded directories
            if any(ex in str(mc_tif) for ex in exclude):
                continue

            mc_stem = mc_tif.stem  # e.g. T1_..._mc
            if mc_stem in seen_stems:
                continue
            seen_stems.add(mc_stem)

            # Find matching RoiSet.zip: strip _mc to get base, then add _RoiSet.zip
            base = mc_stem.removesuffix("_mc")
            roi_zip = mc_tif.parent / f"{base}_RoiSet.zip"

            if not roi_zip.exists():
                # Also check for RoiSet.zip without prefix (loose files)
                unpaired_mc.append(mc_tif)
                continue

            pairs.append((mc_tif, roi_zip, mc_stem))

    if unpaired_mc:
        log.warning("%d _mc.tif files without matching RoiSet.zip:", len(unpaired_mc))
        for p in unpaired_mc:
            log.warning("  %s", p.name)

    return pairs


# ---------------------------------------------------------------------------
# Mask conversion
# ---------------------------------------------------------------------------

def roi_zip_to_mask(roi_zip_path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Convert ImageJ ROI .zip to uint16 label mask."""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint16)
    rois = roifile.roiread(str(roi_zip_path))

    for idx, roi in enumerate(rois, start=1):
        coords = roi.coordinates()
        if coords is None or len(coords) < 3:
            continue
        rr, cc = polygon(coords[:, 1], coords[:, 0], shape=(H, W))
        mask[rr, cc] = idx

    return mask


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------

def process_session(
    mc_tif: Path,
    roi_zip: Path,
    mc_stem: str,
    output_annotated: Path,
    output_masks: Path,
    suite2p_workspace: Path,
    cfg: dict,
) -> dict:
    """Process a single session: Suite2p → extract projections → convert mask.

    Returns a dict with processing metadata for the manifest.
    """
    result = {
        "stem": mc_stem,
        "mc_tif": str(mc_tif),
        "roi_zip": str(roi_zip),
        "status": "error",
        "n_rois": 0,
        "n_frames": 0,
        "elapsed_sec": 0,
    }
    t0 = time.time()

    # The Suite2p output directory uses stem without _mc
    s2p_stem = mc_stem.removesuffix("_mc")
    s2p_plane_dir = suite2p_workspace / s2p_stem / "suite2p" / "plane0"

    # --- Run Suite2p ---
    log.info("Running Suite2p on %s ...", mc_stem)
    run_suite2p_fov(
        mc_tif, suite2p_workspace,
        fs=30.0, tau=1.0,
        do_registration=False,
        cfg=cfg,
    )

    # --- Extract projections from ops.npy ---
    ops_path = s2p_plane_dir / "ops.npy"
    if not ops_path.exists():
        log.error("ops.npy not found at %s", ops_path)
        result["elapsed_sec"] = time.time() - t0
        return result

    ops = np.load(str(ops_path), allow_pickle=True).item()

    if "meanImg" not in ops:
        log.error("No meanImg in ops.npy for %s", mc_stem)
        result["elapsed_sec"] = time.time() - t0
        return result

    mean_img = ops["meanImg"].astype(np.float32)
    H, W = mean_img.shape
    result["n_frames"] = int(ops.get("nframes", 0))

    mean_path = output_annotated / f"{mc_stem}_mean.tif"
    tifffile.imwrite(str(mean_path), mean_img)
    log.info("  Saved mean projection: %s", mean_path.name)

    if "Vcorr" in ops:
        vcorr = ops["Vcorr"].astype(np.float32)
        vcorr_path = output_annotated / f"{mc_stem}_vcorr.tif"
        tifffile.imwrite(str(vcorr_path), vcorr)
        log.info("  Saved Vcorr: %s", vcorr_path.name)
    else:
        log.warning("  No Vcorr in ops.npy for %s", mc_stem)

    # --- Convert ROI zip to mask ---
    mask = roi_zip_to_mask(roi_zip, (H, W))
    n_rois = int(mask.max())
    result["n_rois"] = n_rois

    if n_rois == 0:
        log.warning("  Empty mask (0 ROIs) for %s", mc_stem)

    mask_path = output_masks / f"{mc_stem}_masks.tif"
    tifffile.imwrite(str(mask_path), mask)
    log.info("  Saved mask (%d ROIs): %s", n_rois, mask_path.name)

    # --- Cleanup Suite2p workspace ---
    s2p_fov_dir = suite2p_workspace / s2p_stem
    if s2p_fov_dir.exists():
        shutil.rmtree(s2p_fov_dir, ignore_errors=True)
        log.info("  Cleaned up Suite2p workspace for %s", s2p_stem)

    result["status"] = "ok"
    result["elapsed_sec"] = round(time.time() - t0, 1)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Process external ROIGBIV-DATA for Cellpose training.")
    ap.add_argument("--data_root", default=str(DATA_ROOT),
                    help="Root of external data (default: /mnt/external/ROIGBIV-DATA)")
    ap.add_argument("--output_dir", default=None,
                    help="Output base dir (default: {data_root}/cellpose_ready)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually process (default is dry-run)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip sessions whose outputs already exist")
    ap.add_argument("--nbinned", type=int, default=2000,
                    help="Suite2p nbinned for SVD (default: 2000, memory-safe)")
    ap.add_argument("--batch_size", type=int, default=250,
                    help="Suite2p batch_size for I/O (default: 250)")
    args = ap.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "cellpose_ready"
    output_annotated = output_dir / "annotated"
    output_masks = output_dir / "masks"
    suite2p_workspace = data_root / "suite2p_workspace"

    # Update search roots if custom data_root
    search_roots = SEARCH_ROOTS
    if args.data_root != str(DATA_ROOT):
        dr = Path(args.data_root)
        search_roots = [
            dr / "PrL-NAc-G6-5M",
            dr / "PrL-NAc-G6-6M",
            dr / "PrL-NAc-G6-6F",
            dr / "PrL-NAc-G6-7F",
            dr / "040226" / "PrL-NAc-G6-10M",
            dr / "040226" / "PrL-NAc-G6-11M",
        ]

    # --- Discovery ---
    log.info("Discovering _mc.tif + RoiSet.zip pairs ...")
    pairs = discover_pairs(search_roots, EXCLUDE_PATTERNS)
    log.info("Found %d complete pairs", len(pairs))

    if not pairs:
        log.error("No pairs found. Check data_root: %s", data_root)
        sys.exit(1)

    # Print summary by cohort
    cohort_counts: dict[str, int] = {}
    for _, _, stem in pairs:
        for cohort in ("G6-5M", "G6-6M", "G6-6F", "G6-7F", "G6-10M", "G6-11M"):
            if cohort in stem:
                cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1
                break
    for cohort, count in sorted(cohort_counts.items()):
        log.info("  %s: %d pairs", cohort, count)

    if not args.execute:
        log.info("DRY RUN — exiting without processing. Use --execute to process.")
        return

    # --- Setup output directories ---
    output_annotated.mkdir(parents=True, exist_ok=True)
    output_masks.mkdir(parents=True, exist_ok=True)
    suite2p_workspace.mkdir(parents=True, exist_ok=True)

    # Add file logging
    log_file = output_dir / "processing.log"
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(fh)

    # Suite2p config for memory safety
    cfg = {"suite2p": {"nbinned": args.nbinned, "batch_size": args.batch_size}}

    # --- Process sessions ---
    manifest_rows: list[dict] = []
    n_processed = n_skipped = n_errors = 0

    for i, (mc_tif, roi_zip, mc_stem) in enumerate(pairs, 1):
        log.info("[%d/%d] %s", i, len(pairs), mc_stem)

        # Check if outputs already exist
        if args.skip_existing:
            mean_exists = (output_annotated / f"{mc_stem}_mean.tif").exists()
            mask_exists = (output_masks / f"{mc_stem}_masks.tif").exists()
            if mean_exists and mask_exists:
                log.info("  Skipping (outputs exist)")
                n_skipped += 1
                continue

        try:
            result = process_session(
                mc_tif, roi_zip, mc_stem,
                output_annotated, output_masks, suite2p_workspace,
                cfg,
            )
            manifest_rows.append(result)

            if result["status"] == "ok":
                n_processed += 1
                log.info("  Done in %.1fs (%d ROIs, %d frames)",
                         result["elapsed_sec"], result["n_rois"], result["n_frames"])
            else:
                n_errors += 1
                log.error("  Failed: %s", mc_stem)

        except Exception as exc:
            n_errors += 1
            log.error("  Exception processing %s: %s", mc_stem, exc)
            manifest_rows.append({
                "stem": mc_stem,
                "mc_tif": str(mc_tif),
                "roi_zip": str(roi_zip),
                "status": f"exception: {exc}",
                "n_rois": 0,
                "n_frames": 0,
                "elapsed_sec": 0,
            })
            # Cleanup any partial Suite2p output
            s2p_stem = mc_stem.removesuffix("_mc")
            s2p_dir = suite2p_workspace / s2p_stem
            if s2p_dir.exists():
                shutil.rmtree(s2p_dir, ignore_errors=True)

    # --- Write manifest ---
    manifest_path = output_dir / "manifest.csv"
    if manifest_rows:
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(manifest_rows)
        log.info("Manifest written: %s", manifest_path)

    # --- Summary ---
    log.info("Processing complete: %d processed, %d skipped, %d errors",
             n_processed, n_skipped, n_errors)

    # Verify output counts
    n_mean = len(list(output_annotated.glob("*_mean.tif")))
    n_vcorr = len(list(output_annotated.glob("*_vcorr.tif")))
    n_masks = len(list(output_masks.glob("*_masks.tif")))
    log.info("Output: %d mean, %d vcorr, %d masks", n_mean, n_vcorr, n_masks)

    if n_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
