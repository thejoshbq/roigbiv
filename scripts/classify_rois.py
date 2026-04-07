#!/usr/bin/env python3
"""
ROI G. Biv --- Step 10: Quality control and activity-type classification.

Stage A: Automated cell/not-cell rejection (SNR, area, compactness).
Stage B: Activity-type labeling (phasic, tonic, sparse, ambiguous).

Outputs per FOV:
  {stem}_classification.csv  — per-ROI features + is_cell + activity_type

Batch output:
  classification_summary.csv — all FOVs combined

Usage:
  python classify_rois.py --fs 30.0
  python classify_rois.py --fs 30.0 --snr_min 3.0
  python classify_rois.py --fs 30.0 --dry-run
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

import sys
from config import BASE_DIR, load_config
sys.path.insert(0, str(BASE_DIR))
from roigbiv.classify import classify_fov

MERGED_OUT = BASE_DIR / "inference" / "merged"
TRACES_OUT = BASE_DIR / "inference" / "traces"
CLASSIFIED_OUT = BASE_DIR / "inference" / "classified"
LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("classify_rois")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to console and logs/classify.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "classify.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)
    # Library logger
    lib_log = logging.getLogger("classify")
    lib_log.setLevel(logging.INFO)
    lib_log.addHandler(fh)
    lib_log.addHandler(ch)


# ---------------------------------------------------------------------------
# FOV discovery
# ---------------------------------------------------------------------------

def discover_fovs(traces_dir: Path, merged_dir: Path) -> list[dict]:
    """Find FOVs with both traces and merged masks.

    Returns list of dicts with keys: stem, has_traces, has_mask.
    """
    fovs = []
    if not traces_dir.exists():
        return fovs

    for f_path in sorted(traces_dir.glob("*_F.npy")):
        stem = f_path.stem.replace("_F", "")
        has_mask = (merged_dir / f"{stem}_merged_masks.tif").exists()
        fovs.append({
            "stem": stem,
            "has_traces": True,
            "has_mask": has_mask,
        })
    return fovs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Step 10: QC rejection + activity-type classification.")

    ap.add_argument("--fs", type=float, required=True,
                    help="Acquisition frame rate in Hz (REQUIRED)")
    ap.add_argument("--traces_dir", default=None,
                    help=f"Traces directory (default: {TRACES_OUT})")
    ap.add_argument("--merged_dir", default=None,
                    help=f"Merged mask directory (default: {MERGED_OUT})")
    ap.add_argument("--output_dir", default=None,
                    help=f"Output directory (default: {CLASSIFIED_OUT})")

    # Stage A thresholds
    ap.add_argument("--snr_min", type=float, default=None,
                    help="Min SNR for cell classification (default: 2.0)")
    ap.add_argument("--area_min", type=int, default=None,
                    help="Min ROI area in pixels (default: 30)")
    ap.add_argument("--area_max", type=int, default=None,
                    help="Max ROI area in pixels (default: 500)")
    ap.add_argument("--compact_min", type=float, default=None,
                    help="Min compactness (default: 0.15)")

    # Stage B thresholds
    ap.add_argument("--skew_phasic", type=float, default=None,
                    help="Min skewness for phasic classification (default: 1.5)")
    ap.add_argument("--cv_tonic", type=float, default=None,
                    help="Max CV for tonic classification (default: 0.3)")
    ap.add_argument("--min_transients_sparse", type=int, default=None,
                    help="Max transients for sparse classification (default: 5)")

    ap.add_argument("--config", default=None,
                    help="Path to pipeline YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Discover FOVs and print plan without processing")

    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    cc = cfg.get("classify", {})
    paths_cfg = cfg.get("paths", {})

    # Resolve directories
    traces_dir = Path(args.traces_dir) if args.traces_dir else \
        BASE_DIR / paths_cfg.get("traces_output", "inference/traces")
    merged_dir = Path(args.merged_dir) if args.merged_dir else \
        BASE_DIR / paths_cfg.get("merged_output", "inference/merged")
    out_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / paths_cfg.get("classified_output", "inference/classified")

    # Resolve parameters: CLI > config > defaults
    snr_min = args.snr_min if args.snr_min is not None else cc.get("snr_min", 2.0)
    area_min = args.area_min if args.area_min is not None else cc.get("area_min", 30)
    area_max = args.area_max if args.area_max is not None else cc.get("area_max", 500)
    compact_min = args.compact_min if args.compact_min is not None else cc.get("compact_min", 0.15)
    skew_phasic = args.skew_phasic if args.skew_phasic is not None else cc.get("skew_phasic", 1.5)
    cv_tonic = args.cv_tonic if args.cv_tonic is not None else cc.get("cv_tonic", 0.3)
    min_trans = args.min_transients_sparse if args.min_transients_sparse is not None \
        else cc.get("min_transients_sparse", 5)

    # Print plan
    log.info("Step 10 — QC + activity-type classification")
    log.info("  Traces:        %s", traces_dir)
    log.info("  Merged masks:  %s", merged_dir)
    log.info("  Output:        %s", out_dir)
    log.info("  Stage A: SNR>=%.1f, area=[%d,%d], compact>=%.2f",
             snr_min, area_min, area_max, compact_min)
    log.info("  Stage B: skew_phasic>=%.1f, cv_tonic<%.2f, sparse<%d transients",
             skew_phasic, cv_tonic, min_trans)

    # Discover FOVs
    fovs = discover_fovs(traces_dir, merged_dir)
    ready = [f for f in fovs if f["has_mask"]]
    log.info("  FOVs: %d with traces, %d with masks", len(fovs), len(ready))

    if args.dry_run:
        for f in ready:
            log.info("    %s", f["stem"])
        log.info("DRY RUN — exiting without processing")
        return

    if not ready:
        log.error("No FOVs ready for classification")
        return

    # Process
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    n_done = n_err = 0

    for i, fov in enumerate(ready, 1):
        stem = fov["stem"]
        log.info("[%d/%d] %s", i, len(ready), stem)

        try:
            rec_path = merged_dir / f"{stem}_merge_records.csv"
            df = classify_fov(
                stem=stem,
                traces_dir=traces_dir,
                merged_mask_dir=merged_dir,
                merge_records_path=rec_path,
                out_dir=out_dir,
                fs=args.fs,
                snr_min=snr_min,
                area_min=area_min,
                area_max=area_max,
                compact_min=compact_min,
                skew_phasic=skew_phasic,
                cv_tonic=cv_tonic,
                min_transients_sparse=min_trans,
            )
            if not df.empty:
                all_dfs.append(df)
                n_done += 1
        except Exception as exc:
            log.error("  ERROR: %s — %s", stem, exc)
            n_err += 1

    # Write combined summary
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        summary_path = out_dir / "classification_summary.csv"
        combined.to_csv(str(summary_path), index=False)

        n_cells = int(combined["is_cell"].sum())
        n_total = len(combined)
        type_counts = combined.loc[combined["is_cell"], "activity_type"].value_counts()
        log.info("Summary: %d/%d cells across %d FOVs", n_cells, n_total, n_done)
        for atype, count in type_counts.items():
            log.info("  %s: %d", atype, count)
        log.info("-> %s", summary_path)

    parts = [f"{n_done} processed"]
    if n_err:
        parts.append(f"{n_err} errors")
    log.info("Complete: %s", ", ".join(parts))


if __name__ == "__main__":
    main()
