#!/usr/bin/env python3
"""
ROI G. Biv --- Step 9: Trace extraction from registered movie.

Extracts fluorescence traces for every merged ROI from Suite2p's data.bin
(registered movie). Computes neuropil-corrected traces, dF/F with rolling
baseline, and optional OASIS spike deconvolution.

Outputs per FOV:
  {stem}_F.npy      — (n_rois, T) raw fluorescence
  {stem}_Fneu.npy   — (n_rois, T) neuropil fluorescence
  {stem}_dFF.npy    — (n_rois, T) dF/F
  {stem}_spks.npy   — (n_rois, T) deconvolved spikes
  {stem}_alpha.npy  — (n_rois,)   neuropil contamination coefficients

Usage:
  python extract_traces.py --fs 30.0
  python extract_traces.py --fs 30.0 --no_deconvolve
  python extract_traces.py --fs 30.0 --dry-run
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
from config import BASE_DIR, load_config
sys.path.insert(0, str(BASE_DIR))
from roigbiv.traces import extract_traces_fov

S2P_OUT = BASE_DIR / "suite2p_workspace" / "output"
MERGED_OUT = BASE_DIR / "inference" / "merged"
TRACES_OUT = BASE_DIR / "inference" / "traces"
LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("extract_traces")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to console and logs/traces.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "traces.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)
    # Also configure library logger
    lib_log = logging.getLogger("traces")
    lib_log.setLevel(logging.INFO)
    lib_log.addHandler(fh)
    lib_log.addHandler(ch)


# ---------------------------------------------------------------------------
# FOV discovery
# ---------------------------------------------------------------------------

def discover_fovs(merged_dir: Path, s2p_dir: Path) -> list[dict]:
    """Find FOVs with merged masks and data.bin.

    Returns list of dicts with keys: stem, has_mask, has_databin, bin_mb.
    """
    fovs = []
    if not merged_dir.exists():
        return fovs

    for mask_path in sorted(merged_dir.glob("*_merged_masks.tif")):
        stem = mask_path.stem.replace("_merged_masks", "")
        bin_path = s2p_dir / stem / "suite2p" / "plane0" / "data.bin"
        has_bin = bin_path.exists()
        bin_mb = bin_path.stat().st_size / 1e6 if has_bin else 0

        fovs.append({
            "stem": stem,
            "has_mask": True,
            "has_databin": has_bin,
            "bin_mb": bin_mb,
        })
    return fovs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Step 9: Trace extraction from registered movie (data.bin).")

    ap.add_argument("--fs", type=float, required=True,
                    help="Acquisition frame rate in Hz (REQUIRED)")
    ap.add_argument("--s2p_dir", default=None,
                    help=f"Suite2p output directory (default: {S2P_OUT})")
    ap.add_argument("--merged_dir", default=None,
                    help=f"Merged mask directory (default: {MERGED_OUT})")
    ap.add_argument("--output_dir", default=None,
                    help=f"Output directory for traces (default: {TRACES_OUT})")

    # Trace parameters
    ap.add_argument("--tau", type=float, default=None,
                    help="GCaMP decay constant in seconds (default: 1.0)")
    ap.add_argument("--inner_radius", type=int, default=None,
                    help="Neuropil inner radius in pixels (default: 2)")
    ap.add_argument("--min_neuropil_pixels", type=int, default=None,
                    help="Minimum neuropil pixel count (default: 350)")
    ap.add_argument("--baseline_window", type=float, default=None,
                    help="Rolling baseline window in seconds (default: 60.0)")
    ap.add_argument("--baseline_percentile", type=float, default=None,
                    help="Baseline percentile (default: 10.0)")
    ap.add_argument("--chunk_frames", type=int, default=None,
                    help="Frames per read chunk (default: 200)")
    ap.add_argument("--no_deconvolve", action="store_true",
                    help="Skip spike deconvolution")

    ap.add_argument("--config", default=None,
                    help="Path to pipeline YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Discover FOVs and print plan without processing")

    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    tc = cfg.get("traces", {})
    s2p_cfg = cfg.get("suite2p", {})
    paths_cfg = cfg.get("paths", {})

    # Resolve directories
    s2p_dir = Path(args.s2p_dir) if args.s2p_dir else \
        BASE_DIR / paths_cfg.get("s2p_output", "suite2p_workspace/output")
    merged_dir = Path(args.merged_dir) if args.merged_dir else \
        BASE_DIR / paths_cfg.get("merged_output", "inference/merged")
    out_dir = Path(args.output_dir) if args.output_dir else \
        BASE_DIR / paths_cfg.get("traces_output", "inference/traces")

    # Resolve parameters: CLI > config > defaults
    tau = args.tau if args.tau is not None else s2p_cfg.get("tau", 1.0)
    inner_radius = args.inner_radius if args.inner_radius is not None \
        else tc.get("inner_radius", 2)
    min_neuropil = args.min_neuropil_pixels if args.min_neuropil_pixels is not None \
        else tc.get("min_neuropil_pixels", 350)
    baseline_window = args.baseline_window if args.baseline_window is not None \
        else tc.get("baseline_window_seconds", 60.0)
    baseline_pct = args.baseline_percentile if args.baseline_percentile is not None \
        else tc.get("baseline_percentile", 10.0)
    chunk_frames = args.chunk_frames if args.chunk_frames is not None \
        else tc.get("chunk_frames", 200)
    tonic_mult = tc.get("tonic_baseline_multiplier", 2.0)
    do_deconvolve = not args.no_deconvolve and tc.get("do_deconvolve", True)

    # Print plan
    log.info("Step 9 — Trace extraction")
    log.info("  Suite2p dir:        %s", s2p_dir)
    log.info("  Merged masks:       %s", merged_dir)
    log.info("  Output:             %s", out_dir)
    log.info("  Frame rate:         %.1f Hz", args.fs)
    log.info("  Tau:                %.2f s", tau)
    log.info("  Neuropil radius:    %d px (min %d px)", inner_radius, min_neuropil)
    log.info("  Baseline:           %.0fs window, %.0fth percentile", baseline_window, baseline_pct)
    log.info("  Chunk frames:       %d", chunk_frames)
    log.info("  Deconvolution:      %s", "ON" if do_deconvolve else "OFF")

    # Discover FOVs
    fovs = discover_fovs(merged_dir, s2p_dir)
    ready = [f for f in fovs if f["has_databin"]]
    not_ready = [f for f in fovs if not f["has_databin"]]

    log.info("  FOVs: %d with merged masks, %d with data.bin", len(fovs), len(ready))
    if not_ready:
        for f in not_ready:
            log.warning("    %s: missing data.bin", f["stem"])

    if args.dry_run:
        for f in ready:
            log.info("    %s: data.bin=%.0f MB", f["stem"], f["bin_mb"])
        log.info("DRY RUN — exiting without processing")
        return

    if not ready:
        log.error("No FOVs ready for trace extraction")
        return

    # Load merge records for tonic identification
    records_by_stem = {}
    for f in ready:
        rec_path = merged_dir / f"{f['stem']}_merge_records.csv"
        if rec_path.exists():
            df = pd.read_csv(str(rec_path))
            records_by_stem[f["stem"]] = df.to_dict("records")

    # Process
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = n_err = 0
    total = len(ready)

    for i, fov in enumerate(ready, 1):
        stem = fov["stem"]
        log.info("[%d/%d] %s ...", i, total, stem)
        t0 = time.time()

        try:
            result = extract_traces_fov(
                stem=stem,
                merged_mask_dir=merged_dir,
                s2p_dir=s2p_dir,
                out_dir=out_dir,
                merge_records=records_by_stem.get(stem),
                fs=args.fs,
                tau=tau,
                inner_radius=inner_radius,
                min_neuropil_pixels=min_neuropil,
                baseline_window=baseline_window,
                baseline_percentile=baseline_pct,
                tonic_multiplier=tonic_mult,
                do_deconvolve=do_deconvolve,
                chunk_frames=chunk_frames,
            )
            elapsed = time.time() - t0
            status = result.get("status", "unknown")
            if status == "done":
                log.info("    done (%.0fs)", elapsed)
                n_done += 1
            elif status == "skipped":
                n_skip += 1
            else:
                log.warning("    %s: %s", stem, status)
                n_skip += 1
        except Exception as exc:
            log.error("    ERROR: %s — %s", stem, exc)
            n_err += 1

    # Summary
    parts = [f"{n_done} processed"]
    if n_skip:
        parts.append(f"{n_skip} skipped")
    if n_err:
        parts.append(f"{n_err} errors")
    log.info("Complete: %s -> %s", ", ".join(parts), out_dir)


if __name__ == "__main__":
    main()
