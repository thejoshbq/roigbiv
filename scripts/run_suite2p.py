#!/usr/bin/env python3
"""
ROI G. Biv — Suite2p processing + projection extraction.

Runs Suite2p on raw TIF stacks, then extracts mean, Vcorr, and max
projections from ops.npy. This single step produces all intermediate
products consumed by every branch of the three-branch pipeline.

Registration auto-detect:
  - Filenames containing '_mc' → registration OFF (pre-corrected)
  - All other files → registration ON
  - Override with --no_registration or --force_registration

Usage:
  python run_suite2p.py --fs 30.0 --input_dir data/raw --batch
  python run_suite2p.py --fs 30.0 --single_file data/raw/FOV1_mc.tif
  python run_suite2p.py --fs 30.0 --input_dir data/raw --batch --dry-run
"""

import argparse
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile

from config import BASE_DIR, load_config

S2P_OUT = BASE_DIR / "suite2p_workspace" / "output"
ANNOTATED_DIR = BASE_DIR / "data" / "annotated"
MAX_PROJ_DIR = BASE_DIR / "data" / "max_projections"
LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("run_suite2p")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to console and logs/suite2p.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "suite2p.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)


# ---------------------------------------------------------------------------
# Ops construction
# ---------------------------------------------------------------------------

def build_ops(
    input_dir: Path,
    fs: float,
    tau: float = 1.0,
    cfg: dict | None = None,
    anatomical_only: int = 0,
    do_registration: bool = True,
    threshold_scaling: float | None = None,
) -> dict:
    """Construct Suite2p ops dict from CLI args + YAML config.

    CLI-supplied values always take precedence over config values.
    """
    from suite2p import default_ops

    ops = default_ops()
    s2p_cfg = (cfg or {}).get("suite2p", {})

    ops.update({
        # ── Data ──
        "data_path":       [str(input_dir)],
        "save_folder":     "suite2p",
        "nplanes":         s2p_cfg.get("nplanes", 1),
        "nchannels":       s2p_cfg.get("nchannels", 1),
        "functional_chan":  s2p_cfg.get("functional_chan", 1),
        "tau":             tau,
        "fs":              fs,

        # ── Registration ──
        "do_registration": 1 if do_registration else 0,
        "nimg_init":       s2p_cfg.get("nimg_init", 300),
        "batch_size":      s2p_cfg.get("batch_size", 500),
        "smooth_sigma":    s2p_cfg.get("smooth_sigma", 1.15),
        "maxregshift":     s2p_cfg.get("maxregshift", 0.1),
        "nonrigid":        s2p_cfg.get("nonrigid", True),
        "block_size":      s2p_cfg.get("block_size", [128, 128]),

        # ── Detection ──
        "spatial_scale":     s2p_cfg.get("spatial_scale", 0),
        "threshold_scaling": threshold_scaling if threshold_scaling is not None
                             else s2p_cfg.get("threshold_scaling", 0.5),
        "max_iterations":    s2p_cfg.get("max_iterations", 20),
        "connected":         s2p_cfg.get("connected", True),
        "nbinned":           s2p_cfg.get("nbinned", 5000),
        "allow_overlap":     s2p_cfg.get("allow_overlap", False),

        # ── Classification ──
        "preclassify": s2p_cfg.get("preclassify", 0.0),

        # ── Neuropil ──
        "high_pass":             s2p_cfg.get("high_pass", 100),
        "inner_neuropil_radius": s2p_cfg.get("inner_neuropil_radius", 2),
        "min_neuropil_pixels":   s2p_cfg.get("min_neuropil_pixels", 350),

        # ── Spike deconvolution ──
        "spikedetect": s2p_cfg.get("spikedetect", True),

        # ── Anatomical mode ──
        "anatomical_only": anatomical_only,

        # ── Binary management ──
        "delete_bin": False,
    })
    return ops


# ---------------------------------------------------------------------------
# Projection extraction
# ---------------------------------------------------------------------------

def extract_projections_from_ops(
    ops: dict,
    stem: str,
    annotated_dir: Path,
    max_proj_dir: Path,
) -> dict[str, Path]:
    """Extract mean, Vcorr, max projections from Suite2p ops to TIF files.

    Returns dict mapping projection name to saved path.
    Saves as float32 TIF. Skips any projection not found in ops.
    """
    annotated_dir.mkdir(parents=True, exist_ok=True)
    max_proj_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    projection_map = {
        "meanImg":  (annotated_dir,  f"{stem}_mean.tif"),
        "Vcorr":    (annotated_dir,  f"{stem}_vcorr.tif"),
        "max_proj": (max_proj_dir,   f"{stem}_max.tif"),
    }

    for key, (out_dir, filename) in projection_map.items():
        if key in ops and ops[key] is not None:
            img = np.asarray(ops[key], dtype=np.float32)
            out_path = out_dir / filename
            tifffile.imwrite(str(out_path), img)
            saved[key] = out_path
            log.info("  Saved %s → %s", key, out_path)
        else:
            log.warning("  %s not found in ops for %s", key, stem)

    return saved


# ---------------------------------------------------------------------------
# Registration auto-detect
# ---------------------------------------------------------------------------

def should_register(tif_path: Path, force_on: bool, force_off: bool) -> bool:
    """Determine whether to run registration based on filename and flags.

    Priority: --force_registration > --no_registration > auto-detect (_mc).
    """
    if force_on:
        return True
    if force_off:
        return False
    # Auto-detect: _mc in stem means pre-motion-corrected
    return "_mc" not in tif_path.stem


# ---------------------------------------------------------------------------
# Per-FOV processing
# ---------------------------------------------------------------------------

def run_one_fov(
    tif_path: Path,
    output_dir: Path,
    fs: float,
    tau: float = 1.0,
    do_registration: bool = True,
    anatomical_only: int = 0,
    threshold_scaling: float | None = None,
    cfg: dict | None = None,
    annotated_dir: Path = ANNOTATED_DIR,
    max_proj_dir: Path = MAX_PROJ_DIR,
) -> dict:
    """Run Suite2p on a single FOV, extract projections, return metadata.

    Output convention: Suite2p outputs land at
        output_dir/{stem}/suite2p/plane0/

    Resumability: skips if stat.npy already exists.

    Returns dict with keys: stem, processed, n_rois, projections, elapsed.
    """
    from suite2p.run_s2p import run_s2p

    stem = tif_path.stem.replace("_mc", "")
    result = {"stem": stem, "processed": False, "n_rois": 0,
              "projections": {}, "elapsed": 0.0}

    # Resumability check
    stat_path = output_dir / stem / "suite2p" / "plane0" / "stat.npy"
    if stat_path.exists():
        log.info("  %s: skipped (stat.npy exists)", stem)
        # Still extract projections if missing
        ops_path = output_dir / stem / "suite2p" / "plane0" / "ops.npy"
        if ops_path.exists():
            ops = np.load(str(ops_path), allow_pickle=True).item()
            result["projections"] = extract_projections_from_ops(
                ops, stem, annotated_dir, max_proj_dir)
            stat = np.load(str(stat_path), allow_pickle=True)
            result["n_rois"] = len(stat)
        return result

    t0 = time.time()

    # Stage TIF to local temp dir — Suite2p's file scanner needs a clean dir
    tmp_base = tempfile.mkdtemp()
    named_dir = Path(tmp_base) / stem
    named_dir.mkdir()
    os.symlink(tif_path.resolve(), named_dir / tif_path.name)

    try:
        ops = build_ops(
            named_dir, fs, tau, cfg,
            anatomical_only=anatomical_only,
            do_registration=do_registration,
            threshold_scaling=threshold_scaling,
        )
        ops["save_path0"] = str(output_dir / stem)
        ops["tiff_list"] = [str(named_dir / tif_path.name)]

        output_ops = run_s2p(ops=ops)

        n_rois = output_ops.get("nROIs", "?")
        result["n_rois"] = n_rois
        result["processed"] = True

        # Extract projections from completed ops
        result["projections"] = extract_projections_from_ops(
            output_ops, stem, annotated_dir, max_proj_dir)

    finally:
        shutil.rmtree(tmp_base, ignore_errors=True)

    result["elapsed"] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Run Suite2p + extract projections for the three-branch pipeline.")

    # Required
    ap.add_argument("--fs", type=float, required=True,
                    help="Acquisition frame rate in Hz (REQUIRED)")

    # Input (mutually exclusive)
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input_dir",
                     help="Directory containing raw .tif stacks")
    inp.add_argument("--single_file",
                     help="Path to a single .tif stack")

    # Processing
    ap.add_argument("--batch", action="store_true",
                    help="Process each .tif individually (required for mixed dimensions)")
    ap.add_argument("--tau", type=float, default=1.0,
                    help="GCaMP decay time constant in seconds (default: 1.0)")
    ap.add_argument("--anatomical_only", type=int, default=0, choices=[0, 1],
                    help="0=activity detection (default); 1=anatomy detection")
    ap.add_argument("--threshold_scaling", type=float, default=None,
                    help="Detection sensitivity (default: 0.5 from config). "
                         "Lower = more permissive")

    # Registration
    reg = ap.add_mutually_exclusive_group()
    reg.add_argument("--no_registration", action="store_true",
                     help="Force registration OFF (overrides auto-detect)")
    reg.add_argument("--force_registration", action="store_true",
                     help="Force registration ON even for _mc files")

    # Output
    ap.add_argument("--s2p_out", default=None,
                    help=f"Suite2p output directory (default: {S2P_OUT})")
    ap.add_argument("--annotated_dir", default=None,
                    help=f"Directory for mean/Vcorr projections (default: {ANNOTATED_DIR})")
    ap.add_argument("--max_proj_dir", default=None,
                    help=f"Directory for max projections (default: {MAX_PROJ_DIR})")

    # Config & mode
    ap.add_argument("--config", default=None,
                    help="Path to pipeline YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate inputs and print plan without processing")

    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)

    # Resolve output directories
    s2p_out = Path(args.s2p_out) if args.s2p_out else S2P_OUT
    annotated_dir = Path(args.annotated_dir) if args.annotated_dir else ANNOTATED_DIR
    max_proj_dir = Path(args.max_proj_dir) if args.max_proj_dir else MAX_PROJ_DIR

    # Discover input TIFs
    if args.single_file:
        src = Path(args.single_file)
        if not src.exists():
            ap.error(f"File not found: {src}")
        tif_files = [src]
    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            ap.error(f"Directory not found: {input_dir}")
        tif_files = sorted(input_dir.glob("*.tif"))
        if not tif_files:
            ap.error(f"No .tif files found in {input_dir}")
        if not args.batch and len(tif_files) > 1:
            ap.error(f"Found {len(tif_files)} TIFs — use --batch to process individually")

    # Print plan
    log.info("Suite2p pipeline — %d FOV(s)", len(tif_files))
    log.info("  Frame rate:          %.1f Hz", args.fs)
    log.info("  Tau:                 %.1f s", args.tau)
    log.info("  Anatomical only:     %d", args.anatomical_only)
    log.info("  Threshold scaling:   %s",
             args.threshold_scaling if args.threshold_scaling is not None
             else f"{cfg.get('suite2p', {}).get('threshold_scaling', 0.5)} (config)")
    log.info("  Output:              %s", s2p_out)
    log.info("  Projections (mean):  %s", annotated_dir)
    log.info("  Projections (max):   %s", max_proj_dir)

    for tif in tif_files:
        reg = should_register(tif, args.force_registration, args.no_registration)
        log.info("  %s — registration %s", tif.name, "ON" if reg else "OFF (pre-MC)")

    if args.dry_run:
        log.info("DRY RUN — exiting without processing")
        return

    # Process
    s2p_out.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = n_err = 0
    total = len(tif_files)

    for i, tif in enumerate(tif_files, 1):
        stem = tif.stem.replace("_mc", "")
        do_reg = should_register(tif, args.force_registration, args.no_registration)
        log.info("[%d/%d] %s (registration %s)",
                 i, total, stem, "ON" if do_reg else "OFF")

        try:
            result = run_one_fov(
                tif_path=tif,
                output_dir=s2p_out,
                fs=args.fs,
                tau=args.tau,
                do_registration=do_reg,
                anatomical_only=args.anatomical_only,
                threshold_scaling=args.threshold_scaling,
                cfg=cfg,
                annotated_dir=annotated_dir,
                max_proj_dir=max_proj_dir,
            )
            if result["processed"]:
                log.info("  → %s ROIs, %.0fs", result["n_rois"], result["elapsed"])
                n_done += 1
            else:
                n_skip += 1
        except Exception as exc:
            log.error("  ERROR: %s — %s", stem, exc)
            n_err += 1

    # Summary
    parts = [f"{n_done} processed"]
    if n_skip:
        parts.append(f"{n_skip} skipped")
    if n_err:
        parts.append(f"{n_err} errors")
    log.info("Complete: %s → %s", ", ".join(parts), s2p_out)


if __name__ == "__main__":
    main()
