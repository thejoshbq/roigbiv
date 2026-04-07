#!/usr/bin/env python3
"""
ROI G. Biv --- Branch C: Tonic neuron detection.

Detects tonically active neurons via bandpass filtering + local correlation
clustering in SVD space. These neurons are missed by Suite2p (low temporal
variance) and often missed by Cellpose (dim or low-contrast in mean images).

Pipeline per FOV:
  1. Stream data.bin -> truncated SVD (temporal compression)
  2. Bandpass filter temporal components (0.05-2.0 Hz neuronal band)
  3. Local correlation map in SVD space (soma-radius neighborhood)
  4. Threshold + connected-component + size filter -> uint16 masks

Outputs per FOV:
  {stem}_tonic_masks.tif  — uint16 labeled ROI mask
  {stem}_corr_map.tif     — float32 local correlation map (for QC)

Usage:
  python detect_tonic.py --s2p_dir suite2p_workspace/output --fs 30.0
  python detect_tonic.py --s2p_dir suite2p_workspace/output --fs 30.0 --band astrocyte
  python detect_tonic.py --s2p_dir suite2p_workspace/output --fs 30.0 --dry-run
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import tifffile

import sys
from config import BASE_DIR, load_config
sys.path.insert(0, str(BASE_DIR))
from roigbiv.tonic import run_tonic_detection

S2P_OUT = BASE_DIR / "suite2p_workspace" / "output"
TONIC_OUT = BASE_DIR / "inference" / "tonic"
LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("detect_tonic")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to console and logs/tonic.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "tonic.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)


# ---------------------------------------------------------------------------
# FOV discovery
# ---------------------------------------------------------------------------

def discover_fovs(s2p_dir: Path) -> list[dict]:
    """Find FOV directories with data.bin and ops.npy.

    Returns list of dicts with keys: stem, bin_path, ops_path, ready.
    """
    fovs = []
    for fov_dir in sorted(d for d in s2p_dir.iterdir() if d.is_dir()):
        plane_dir = fov_dir / "suite2p" / "plane0"
        bin_path = plane_dir / "data.bin"
        ops_path = plane_dir / "ops.npy"
        fovs.append({
            "stem": fov_dir.name,
            "bin_path": bin_path,
            "ops_path": ops_path,
            "ready": bin_path.exists() and ops_path.exists(),
        })
    return fovs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Branch C: Tonic neuron detection via bandpass SVD + local correlation.")

    ap.add_argument("--fs", type=float, required=True,
                    help="Acquisition frame rate in Hz (REQUIRED)")
    ap.add_argument("--s2p_dir", default=None,
                    help=f"Suite2p output directory (default: {S2P_OUT})")
    ap.add_argument("--output_dir", default=None,
                    help=f"Output directory for masks and corr maps (default: {TONIC_OUT})")

    # Band selection
    ap.add_argument("--band", default="neuronal", choices=["neuronal", "astrocyte"],
                    help="Frequency band: neuronal (0.05-2.0 Hz) or astrocyte (0.01-0.3 Hz)")
    ap.add_argument("--band_lo", type=float, default=None,
                    help="Override low cutoff frequency in Hz")
    ap.add_argument("--band_hi", type=float, default=None,
                    help="Override high cutoff frequency in Hz")

    # Detection parameters
    ap.add_argument("--n_components", type=int, default=None,
                    help="SVD components (default: 500 from config)")
    ap.add_argument("--soma_radius", type=int, default=None,
                    help="Local correlation radius in pixels (default: 8 from config)")
    ap.add_argument("--corr_threshold", type=float, default=None,
                    help="Correlation threshold (default: 0.25 from config)")
    ap.add_argument("--min_size", type=int, default=None,
                    help="Minimum ROI area in pixels (default: 80 from config)")
    ap.add_argument("--max_size", type=int, default=None,
                    help="Maximum ROI area in pixels (default: 350 from config)")
    ap.add_argument("--min_solidity", type=float, default=None,
                    help="Minimum cluster solidity [0,1]; rejects spindly neuropil "
                         "(default: 0.6 from config)")
    ap.add_argument("--max_eccentricity", type=float, default=None,
                    help="Maximum eccentricity [0,1]; rejects elongated axon fragments "
                         "(default: 0.85 from config)")

    # Config & mode
    ap.add_argument("--config", default=None,
                    help="Path to pipeline YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Discover FOVs and print plan without processing")

    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    tc = cfg.get("tonic", {})

    # Resolve directories
    s2p_dir = Path(args.s2p_dir) if args.s2p_dir else S2P_OUT
    out_dir = Path(args.output_dir) if args.output_dir else TONIC_OUT

    # Resolve parameters: CLI > config > defaults
    n_components = args.n_components or tc.get("n_components", 500)
    chunk_frames = tc.get("chunk_frames", 500)
    soma_radius = args.soma_radius if args.soma_radius is not None else tc.get("soma_radius", 8)
    corr_threshold = args.corr_threshold if args.corr_threshold is not None else tc.get("corr_threshold", 0.25)
    min_size = args.min_size if args.min_size is not None else tc.get("min_roi_pixels", 80)
    max_size = args.max_size if args.max_size is not None else tc.get("max_roi_pixels", 350)
    filter_order = tc.get("filter_order", 3)
    min_solidity     = args.min_solidity     if args.min_solidity     is not None else tc.get("min_solidity",     0.6)
    max_eccentricity = args.max_eccentricity if args.max_eccentricity is not None else tc.get("max_eccentricity", 0.85)

    # Resolve frequency band
    band = args.band
    if args.band_lo is not None and args.band_hi is not None:
        band_lo, band_hi = args.band_lo, args.band_hi
    elif band == "astrocyte":
        band_lo = tc.get("astrocyte_band_lo", 0.01)
        band_hi = tc.get("astrocyte_band_hi", 0.3)
    else:
        band_lo = tc.get("band_lo", 0.05)
        band_hi = tc.get("band_hi", 2.0)

    # Print plan
    log.info("Branch C — Tonic neuron detection")
    log.info("  Suite2p dir:       %s", s2p_dir)
    log.info("  Output:            %s", out_dir)
    log.info("  Frame rate:        %.1f Hz", args.fs)
    log.info("  Band:              %s (%.3f - %.1f Hz)", band, band_lo, band_hi)
    log.info("  SVD components:    %d", n_components)
    log.info("  Soma radius:       %d px", soma_radius)
    log.info("  Corr threshold:    %.2f", corr_threshold)
    log.info("  Size filter:       %d - %d px", min_size, max_size)
    log.info("  Morph filters:     solidity >= %.2f, eccentricity <= %.2f",
             min_solidity, max_eccentricity)

    if not s2p_dir.exists():
        log.error("Suite2p directory not found: %s", s2p_dir)
        return

    fovs = discover_fovs(s2p_dir)
    ready = [f for f in fovs if f["ready"]]
    not_ready = [f for f in fovs if not f["ready"]]

    log.info("  FOVs found: %d total, %d ready (have data.bin + ops.npy)", len(fovs), len(ready))
    if not_ready:
        for f in not_ready:
            missing = []
            if not f["bin_path"].exists():
                missing.append("data.bin")
            if not f["ops_path"].exists():
                missing.append("ops.npy")
            log.warning("  %s: missing %s", f["stem"], ", ".join(missing))

    if args.dry_run:
        for f in ready:
            ops = np.load(str(f["ops_path"]), allow_pickle=True).item()
            nframes = ops.get("nframes", "?")
            Ly, Lx = ops.get("Ly", "?"), ops.get("Lx", "?")
            bin_mb = f["bin_path"].stat().st_size / 1e6
            log.info("    %s: %s frames, %sx%s, data.bin=%.0f MB",
                     f["stem"], nframes, Ly, Lx, bin_mb)
        log.info("DRY RUN — exiting without processing")
        return

    if not ready:
        log.error("No FOVs ready for processing")
        return

    # Process
    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = n_err = 0
    total = len(ready)

    for i, fov in enumerate(ready, 1):
        stem = fov["stem"]
        mask_path = out_dir / f"{stem}_tonic_masks.tif"

        # Resumability: skip if masks already exist
        if mask_path.exists():
            log.info("[%d/%d] %s: skipped (masks exist)", i, total, stem)
            n_skip += 1
            continue

        log.info("[%d/%d] %s ...", i, total, stem)
        t0 = time.time()

        try:
            masks, corr_map, info = run_tonic_detection(
                bin_path=fov["bin_path"],
                ops_path=fov["ops_path"],
                fs=args.fs,
                band=band,
                n_components=n_components,
                chunk_frames=chunk_frames,
                soma_radius=soma_radius,
                corr_threshold=corr_threshold,
                min_size=min_size,
                max_size=max_size,
                filter_order=filter_order,
                band_lo=band_lo,
                band_hi=band_hi,
                min_solidity=min_solidity,
                max_eccentricity=max_eccentricity,
            )

            tifffile.imwrite(str(mask_path), masks)
            corr_path = out_dir / f"{stem}_corr_map.tif"
            tifffile.imwrite(str(corr_path), corr_map)

            elapsed = time.time() - t0
            log.info("  -> %d tonic ROIs (%.0fs)", info["n_rois"], elapsed)
            n_done += 1

        except Exception as exc:
            log.error("  ERROR: %s — %s", stem, exc)
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
