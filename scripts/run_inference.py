#!/usr/bin/env python3
"""
ROI G. Biv --- Branch A: Cellpose spatial segmentation.

Runs Cellpose on mean projections (with optional Vcorr as 2nd channel) to
produce uint16 ROI masks. Designed for GRIN lens data with permissive
thresholds and tile normalization to handle vignetting.

Features:
  - Dual-channel [mean, Vcorr] input by default (--no_vcorr for single-channel)
  - Tile normalization (block=128) compensates for GRIN vignetting
  - Optional Cellpose3 denoising (--denoise)
  - Permissive defaults for deep brain GRIN data (cellprob=-2.0, flow=0.6)
  - --dry-run validates inputs without processing

Usage:
  python run_inference.py --input_dir data/annotated --diameter 12
  python run_inference.py --input_dir data/annotated --denoise --dry-run
  python run_inference.py --input_dir data/annotated --no_vcorr --diameter 12
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import tifffile
from cellpose import models

from config import BASE_DIR, load_config

LOG_DIR = BASE_DIR / "logs"

log = logging.getLogger("run_inference")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure logging to console and logs/inference.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOG_DIR / "inference.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model(model_path: Path, denoise: bool = False):
    """Load Cellpose model, optionally with Cellpose3 denoising wrapper.

    Parameters
    ----------
    model_path : path to fine-tuned model checkpoint
    denoise    : if True, wrap with CellposeDenoiseModel for image restoration

    Returns
    -------
    model      : Cellpose model instance
    has_denoise: bool — whether denoising is active
    """
    if denoise:
        try:
            from cellpose import denoise as cp_denoise
            model = cp_denoise.CellposeDenoiseModel(
                gpu=True,
                pretrained_model=str(model_path),
                restore_type="denoise_cyto3",
            )
            return model, True
        except (ImportError, AttributeError, Exception) as exc:
            log.warning("Cellpose3 denoising unavailable (%s), falling back to standard model", exc)

    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    return model, False


def run_inference_fov(
    model,
    img: np.ndarray,
    diameter: float,
    channels: list,
    flow_threshold: float,
    cellprob_threshold: float,
    tile_norm_blocksize: int,
    has_denoise: bool = False,
) -> tuple[np.ndarray, dict]:
    """Run Cellpose on a single FOV image.

    Returns (masks, flows_dict).
    """
    normalize_params = {"tile_norm_blocksize": tile_norm_blocksize} if tile_norm_blocksize > 0 else True

    if has_denoise:
        masks, flows, styles, imgs_dn = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize=normalize_params,
        )
    else:
        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            normalize=normalize_params,
        )

    return masks, flows


def run_inference(
    input_dir: Path,
    out_dir: Path,
    model_path: Path,
    diameter: float,
    flow_threshold: float,
    cellprob_threshold: float,
    tile_norm_blocksize: int,
    use_vcorr: bool = True,
    denoise: bool = False,
) -> list[dict]:
    """Run Branch A inference on all *_mean.tif files in input_dir.

    Parameters
    ----------
    input_dir           : directory containing *_mean.tif projections
    out_dir             : output directory for mask TIFs and summary CSV
    model_path          : path to Cellpose model checkpoint
    diameter            : expected cell diameter in pixels
    flow_threshold      : flow error threshold (higher = more permissive)
    cellprob_threshold  : cell probability threshold (lower = more permissive)
    tile_norm_blocksize : tile normalization block size (0 = default normalization)
    use_vcorr           : stack Vcorr as 2nd channel
    denoise             : use Cellpose3 denoising

    Returns
    -------
    list of result dicts with keys: stem, n_rois, mask_path
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    model, has_denoise = load_model(model_path, denoise=denoise)
    channels = [1, 2] if use_vcorr else [0, 0]

    mean_files = sorted(input_dir.glob("*_mean.tif"))
    if not mean_files:
        log.warning("No *_mean.tif files found in %s", input_dir)
        return []

    results = []
    n = len(mean_files)

    for i, tif in enumerate(mean_files, 1):
        stem = tif.stem.replace("_mean", "")
        log.info("[%d/%d] %s", i, n, stem)

        img = tifffile.imread(str(tif)).astype(np.float32)

        if use_vcorr:
            vcorr_path = input_dir / f"{stem}_vcorr.tif"
            if vcorr_path.exists():
                vcorr = tifffile.imread(str(vcorr_path)).astype(np.float32)
                img = np.stack([img, vcorr], axis=-1)
            else:
                log.warning("  No Vcorr for %s, zero-padding channel 2", stem)
                img = np.stack([img, np.zeros_like(img)], axis=-1)

        masks, flows = run_inference_fov(
            model, img, diameter, channels,
            flow_threshold, cellprob_threshold,
            tile_norm_blocksize, has_denoise,
        )

        n_rois = int(masks.max())
        mask_path = out_dir / f"{stem}_masks.tif"
        tifffile.imwrite(str(mask_path), masks.astype(np.uint16))

        results.append({"stem": stem, "n_rois": n_rois, "mask_path": str(mask_path)})
        log.info("  -> %d ROIs", n_rois)

    # Write summary CSV
    summary_path = out_dir / "inference_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "n_rois", "mask_path"])
        w.writeheader()
        w.writerows(results)
    log.info("Summary: %s", summary_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Branch A: Cellpose spatial segmentation for the three-branch pipeline.")

    ap.add_argument("--input_dir", required=True,
                    help="Directory with *_mean.tif (and *_vcorr.tif) projections")
    ap.add_argument("--output_dir", default=None,
                    help="Output directory for masks (default: inference/output)")
    ap.add_argument("--diameter", type=float, default=None,
                    help="Cell diameter in pixels (default: 12 from config)")
    ap.add_argument("--flow_threshold", type=float, default=None,
                    help="Flow error threshold (default: 0.6 from config)")
    ap.add_argument("--cellprob_threshold", type=float, default=None,
                    help="Cell probability threshold (default: -2.0 from config)")
    ap.add_argument("--tile_norm_blocksize", type=int, default=None,
                    help="Tile normalization block size (default: 128 from config, 0=off)")

    ap.add_argument("--no_vcorr", action="store_true",
                    help="Disable Vcorr 2nd channel — use single-channel mean only")
    ap.add_argument("--denoise", action="store_true", default=False,
                    help="Enable Cellpose3 image denoising before segmentation")

    ap.add_argument("--config", default=None,
                    help="Path to pipeline YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate inputs and print plan without processing")

    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    cp_cfg = cfg.get("cellpose", {})

    # Resolve model path
    model_path = Path(cp_cfg.get("model_path", ""))
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path
    if not model_path.exists():
        model_path = BASE_DIR / "models" / "deployed" / "current_model"

    # Resolve parameters: CLI > config > defaults
    diameter = args.diameter if args.diameter is not None else cp_cfg.get("diameter", 12)
    flow_threshold = args.flow_threshold if args.flow_threshold is not None else cp_cfg.get("flow_threshold", 0.6)
    cellprob_threshold = args.cellprob_threshold if args.cellprob_threshold is not None else cp_cfg.get("cellprob_threshold", -2.0)
    tile_norm_blocksize = args.tile_norm_blocksize if args.tile_norm_blocksize is not None else cp_cfg.get("tile_norm_blocksize", 128)
    denoise = args.denoise or cp_cfg.get("denoise", False)
    use_vcorr = not args.no_vcorr

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / "inference" / "output"

    # Print plan
    log.info("Branch A — Cellpose spatial segmentation")
    log.info("  Input:              %s", input_dir)
    log.info("  Output:             %s", out_dir)
    log.info("  Model:              %s", model_path)
    log.info("  Diameter:           %.0f px", diameter)
    log.info("  Flow threshold:     %.2f", flow_threshold)
    log.info("  Cellprob threshold: %.1f", cellprob_threshold)
    log.info("  Tile norm block:    %d", tile_norm_blocksize)
    log.info("  Vcorr 2nd channel:  %s", "ON" if use_vcorr else "OFF")
    log.info("  Denoise:            %s", "ON" if denoise else "OFF")

    if not input_dir.exists():
        log.error("Input directory not found: %s", input_dir)
        return

    mean_files = sorted(input_dir.glob("*_mean.tif"))
    log.info("  Found %d *_mean.tif files", len(mean_files))

    if args.dry_run:
        for f in mean_files:
            stem = f.stem.replace("_mean", "")
            vcorr_exists = (input_dir / f"{stem}_vcorr.tif").exists()
            log.info("    %s  vcorr=%s", f.name, "yes" if vcorr_exists else "MISSING")
        log.info("DRY RUN — exiting without processing")
        return

    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        return

    results = run_inference(
        input_dir=input_dir,
        out_dir=out_dir,
        model_path=model_path,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        tile_norm_blocksize=tile_norm_blocksize,
        use_vcorr=use_vcorr,
        denoise=denoise,
    )

    # Final summary
    total_rois = sum(r["n_rois"] for r in results)
    log.info("Complete: %d FOVs, %d total ROIs -> %s", len(results), total_rois, out_dir)


if __name__ == "__main__":
    main()
