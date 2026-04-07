#!/usr/bin/env python3
"""
ROI G. Biv --- Cellpose fine-tuning for GRIN lens neuron segmentation.

Fine-tunes from cyto3 base model (never from a previous checkpoint) on
annotated mean projections with optional Vcorr as 2nd channel.

HITL workflow: run inference -> correct in Cellpose GUI -> ingest corrections
-> retrain from cyto3 base -> repeat until <5% ROI changes per round.

Usage:
  python train.py --run_id run001
  python train.py --run_id run001 --epochs 200 --lr 0.05 --diameter 12
  python train.py --run_id run001 --no_vcorr --base_model cyto3
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose import models, train, metrics

from config import BASE_DIR, load_config

# ── Defaults ──────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data" / "annotated"
MASKS_DIR = BASE_DIR / "data" / "masks"
MODELS_DIR = BASE_DIR / "models" / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(run_id: str) -> None:
    """Configure logging to console and logs/{run_id}.log."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(LOGS_DIR / f"{run_id}.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _load_fov_groups(
    data_dir: Path,
    masks_dir: Path,
    use_vcorr: bool,
    fov_groups: dict,
) -> None:
    """Scan a single masks_dir and add FOV groups (mutates *fov_groups* in place).

    Stems already present in *fov_groups* are skipped (deduplication).
    """
    mask_files = sorted(masks_dir.glob("*_masks.tif"))

    for mf in mask_files:
        stem = mf.stem.replace("_masks", "")

        if stem in fov_groups:
            log.info("Dedup: skipping %s (already loaded)", stem)
            continue

        mask = tifffile.imread(str(mf))
        if mask.max() == 0:
            log.warning("Empty mask %s, skipping", mf.name)
            continue

        # Load Vcorr for this FOV (shared across mean/max projections)
        vcorr = None
        if use_vcorr:
            vcorr_path = data_dir / f"{stem}_vcorr.tif"
            if not vcorr_path.exists():
                log.warning("No Vcorr for %s, skipping FOV", stem)
                continue
            vcorr = tifffile.imread(str(vcorr_path)).astype(np.float32)

        pairs = []
        for proj in ("_mean", "_max"):
            img_path = data_dir / f"{stem}{proj}.tif"
            if not img_path.exists():
                continue
            img = tifffile.imread(str(img_path)).astype(np.float32)

            if use_vcorr and vcorr is not None:
                img = np.stack([img, vcorr], axis=-1)

            pairs.append((img, mask))

        if pairs:
            fov_groups[stem] = pairs


def load_dataset(
    data_dir: Path,
    masks_dir: Path,
    seed: int = 42,
    use_vcorr: bool = True,
    extra_dirs: list[tuple[Path, Path]] | None = None,
) -> tuple[list, list]:
    """Load images and masks grouped by FOV, shuffled with seeded RNG.

    If use_vcorr=True, loads Vcorr maps and stacks as 2nd channel:
      image shape becomes (H, W, 2) = [projection, Vcorr].
    FOVs missing a Vcorr file are skipped when use_vcorr is True.

    extra_dirs: list of (data_dir, masks_dir) tuples for additional data.
    Stems already loaded from the primary directories take precedence.

    Returns (images, masks) lists.
    """
    fov_groups = {}

    # Load primary dataset
    _load_fov_groups(data_dir, masks_dir, use_vcorr, fov_groups)

    # Load extra datasets (deduplicated by stem)
    if extra_dirs:
        for extra_data, extra_masks in extra_dirs:
            log.info("Loading extra data: %s", extra_masks)
            _load_fov_groups(extra_data, extra_masks, use_vcorr, fov_groups)

    # Shuffle FOV groups with seeded RNG so mean/max stay together
    rng = np.random.default_rng(seed)
    fov_stems = list(fov_groups.keys())
    rng.shuffle(fov_stems)

    images, masks = [], []
    for stem in fov_stems:
        for img, mask in fov_groups[stem]:
            images.append(img)
            masks.append(mask)

    return images, masks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune Cellpose for GRIN lens neuron segmentation.")

    ap.add_argument("--run_id", required=True,
                    help="Unique run identifier (used for model name and log file)")
    ap.add_argument("--epochs", type=int, default=200,
                    help="Training epochs (default: 200)")
    ap.add_argument("--lr", type=float, default=0.05,
                    help="Learning rate (default: 0.05)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Training batch size (default: 8)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible splits (default: 42)")
    ap.add_argument("--base_model", default="cyto3",
                    help="Base model name (default: cyto3). WARNING: always retrain "
                         "from base, never from a previous checkpoint.")
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="Fraction of data for validation (default: 0.15)")
    ap.add_argument("--diameter", type=float, default=12,
                    help="Cell diameter for AP evaluation (default: 12)")

    ap.add_argument("--data_dir", default=None,
                    help="Directory with *_mean.tif and *_max.tif (default: data/annotated)")
    ap.add_argument("--masks_dir", default=None,
                    help="Directory with *_masks.tif (default: data/masks)")

    ap.add_argument("--use_vcorr", action="store_true", default=True,
                    help="Use Vcorr as 2nd channel (default: True)")
    ap.add_argument("--no_vcorr", action="store_true",
                    help="Disable Vcorr — train on single-channel projections only")

    ap.add_argument("--extra_data_dir", nargs="+", default=None,
                    help="Additional data directories (space-separated)")
    ap.add_argument("--extra_masks_dir", nargs="+", default=None,
                    help="Additional masks directories (must match --extra_data_dir)")

    ap.add_argument("--config", default=None,
                    help="Path to pipeline YAML config")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load data, print stats, exit without training")

    args = ap.parse_args()
    setup_logging(args.run_id)

    use_vcorr = args.use_vcorr and not args.no_vcorr

    # Warn if using a checkpoint instead of base model
    base_model = args.base_model
    base_path = Path(base_model)
    if not base_path.is_absolute():
        base_path = BASE_DIR / base_path
    if base_path.exists() and base_model != "cyto3":
        log.warning("Training from checkpoint %s. The algorithm recommends "
                     "always retraining from cyto3 base, not checkpoints.", base_model)

    # Resolve data directories
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    masks_dir = Path(args.masks_dir) if args.masks_dir else MASKS_DIR
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir
    if not masks_dir.is_absolute():
        masks_dir = BASE_DIR / masks_dir

    log.info("PyTorch CUDA: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        log.info("GPU: %s", torch.cuda.get_device_name(0))
    log.info("Run ID: %s, base model: %s, seed: %d", args.run_id, args.base_model, args.seed)
    log.info("Data dir:  %s", data_dir)
    log.info("Masks dir: %s", masks_dir)
    log.info("Vcorr 2nd channel: %s", use_vcorr)
    log.info("Epochs: %d, LR: %.4f, Batch: %d", args.epochs, args.lr, args.batch_size)

    channels = [1, 2] if use_vcorr else [0, 0]

    # Build extra_dirs list
    extra_dirs = None
    if args.extra_data_dir and args.extra_masks_dir:
        if len(args.extra_data_dir) != len(args.extra_masks_dir):
            log.error("--extra_data_dir and --extra_masks_dir must have the same number of entries")
            sys.exit(1)
        extra_dirs = []
        for ed, em in zip(args.extra_data_dir, args.extra_masks_dir):
            edp = Path(ed) if Path(ed).is_absolute() else BASE_DIR / ed
            emp = Path(em) if Path(em).is_absolute() else BASE_DIR / em
            extra_dirs.append((edp, emp))
        log.info("Extra data dirs: %s", extra_dirs)
    elif args.extra_data_dir or args.extra_masks_dir:
        log.error("Both --extra_data_dir and --extra_masks_dir are required together")
        sys.exit(1)

    # Load dataset
    images, masks = load_dataset(data_dir, masks_dir, seed=args.seed,
                                 use_vcorr=use_vcorr, extra_dirs=extra_dirs)
    log.info("Loaded %d image-mask pairs", len(images))

    if len(images) < 3:
        log.error("Need at least 3 pairs for training, got %d", len(images))
        sys.exit(1)

    # FOV-aware train/val split
    n_val = max(1, int(len(images) * args.val_frac))
    train_imgs, val_imgs = images[n_val:], images[:n_val]
    train_masks, val_masks = masks[n_val:], masks[:n_val]
    log.info("Train: %d, Val: %d (val_frac=%.2f)", len(train_imgs), len(val_imgs), args.val_frac)

    if args.dry_run:
        log.info("DRY RUN — exiting without training")
        return

    # Initialize model
    if base_path.exists() and base_model != "cyto3":
        log.info("Loading from checkpoint: %s", base_path)
        model = models.CellposeModel(gpu=True, pretrained_model=str(base_path))
    else:
        log.info("Using base model: %s", base_model)
        model = models.CellposeModel(gpu=True, model_type=base_model)

    # Train
    t0 = time.time()
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=train_imgs,
        train_labels=train_masks,
        test_data=val_imgs,
        test_labels=val_masks,
        channels=channels,
        normalize=True,
        learning_rate=args.lr,
        weight_decay=1e-4,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        min_train_masks=1,
        save_path=str(MODELS_DIR),
        save_every=10,
        save_each=True,
        model_name=args.run_id,
    )
    elapsed = time.time() - t0
    log.info("Training complete in %.1f min. Model: %s", elapsed / 60, model_path)

    # Identify best checkpoint by val loss
    test_losses = np.array(test_losses)
    val_recorded = np.where(test_losses > 0)[0]
    if len(val_recorded) > 0:
        best_epoch = val_recorded[np.argmin(test_losses[val_recorded])]
        log.info("Best val loss %.4f at epoch %d", test_losses[best_epoch], best_epoch)
        best_ckpt = MODELS_DIR / "models" / f"{args.run_id}_epoch_{best_epoch:04d}"
        eval_path = str(best_ckpt) if best_ckpt.exists() else str(model_path)
    else:
        eval_path = str(model_path)

    # Post-training AP evaluation
    log.info("AP eval using: %s", eval_path)
    model_eval = models.CellposeModel(gpu=True, pretrained_model=eval_path)
    pred_masks, _, _ = model_eval.eval(
        val_imgs, diameter=args.diameter, channels=channels,
        batch_size=args.batch_size,
    )
    ap, tp, fp, fn = metrics.average_precision(
        val_masks, pred_masks, threshold=[0.5, 0.75, 0.9],
    )
    log.info("Val AP@0.5=%.4f, AP@0.75=%.4f, AP@0.9=%.4f",
             ap[:, 0].mean(), ap[:, 1].mean(), ap[:, 2].mean())


if __name__ == "__main__":
    main()
