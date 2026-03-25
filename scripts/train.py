#!/usr/bin/env python3
"""
ROI G. Biv — Cellpose fine-tuning script.
Usage:
  python train.py --run_id run009 [--epochs 300] [--lr 0.001] [--seed 42] [--val_frac 0.15]
  python train.py --run_id run009 --data_dir data/training --masks_dir data/training
  python train.py --run_id run009 --base_model models/checkpoints/models/run008_epoch_0220
"""
import argparse, logging, sys, time
from pathlib import Path
import numpy as np
import tifffile
from cellpose import models, io, train, metrics
import torch

from config import BASE_DIR, load_config

# ── Defaults ──────────────────────────────────────────────────────────
DATA_DIR    = BASE_DIR / 'data' / 'annotated'
MASKS_DIR   = BASE_DIR / 'data' / 'masks'
MODELS_DIR  = BASE_DIR / 'models' / 'checkpoints'
LOGS_DIR    = BASE_DIR / 'logs'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset(data_dir, masks_dir, seed=42, use_vcorr=False):
    """Load images and masks grouped by FOV, shuffled with seeded RNG.

    If use_vcorr=True, loads Vcorr maps and stacks them as a 2nd channel:
      image shape becomes (H, W, 2) = [projection, Vcorr].
    FOVs missing a Vcorr file are skipped when use_vcorr is True.
    """
    fov_groups = {}
    mask_files = sorted(masks_dir.glob('*_masks.tif'))
    for mf in mask_files:
        stem = mf.stem.replace('_masks', '')
        mask = tifffile.imread(mf)
        if mask.max() == 0:
            print(f'WARNING: Empty mask {mf.name}, skipping')
            continue

        # Load Vcorr for this FOV (shared across mean/max projections)
        vcorr = None
        if use_vcorr:
            vcorr_path = data_dir / f'{stem}_vcorr.tif'
            if not vcorr_path.exists():
                print(f'WARNING: No Vcorr for {stem}, skipping FOV')
                continue
            vcorr = tifffile.imread(vcorr_path).astype(np.float32)

        pairs = []
        for proj in ('_mean', '_max'):
            img_path = data_dir / f'{stem}{proj}.tif'
            if not img_path.exists():
                print(f'WARNING: Missing {img_path}, skipping')
                continue
            img = tifffile.imread(img_path).astype(np.float32)

            if use_vcorr and vcorr is not None:
                # Stack as 2-channel: [projection, Vcorr]
                img = np.stack([img, vcorr], axis=-1)

            pairs.append((img, mask))
        if pairs:
            fov_groups[stem] = pairs

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_id', required=True)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--base_model', default='cyto3',
                    help='Path to previous checkpoint or Cellpose model name')
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--diameter', type=float, default=17,
                    help='Cell diameter for post-training AP eval (default: 17 for neurons)')
    ap.add_argument('--data_dir', default=None,
                    help='Directory with *_mean.tif and *_max.tif projections (default: data/annotated)')
    ap.add_argument('--masks_dir', default=None,
                    help='Directory with *_masks.tif labeled masks (default: data/masks)')
    ap.add_argument('--use_vcorr', action='store_true', default=True,
                    help='Use Vcorr temporal maps as 2nd channel (default: True)')
    ap.add_argument('--no_vcorr', action='store_true',
                    help='Disable Vcorr — train on single-channel projections only')
    args = ap.parse_args()

    use_vcorr = args.use_vcorr and not args.no_vcorr

    # Resolve data directories (CLI overrides defaults)
    data_dir  = Path(args.data_dir)  if args.data_dir  else DATA_DIR
    masks_dir = Path(args.masks_dir) if args.masks_dir else MASKS_DIR
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir
    if not masks_dir.is_absolute():
        masks_dir = BASE_DIR / masks_dir

    log_path = LOGS_DIR / f'{args.run_id}.log'
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    log = logging.getLogger('roigbiv')

    log.info(f'PyTorch CUDA: {torch.cuda.is_available()}')
    log.info(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    log.info(f'Run ID: {args.run_id}, base model: {args.base_model}, seed: {args.seed}')
    log.info(f'Data dir:  {data_dir}')
    log.info(f'Masks dir: {masks_dir}')
    log.info(f'Vcorr 2nd channel: {use_vcorr}')

    # Channel config: [1, 2] = use both channels; [0, 0] = grayscale only
    channels = [1, 2] if use_vcorr else [0, 0]

    # Load dataset with FOV-aware shuffled split
    images, masks = load_dataset(data_dir, masks_dir, seed=args.seed,
                                 use_vcorr=use_vcorr)
    log.info(f'Loaded {len(images)} pairs')
    assert len(images) >= 3, 'Need at least 3 pairs for training'

    # FOV-aware train/val split
    n_val = max(1, int(len(images) * args.val_frac))
    train_imgs,  val_imgs  = images[n_val:], images[:n_val]
    train_masks, val_masks = masks[n_val:],  masks[:n_val]
    log.info(f'Loaded {len(images)} pairs, Train: {len(train_imgs)}, Val: {len(val_imgs)} (val_frac={args.val_frac})')

    # Initialize model — support both model names (cyto3) and checkpoint paths
    base_model = args.base_model
    base_path = Path(base_model)
    if not base_path.is_absolute():
        base_path = BASE_DIR / base_path
    if base_path.exists():
        log.info(f'Loading from checkpoint: {base_path}')
        model = models.CellposeModel(gpu=True, pretrained_model=str(base_path))
    else:
        log.info(f'Using base model: {base_model}')
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
        min_train_masks=5,
        save_path=str(MODELS_DIR),
        save_every=10,
        save_each=True,
        model_name=args.run_id,
    )
    elapsed = time.time() - t0
    log.info(f'Training complete in {elapsed/60:.1f} min. Model: {model_path}')

    # Identify best checkpoint by val loss
    test_losses = np.array(test_losses)
    val_recorded = np.where(test_losses > 0)[0]
    if len(val_recorded) > 0:
        best_epoch = val_recorded[np.argmin(test_losses[val_recorded])]
        log.info(f'Best val loss {test_losses[best_epoch]:.4f} at epoch {best_epoch}')

    # Post-training AP evaluation on best checkpoint
    if len(val_recorded) > 0:
        best_ckpt = MODELS_DIR / 'models' / f'{args.run_id}_epoch_{best_epoch:04d}'
        eval_path = str(best_ckpt) if best_ckpt.exists() else str(model_path)
        log.info(f'AP eval using: {eval_path}')
    else:
        eval_path = str(model_path)
    model_eval = models.CellposeModel(gpu=True, pretrained_model=eval_path)
    pred_masks, _, _ = model_eval.eval(val_imgs, diameter=args.diameter, channels=channels,
                                       batch_size=args.batch_size)
    ap, tp, fp, fn = metrics.average_precision(val_masks, pred_masks,
                                               threshold=[0.5, 0.75, 0.9])
    log.info(f'Val AP@0.5={ap[:,0].mean():.4f}, AP@0.75={ap[:,1].mean():.4f}, AP@0.9={ap[:,2].mean():.4f}')

if __name__ == '__main__':
    main()
