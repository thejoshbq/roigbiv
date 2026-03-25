#!/usr/bin/env python3
"""
ROI G. Biv — Standalone AP evaluation for any checkpoint.
Usage:
  python eval_model.py --model run011_epoch_0060 --diameter 17
  python eval_model.py --model models/checkpoints/models/run011_epoch_0060 --diameter 0
  python eval_model.py --model run011_epoch_0060 --no_vcorr --diameter 17
"""
import argparse, logging
from pathlib import Path
import numpy as np
from cellpose import models, metrics

from config import BASE_DIR
from train import load_dataset

MODELS_DIR = BASE_DIR / 'models' / 'checkpoints' / 'models'
LOGS_DIR   = BASE_DIR / 'logs'
DATA_DIR   = BASE_DIR / 'data' / 'annotated'
MASKS_DIR  = BASE_DIR / 'data' / 'masks'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help='Checkpoint name (e.g. run011_epoch_0060) or full path')
    ap.add_argument('--diameter', type=float, default=17,
                    help='Cell diameter for inference rescaling (0 = auto-estimate)')
    ap.add_argument('--val_frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--use_vcorr', action='store_true', default=True)
    ap.add_argument('--no_vcorr', action='store_true')
    ap.add_argument('--data_dir', default=None)
    ap.add_argument('--masks_dir', default=None)
    args = ap.parse_args()

    use_vcorr = args.use_vcorr and not args.no_vcorr
    channels  = [1, 2] if use_vcorr else [0, 0]

    # Resolve model path — accept short name or full path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        candidate = MODELS_DIR / args.model
        if candidate.exists():
            model_path = candidate
        else:
            model_path = BASE_DIR / args.model

    # Append to shared eval log (never overwrites run-specific logs)
    log_path = LOGS_DIR / 'eval.log'
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    log = logging.getLogger('roigbiv.eval')

    data_dir  = Path(args.data_dir)  if args.data_dir  else DATA_DIR
    masks_dir = Path(args.masks_dir) if args.masks_dir else MASKS_DIR

    # Load dataset with same split as training
    images, masks_list = load_dataset(data_dir, masks_dir, seed=args.seed,
                                      use_vcorr=use_vcorr)
    n_val = max(1, int(len(images) * args.val_frac))
    val_imgs, val_masks = images[:n_val], masks_list[:n_val]
    log.info(f'Val set: {len(val_imgs)} images (val_frac={args.val_frac}, seed={args.seed})')

    log.info(f'Model:    {model_path}')
    log.info(f'Diameter: {args.diameter}  Channels: {channels}  Vcorr: {use_vcorr}')

    model_eval = models.CellposeModel(gpu=True, pretrained_model=str(model_path))
    pred_masks, _, _ = model_eval.eval(val_imgs, diameter=args.diameter,
                                       channels=channels, batch_size=args.batch_size)
    ap_scores, _, _, _ = metrics.average_precision(val_masks, pred_masks,
                                                   threshold=[0.5, 0.75, 0.9])
    log.info(f'AP@0.5={ap_scores[:,0].mean():.4f}, '
             f'AP@0.75={ap_scores[:,1].mean():.4f}, '
             f'AP@0.9={ap_scores[:,2].mean():.4f}')


if __name__ == '__main__':
    main()
