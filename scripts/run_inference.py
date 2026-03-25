#!/usr/bin/env python3
"""
ROI G. Biv — Batch inference on new imaging sessions.
Outputs: uint16 mask .tif and summary CSV per FOV.
Usage:
  python run_inference.py --input_dir ~/Otis-Lab/Projects/roigbiv/data/annotated --diameter 30
  python run_inference.py --config configs/pipeline.yaml --input_dir ~/data
"""
import argparse, csv
from pathlib import Path
import numpy as np
import tifffile
from cellpose import models

from config import BASE_DIR, load_config

MODEL_PATH = BASE_DIR / 'models' / 'deployed' / 'current_model'

def run_inference(input_dir, diameter, out_dir, vcorr_dir=None, cfg=None):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cp_cfg = (cfg or {}).get('cellpose', {})

    model_path = Path(cp_cfg.get('model_path', ''))
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path
    if not model_path.exists():
        model_path = MODEL_PATH

    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))

    # If Vcorr dir provided, use 2-channel input [mean, Vcorr]
    use_vcorr = vcorr_dir is not None
    channels = [1, 2] if use_vcorr else cp_cfg.get('channels', [0, 0])
    if use_vcorr:
        vcorr_dir = Path(vcorr_dir)

    flow_threshold  = cp_cfg.get('flow_threshold', 0.4)
    cellprob_thresh = cp_cfg.get('cellprob_threshold', 0.0)
    normalize       = cp_cfg.get('normalize', True)

    results = []
    for tif in sorted(input_dir.glob('*_mean.tif')):
        print(f'Processing {tif.name} ...')
        img = tifffile.imread(tif).astype(np.float32)

        if use_vcorr:
            # Find matching Vcorr file
            vcorr_name = tif.name.replace('_mean.tif', '_vcorr.tif')
            vcorr_path = vcorr_dir / vcorr_name
            if vcorr_path.exists():
                vcorr = tifffile.imread(vcorr_path).astype(np.float32)
                img = np.stack([img, vcorr], axis=-1)
            else:
                print(f'  WARNING: No Vcorr for {tif.stem}, using zero-padded')
                img = np.stack([img, np.zeros_like(img)], axis=-1)

        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            channels=channels,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_thresh,
            normalize=normalize,
        )

        n_rois = masks.max()
        out_mask = out_dir / tif.name.replace('_mean.tif', '_masks.tif')
        tifffile.imwrite(out_mask, masks.astype(np.uint16))
        results.append({'fov': tif.stem, 'n_rois': n_rois, 'mask': str(out_mask)})
        print(f'  -> {n_rois} ROIs')

    # Write summary
    summary_path = out_dir / 'inference_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['fov', 'n_rois', 'mask'])
        w.writeheader()
        w.writerows(results)
    print(f'\nSummary: {summary_path}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--diameter', type=float, default=0,
                    help='Expected cell diameter in pixels. 0 = auto-estimate.')
    ap.add_argument('--output_dir', default=None,
                    help='Output directory for masks (default: inference/output)')
    ap.add_argument('--vcorr_dir', default=None,
                    help='Directory with *_vcorr.tif files for 2-channel [mean, Vcorr] input')
    ap.add_argument('--config', default=None,
                    help='Path to pipeline YAML config')
    args = ap.parse_args()
    cfg = load_config(args.config)
    diameter = args.diameter or cfg.get('cellpose', {}).get('diameter', 0)
    out_dir = Path(args.output_dir) if args.output_dir else BASE_DIR / 'inference' / 'output'
    run_inference(args.input_dir, diameter, out_dir, vcorr_dir=args.vcorr_dir, cfg=cfg)
