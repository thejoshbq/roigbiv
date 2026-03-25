#!/usr/bin/env python3
"""
Extract mean fluorescence traces from raw stacks using ROI masks.
Output: CSV and npy file per FOV, rows=frames, cols=ROI IDs.
Usage:
  python extract_traces.py --raw_dir /path/to/raw --mask_dir /path/to/masks
  python extract_traces.py --config configs/pipeline.yaml --raw_dir /path/to/raw
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile
import pandas as pd

from config import BASE_DIR, load_config

OUT_DIR  = BASE_DIR / 'inference' / 'traces'

def extract_traces(raw_dir, mask_dir):
    raw_dir  = Path(raw_dir)
    mask_dir = Path(mask_dir)

    for mask_path in sorted(mask_dir.glob('*_masks.tif')):
        stem = mask_path.stem.replace('_masks', '').replace('_consensus', '')
        # Try to find matching raw stack
        raw_candidates = list(raw_dir.glob(f'{stem}*.tif'))
        if not raw_candidates:
            print(f'WARNING: No raw stack for {stem}')
            continue
        raw_path = raw_candidates[0]

        print(f'Extracting traces: {stem}')
        stack = tifffile.imread(raw_path).astype(np.float32)
        mask  = tifffile.imread(mask_path)

        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]

        n_frames = stack.shape[0]
        roi_ids  = np.unique(mask[mask > 0])
        traces   = np.zeros((n_frames, len(roi_ids)), dtype=np.float32)

        for i, roi_id in enumerate(roi_ids):
            roi_pixels = mask == roi_id
            for t in range(n_frames):
                traces[t, i] = stack[t][roi_pixels].mean()

        # Save as CSV and npy
        df = pd.DataFrame(traces, columns=[f'ROI_{r}' for r in roi_ids])
        df.to_csv(OUT_DIR / f'{stem}_traces.csv', index=False)
        np.save(OUT_DIR / f'{stem}_traces.npy', traces)
        print(f'  {len(roi_ids)} ROIs x {n_frames} frames')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--mask_dir',
                    default=str(BASE_DIR / 'inference' / 'output'),
                    help='Directory with *_masks.tif or *_consensus_masks.tif')
    ap.add_argument('--config', default=None,
                    help='Path to pipeline YAML config')
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg.get('paths', {})
    mask_dir = args.mask_dir
    if mask_dir == str(BASE_DIR / 'inference' / 'output') and 'consensus_output' in paths_cfg:
        consensus_dir = BASE_DIR / paths_cfg['consensus_output']
        if consensus_dir.exists() and any(consensus_dir.glob('*_masks.tif')):
            mask_dir = str(consensus_dir)
    extract_traces(args.raw_dir, mask_dir)
