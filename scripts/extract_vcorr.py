#!/usr/bin/env python3
"""
ROI G. Biv — Extract Vcorr (local temporal correlation) maps from Suite2p ops.npy.
Vcorr encodes pixel-level temporal synchrony as a 2D spatial map, serving as the
temporal input channel for Cellpose training and inference.

Usage:
  python extract_vcorr.py
  python extract_vcorr.py --s2p_dir suite2p_workspace/output --out_dir data/annotated
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile

from config import BASE_DIR, load_config


def extract_vcorr(s2p_dir, out_dir):
    s2p_dir = Path(s2p_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fov_dirs = sorted(d for d in s2p_dir.iterdir() if d.is_dir())
    n_extracted = 0

    for fov_dir in fov_dirs:
        ops_path = fov_dir / 'suite2p' / 'plane0' / 'ops.npy'
        if not ops_path.exists():
            continue

        stem = fov_dir.name  # e.g. T1_221209_..._PRE-002
        ops = np.load(ops_path, allow_pickle=True).item()

        if 'Vcorr' not in ops:
            print(f'  WARNING: No Vcorr in {stem}, skipping')
            continue

        vcorr = ops['Vcorr'].astype(np.float32)
        # Save with _mc suffix to match annotated file naming convention
        out_path = out_dir / f'{stem}_mc_vcorr.tif'
        tifffile.imwrite(out_path, vcorr)
        n_extracted += 1
        print(f'  {stem} → {out_path.name}  (range {vcorr.min():.1f}–{vcorr.max():.1f})')

    print(f'\nExtracted {n_extracted} Vcorr maps to {out_dir}')
    return n_extracted


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Extract Vcorr temporal correlation maps from Suite2p output')
    ap.add_argument('--s2p_dir', default=None,
                    help='Suite2p output directory (default: suite2p_workspace/output)')
    ap.add_argument('--out_dir', default=None,
                    help='Output directory for Vcorr TIFs (default: data/annotated)')
    ap.add_argument('--config', default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get('paths', {})

    s2p_dir = Path(args.s2p_dir) if args.s2p_dir else BASE_DIR / paths.get('s2p_output', 'suite2p_workspace/output')
    out_dir = Path(args.out_dir) if args.out_dir else BASE_DIR / paths.get('annotated_dir', 'data/annotated')

    extract_vcorr(s2p_dir, out_dir)
