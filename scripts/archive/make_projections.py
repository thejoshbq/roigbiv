#!/usr/bin/env python3
"""
Generate mean and max projections from raw two-photon .tif stacks.
Outputs paired files: FOVNAME_mean.tif and FOVNAME_max.tif
"""
import sys
import tifffile
import numpy as np
from pathlib import Path
import argparse

def make_projections(raw_dir, out_dir, percentile_clip=(0.5, 99.5)):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(raw_dir.glob('*.tif'))
    skipped = []

    for tif in tifs:
        try:
            stack = tifffile.imread(tif).astype(np.float32)
            # Handle (T, Y, X) or (Z, Y, X) stacks
            if stack.ndim == 2:
                stack = stack[np.newaxis, ...]

            mean_proj = stack.mean(axis=0)
            max_proj  = stack.max(axis=0)

            # Percentile clip and rescale to uint16
            for proj, suffix in [(mean_proj, 'mean'), (max_proj, 'max')]:
                lo, hi = np.percentile(proj, percentile_clip)
                proj = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
                proj = (proj * 65535).astype(np.uint16)
                out_path = out_dir / f'{tif.stem}_{suffix}.tif'
                tifffile.imwrite(out_path, proj)
                print(f'Wrote {out_path}')
        except Exception as exc:
            print(f'WARNING: skipping {tif.name}: {exc}', file=sys.stderr)
            skipped.append(tif.name)

    total = len(tifs)
    processed = total - len(skipped)
    print(f'Processed {processed}/{total} files, {len(skipped)} skipped')
    if skipped:
        sys.exit(1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('raw_dir')
    ap.add_argument('out_dir')
    args = ap.parse_args()
    make_projections(args.raw_dir, args.out_dir)

