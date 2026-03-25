#!/usr/bin/env python3
"""
Ingest Cellpose GUI corrections (*_seg.npy) into the training dataset.
Run after each review session to update masks for retraining.
"""
import numpy as np
import tifffile
from pathlib import Path
import shutil

GUI_DIR   = Path('~/Otis-Lab/Projects/roigbiv/data/annotated').expanduser()
MASKS_DIR = Path('~/Otis-Lab/Projects/roigbiv/data/masks').expanduser()

updated = 0
for seg_file in sorted(GUI_DIR.glob('*_seg.npy')):
    data = np.load(seg_file, allow_pickle=True).item()
    masks = data.get('masks', None)
    if masks is None or masks.max() == 0:
        print(f'WARNING: No masks in {seg_file.name}, skipping')
        continue
    stem = seg_file.stem.replace('_mean_seg', '').replace('_seg', '')
    out_path = MASKS_DIR / f'{stem}_masks.tif'
    tifffile.imwrite(out_path, masks.astype(np.uint16))
    print(f'Updated: {stem} ({masks.max()} ROIs) -> {out_path.name}')
    updated += 1

print(f'\n{updated} mask files updated.')
