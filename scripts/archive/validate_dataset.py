#!/usr/bin/env python3
# ~/roigbiv/scripts/validate_dataset.py
import tifffile, numpy as np
from pathlib import Path

annotated = Path('~/Otis-Lab/Projects/roigbiv/data/annotated').expanduser()
masks_dir = Path('~/Otis-Lab/Projects/roigbiv/data/masks').expanduser()

issues = []
pairs = 0
for mask_path in sorted(masks_dir.glob('*_masks.tif')):
    stem = mask_path.stem.replace('_masks', '')
    img_path = annotated / f'{stem}_mean.tif'
    if not img_path.exists():
        issues.append(f'MISSING image: {stem}_mean.tif')
        continue
    img  = tifffile.imread(img_path)
    mask = tifffile.imread(mask_path)
    if img.shape[-2:] != mask.shape[-2:]:
        issues.append(f'SHAPE MISMATCH: {stem} img={img.shape} mask={mask.shape}')
    if mask.max() == 0:
        issues.append(f'EMPTY MASK: {stem}')
    else:
        pairs += 1
        print(f'OK  {stem}: {mask.max()} ROIs, image dtype={img.dtype}')

print(f'\n{pairs} valid pairs found.')
if issues:
    print('\nISSUES:')
    for i in issues: print(' ', i)
else:
    print('All pairs validated successfully.')
