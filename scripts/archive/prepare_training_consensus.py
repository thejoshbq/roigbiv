#!/usr/bin/env python3
"""
ROI G. Biv — Prepare consensus-enhanced training data.
Merges manual ground-truth masks with GOLD-tier consensus discoveries
(cells confirmed by both Suite2p and Cellpose that the annotator missed).

Usage:
  python prepare_training_consensus.py
  python prepare_training_consensus.py --config configs/pipeline.yaml --out_dir data/training
"""

import argparse, csv, os
from pathlib import Path
import numpy as np
import tifffile
import yaml

BASE_DIR = Path.home() / 'Otis-Lab' / 'Projects' / 'roigbiv'

def load_config(config_path=None):
    default = BASE_DIR / 'configs' / 'pipeline.yaml'
    path = Path(config_path) if config_path else default
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def find_uncovered_gold_rois(manual_mask, consensus_mask, consensus_csv_path,
                              overlap_threshold=0.1):
    """
    Find GOLD-tier consensus ROIs that are NOT covered by any manual annotation.

    A consensus ROI is "uncovered" if its max IoU with any manual ROI is below
    overlap_threshold. These are cells that both Suite2p and Cellpose detected
    but the human annotator missed.

    Returns list of (consensus_label, pixels_y, pixels_x) for uncovered GOLD ROIs.
    """
    # Read tier assignments from consensus CSV
    gold_labels = set()
    with open(consensus_csv_path) as f:
        for row in csv.DictReader(f):
            if row['tier'] == 'GOLD' and int(row['cellpose_label']) > 0:
                gold_labels.add(int(row['roi_id']))

    if not gold_labels:
        return []

    manual_labels = np.unique(manual_mask[manual_mask > 0])
    uncovered = []

    for cons_label in sorted(gold_labels):
        cons_pixels = consensus_mask == cons_label
        if cons_pixels.sum() == 0:
            continue

        # Check overlap with every manual ROI
        max_iou = 0.0
        for man_label in manual_labels:
            man_pixels = manual_mask == man_label
            intersection = np.logical_and(cons_pixels, man_pixels).sum()
            if intersection == 0:
                continue
            union = np.logical_or(cons_pixels, man_pixels).sum()
            iou = intersection / union
            max_iou = max(max_iou, iou)
            if max_iou >= overlap_threshold:
                break  # Already covered

        if max_iou < overlap_threshold:
            ys, xs = np.where(cons_pixels)
            uncovered.append((cons_label, ys, xs))

    return uncovered


def merge_masks(manual_mask, uncovered_rois):
    """
    Add uncovered GOLD ROIs to the manual mask with new contiguous labels.

    Returns merged mask and count of added ROIs.
    """
    merged = manual_mask.copy()
    next_label = int(merged.max()) + 1

    for cons_label, ys, xs in uncovered_rois:
        # Skip if any of these pixels are already labeled (avoid overwrite)
        if merged[ys, xs].max() > 0:
            continue
        merged[ys, xs] = next_label
        next_label += 1

    n_added = next_label - int(manual_mask.max()) - 1
    return merged, n_added


def main():
    ap = argparse.ArgumentParser(
        description='Merge manual GT masks with GOLD consensus discoveries')
    ap.add_argument('--config', default=None)
    ap.add_argument('--out_dir', default=None,
                    help='Output directory for merged training data (default: data/training)')
    ap.add_argument('--overlap_threshold', type=float, default=0.1,
                    help='Max IoU to consider a consensus ROI as "uncovered" by manual (default: 0.1)')
    ap.add_argument('--dry_run', action='store_true',
                    help='Report what would be done without writing files')
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get('paths', {})

    masks_dir     = BASE_DIR / paths.get('masks_dir', 'data/masks')
    annotated_dir = BASE_DIR / paths.get('annotated_dir', 'data/annotated')
    consensus_dir = BASE_DIR / paths.get('consensus_output', 'inference/consensus')
    out_dir       = Path(args.out_dir) if args.out_dir else BASE_DIR / 'data' / 'training'

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Find all FOVs with both manual masks and consensus output
    manual_masks = {p.stem.replace('_masks', ''): p
                    for p in sorted(masks_dir.glob('*_masks.tif'))}
    consensus_csvs = {p.stem.replace('_consensus', ''): p
                      for p in sorted(consensus_dir.glob('*_consensus.csv'))}
    consensus_masks = {p.stem.replace('_consensus_masks', ''): p
                       for p in sorted(consensus_dir.glob('*_consensus_masks.tif'))}

    total_fovs = 0
    total_added = 0
    fovs_with_additions = 0

    print(f'Manual masks:    {len(manual_masks)}')
    print(f'Consensus FOVs:  {len(consensus_csvs)}')
    print(f'Output:          {out_dir}')
    print(f'Overlap thresh:  {args.overlap_threshold}')
    print()

    for stem, manual_path in sorted(manual_masks.items()):
        total_fovs += 1
        manual_mask = tifffile.imread(manual_path)
        n_manual = int(manual_mask.max())

        # Check if consensus data exists for this FOV
        # Consensus files use stems without _mc suffix
        cons_stem = stem.replace('_mc', '')
        cons_csv  = consensus_csvs.get(cons_stem)
        cons_mask_path = consensus_masks.get(cons_stem)

        if cons_csv and cons_mask_path:
            cons_mask = tifffile.imread(cons_mask_path)

            # Ensure shapes match
            if cons_mask.shape != manual_mask.shape:
                print(f'  WARNING: {stem} shape mismatch (manual {manual_mask.shape} vs consensus {cons_mask.shape}), using manual only')
                merged, n_added = manual_mask, 0
            else:
                uncovered = find_uncovered_gold_rois(
                    manual_mask, cons_mask, cons_csv,
                    overlap_threshold=args.overlap_threshold)
                merged, n_added = merge_masks(manual_mask, uncovered)
        else:
            # No consensus data — use manual mask as-is
            merged, n_added = manual_mask, 0

        total_added += n_added
        if n_added > 0:
            fovs_with_additions += 1

        status = f'+{n_added} GOLD' if n_added > 0 else 'unchanged'
        print(f'  {stem}: {n_manual} manual ROIs {status} → {int(merged.max())} total')

        if not args.dry_run:
            # Save merged mask
            tifffile.imwrite(out_dir / f'{stem}_masks.tif', merged.astype(np.uint16))

            # Symlink projections
            for suffix in ('_mean.tif', '_max.tif'):
                src = annotated_dir / f'{stem}{suffix}'
                dst = out_dir / f'{stem}{suffix}'
                if src.exists() and not dst.exists():
                    os.symlink(src, dst)

    print(f'\n{"="*50}')
    print(f'Summary:')
    print(f'  FOVs processed:     {total_fovs}')
    print(f'  FOVs with additions:{fovs_with_additions}')
    print(f'  Total GOLD ROIs added: {total_added}')
    if not args.dry_run:
        print(f'  Output directory:   {out_dir}')
    else:
        print(f'  (dry run — no files written)')


if __name__ == '__main__':
    main()
