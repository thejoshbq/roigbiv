#!/usr/bin/env python3
"""
ROI G. Biv — Evaluate consensus pipeline output against ground-truth annotations.
Computes Average Precision at multiple IoU thresholds, broken down by tier.

Usage:
  python evaluate_consensus.py --consensus_dir inference/consensus --gt_dir data/masks
  python evaluate_consensus.py --config configs/pipeline.yaml
"""

import argparse, csv
from pathlib import Path
import numpy as np
import tifffile
import yaml
from cellpose import metrics

BASE_DIR = Path.home() / 'Otis-Lab' / 'Projects' / 'roigbiv'

def load_config(config_path=None):
    default = BASE_DIR / 'configs' / 'pipeline.yaml'
    path = Path(config_path) if config_path else default
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def load_tier_masks(consensus_dir, stem, cp_out_dir):
    """Load per-tier masks from consensus CSV and source masks."""
    csv_path = consensus_dir / f'{stem}_consensus.csv'
    cons_mask = consensus_dir / f'{stem}_consensus_masks.tif'
    all_mask  = consensus_dir / f'{stem}_all_tiers_masks.tif'

    if not csv_path.exists() or not cons_mask.exists():
        return None

    # Read tier assignments
    records = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            records.append(row)

    result = {
        'consensus': tifffile.imread(cons_mask),
    }

    if all_mask.exists():
        result['all_tiers'] = tifffile.imread(all_mask)

    # Build GOLD-only mask from consensus mask + records
    cons = result['consensus']
    gold_mask = np.zeros_like(cons)
    gold_id = 0
    for rec in records:
        if rec['tier'] == 'GOLD':
            gold_id += 1
            roi_id = int(rec['roi_id'])
            gold_mask[cons == roi_id] = gold_id
    result['gold_only'] = gold_mask

    # Cellpose-only baseline
    cp_path = cp_out_dir / f'{stem}_mc_masks.tif'
    if not cp_path.exists():
        cp_path = cp_out_dir / f'{stem}_masks.tif'
    if cp_path.exists():
        result['cellpose_only'] = tifffile.imread(cp_path)

    return result


def evaluate(gt_dir, consensus_dir, cp_out_dir, thresholds=(0.5, 0.75)):
    """
    Evaluate all FOVs in consensus_dir against ground truth in gt_dir.

    Returns per-FOV and aggregate AP at each IoU threshold.
    """
    gt_dir = Path(gt_dir)
    consensus_dir = Path(consensus_dir)
    cp_out_dir = Path(cp_out_dir)

    gt_files = sorted(gt_dir.glob('*_masks.tif'))
    if not gt_files:
        print(f'No ground-truth masks found in {gt_dir}')
        return []

    all_results = []
    agg = {k: {t: [] for t in thresholds}
           for k in ('cellpose_only', 'gold_only', 'consensus', 'all_tiers')}

    for gt_path in gt_files:
        stem = gt_path.stem.replace('_masks', '')
        # Consensus files use stems without _mc suffix
        cons_stem = stem.replace('_mc', '')
        tier_masks = load_tier_masks(consensus_dir, cons_stem, cp_out_dir)
        if tier_masks is None:
            continue

        gt_mask = tifffile.imread(gt_path)
        row = {'fov': stem}

        for key in ('cellpose_only', 'gold_only', 'consensus', 'all_tiers'):
            pred = tier_masks.get(key)
            if pred is None:
                for t in thresholds:
                    row[f'{key}_AP@{t}'] = -1.0
                continue

            # Ensure shapes match
            if pred.shape != gt_mask.shape:
                continue

            ap, _, _, _ = metrics.average_precision(
                [gt_mask], [pred], threshold=list(thresholds))

            for ti, t in enumerate(thresholds):
                val = float(ap[0, ti])
                row[f'{key}_AP@{t}'] = round(val, 4)
                agg[key][t].append(val)

        all_results.append(row)

    # Print summary
    print(f'\n{"="*70}')
    print(f'Evaluation Summary — {len(all_results)} FOVs')
    print(f'{"="*70}')
    for key in ('cellpose_only', 'gold_only', 'consensus', 'all_tiers'):
        parts = []
        for t in thresholds:
            vals = agg[key][t]
            if vals:
                parts.append(f'AP@{t}={np.mean(vals):.4f}')
            else:
                parts.append(f'AP@{t}=N/A')
        print(f'  {key:20s}  {", ".join(parts)}')
    print(f'{"="*70}\n')

    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--consensus_dir', default=None,
                    help='Directory with consensus output')
    ap.add_argument('--gt_dir', default=None,
                    help='Directory with ground-truth masks')
    ap.add_argument('--cp_out_dir', default=None,
                    help='Directory with Cellpose-only inference output')
    ap.add_argument('--out_csv', default=None,
                    help='Path to save per-FOV results CSV')
    ap.add_argument('--config', default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get('paths', {})

    consensus_dir = Path(args.consensus_dir) if args.consensus_dir else BASE_DIR / paths.get('consensus_output', 'inference/consensus')
    gt_dir        = Path(args.gt_dir) if args.gt_dir else BASE_DIR / paths.get('masks_dir', 'data/masks')
    cp_out_dir    = Path(args.cp_out_dir) if args.cp_out_dir else BASE_DIR / paths.get('inference_output', 'inference/output')

    results = evaluate(gt_dir, consensus_dir, cp_out_dir)

    if results and args.out_csv:
        out_path = Path(args.out_csv)
        fields = list(results[0].keys())
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
        print(f'Per-FOV results saved to: {out_path}')

if __name__ == '__main__':
    main()
