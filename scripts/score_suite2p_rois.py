#!/usr/bin/env python3
"""
ROI G. Biv — Score all Suite2p ROIs with Cellpose neuron probability.

For each FOV:
  - Loads ALL Suite2p ROIs (including iscell=0 candidates)
  - Runs the Cellpose model to get a per-pixel cell probability map
  - Computes mean Cellpose probability for each ROI's pixels
  - Saves a float probability heatmap, a uint16 all-ROI label image, and a CSV

Usage:
  python score_suite2p_rois.py
  python score_suite2p_rois.py --model run011_epoch_0060 --diameter 17
  python score_suite2p_rois.py --no_vcorr --out_dir data/scored_no_vcorr
"""
import argparse, csv
from pathlib import Path
import numpy as np
import tifffile
from cellpose import models

from config import BASE_DIR

S2P_DIR    = BASE_DIR / 'suite2p_workspace' / 'output'
DATA_DIR   = BASE_DIR / 'data' / 'annotated'
OUT_DIR    = BASE_DIR / 'data' / 'scored'
MODEL_PATH = BASE_DIR / 'models' / 'deployed' / 'current_model'
MODELS_DIR = BASE_DIR / 'models' / 'checkpoints' / 'models'


def _get_cellprob(flows, Ly, Lx):
    """Extract per-pixel cell probability from Cellpose flows output.

    Cellpose 2.x places the (H, W) cell probability at flows[2] or flows[1]
    depending on the minor version. We search for the first 2D array with the
    correct spatial shape and apply sigmoid if values are outside [0, 1].
    """
    for idx in (2, 1):
        if idx >= len(flows):
            continue
        arr = np.asarray(flows[idx])
        if arr.ndim == 2 and arr.shape == (Ly, Lx):
            prob = arr.astype(np.float32)
            if prob.min() < 0 or prob.max() > 1:
                prob = 1.0 / (1.0 + np.exp(-prob))
            return prob, idx
    raise RuntimeError(
        f'Cannot find a ({Ly}, {Lx}) 2D probability array in flows. '
        f'Shapes: {[np.asarray(f).shape for f in flows]}'
    )


def score_fovs(s2p_dir, data_dir, out_dir, model_path, diameter, use_vcorr):
    s2p_dir  = Path(s2p_dir)
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = [1, 2] if use_vcorr else [0, 0]
    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))

    fov_dirs = sorted(d for d in s2p_dir.iterdir() if d.is_dir())
    all_rows = []
    n_processed = 0

    for fov_dir in fov_dirs:
        plane0      = fov_dir / 'suite2p' / 'plane0'
        stat_path   = plane0 / 'stat.npy'
        iscell_path = plane0 / 'iscell.npy'
        ops_path    = plane0 / 'ops.npy'
        if not all(p.exists() for p in (stat_path, iscell_path, ops_path)):
            continue

        stem = fov_dir.name
        # Support both _mc_mean.tif and _mean.tif naming conventions
        mean_path = data_dir / f'{stem}_mc_mean.tif'
        if not mean_path.exists():
            mean_path = data_dir / f'{stem}_mean.tif'
        if not mean_path.exists():
            print(f'  WARNING: No mean.tif for {stem}, skipping')
            continue

        print(f'Processing {stem} ...')

        stat   = np.load(stat_path,   allow_pickle=True)
        iscell = np.load(iscell_path)
        ops    = np.load(ops_path,    allow_pickle=True).item()
        Ly, Lx = ops['Ly'], ops['Lx']

        img = tifffile.imread(mean_path).astype(np.float32)
        if use_vcorr:
            vcorr_path = data_dir / f'{stem}_mc_vcorr.tif'
            if vcorr_path.exists():
                vcorr = tifffile.imread(vcorr_path).astype(np.float32)
                img = np.stack([img, vcorr], axis=-1)
            else:
                print(f'  WARNING: No Vcorr for {stem}, using zero 2nd channel')
                img = np.stack([img, np.zeros_like(img)], axis=-1)

        # cellprob_threshold=-6 ensures no pixels are suppressed before
        # the probability map is computed; we read flows directly.
        _, flows, _ = model.eval(
            img, diameter=diameter, channels=channels,
            cellprob_threshold=-6, normalize=True,
        )
        cellprob_map, prob_idx = _get_cellprob(flows, Ly, Lx)
        print(f'  Probability map: flows[{prob_idx}], '
              f'range {cellprob_map.min():.3f}–{cellprob_map.max():.3f}')

        prob_img  = np.zeros((Ly, Lx), dtype=np.float32)
        label_img = np.zeros((Ly, Lx), dtype=np.uint16)

        for i, s in enumerate(stat):
            ypix = s['ypix']
            xpix = s['xpix']
            valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
            ypix, xpix = ypix[valid], xpix[valid]
            mean_prob = float(cellprob_map[ypix, xpix].mean()) if valid.any() else 0.0

            prob_img[ypix, xpix]  = mean_prob
            label_img[ypix, xpix] = i + 1   # 1-indexed; 0 = background

            all_rows.append({
                'fov':                stem,
                'roi_id':             i + 1,
                's2p_iscell':         int(iscell[i, 0]),
                's2p_prob':           float(iscell[i, 1]),
                'cellpose_mean_prob': mean_prob,
            })

        tifffile.imwrite(out_dir / f'{stem}_roi_cellprob.tif',  prob_img)
        tifffile.imwrite(out_dir / f'{stem}_all_s2p_masks.tif', label_img)
        n_processed += 1
        print(f'  {len(stat)} ROIs scored → {stem}_roi_cellprob.tif')

    csv_path = out_dir / 'scored_rois_summary.csv'
    if all_rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
    print(f'\nProcessed {n_processed} FOVs. Summary: {csv_path}')


def main():
    ap = argparse.ArgumentParser(
        description='Score all Suite2p ROIs with Cellpose neuron probability')
    ap.add_argument('--model', default=None,
                    help='Checkpoint name or path (default: models/deployed/current_model)')
    ap.add_argument('--diameter', type=float, default=17,
                    help='Cell diameter for Cellpose rescaling (default: 17)')
    ap.add_argument('--s2p_dir', default=None,
                    help='Suite2p output root (default: suite2p_workspace/output)')
    ap.add_argument('--data_dir', default=None,
                    help='Directory with *_mean.tif + *_vcorr.tif (default: data/annotated)')
    ap.add_argument('--out_dir', default=None,
                    help='Output directory (default: data/scored)')
    ap.add_argument('--use_vcorr', action='store_true', default=True)
    ap.add_argument('--no_vcorr', action='store_true')
    args = ap.parse_args()

    use_vcorr = args.use_vcorr and not args.no_vcorr

    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            candidate = MODELS_DIR / args.model
            model_path = candidate if candidate.exists() else BASE_DIR / args.model
    else:
        model_path = MODEL_PATH

    s2p_dir  = Path(args.s2p_dir)  if args.s2p_dir  else S2P_DIR
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    out_dir  = Path(args.out_dir)  if args.out_dir  else OUT_DIR

    print(f'Model:    {model_path}')
    print(f'Diameter: {args.diameter}')
    print(f'Vcorr 2nd channel: {use_vcorr}')

    score_fovs(s2p_dir, data_dir, out_dir, model_path, args.diameter, use_vcorr)


if __name__ == '__main__':
    main()
