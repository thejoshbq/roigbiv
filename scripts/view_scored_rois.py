#!/usr/bin/env python3
"""
ROI G. Biv — Napari viewer for Suite2p ROIs scored with Cellpose probability.

Opens three layers for a single FOV:
  1. mean.tif           — grayscale background image
  2. *_roi_cellprob.tif — hot colormap heatmap (bright = high Cellpose confidence)
  3. *_all_s2p_masks.tif — ALL Suite2p ROIs, each colored by its Cellpose probability

Hover over any mask outline to see its exact probability in the napari status bar.

Usage:
  python view_scored_rois.py --stem T1_221209_..._PRE-002
  python view_scored_rois.py   # auto-picks first available FOV, lists all options
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from matplotlib import colormaps
from matplotlib.colors import Normalize

from config import BASE_DIR

SCORED_DIR = BASE_DIR / 'data' / 'scored'
DATA_DIR   = BASE_DIR / 'data' / 'annotated'


def _build_color_dict(per_roi_probs):
    """Map label IDs → RGBA colors scaled by Cellpose probability.

    Uses a green-yellow-red colormap: green = high confidence neuron,
    red = low confidence. Background (label 0) is fully transparent.
    """
    cmap = colormaps['RdYlGn']
    norm = Normalize(vmin=0, vmax=1)
    color_dict = {}
    for label_id, prob in enumerate(per_roi_probs):
        if label_id == 0:
            color_dict[0] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            rgba = np.array(cmap(norm(prob)), dtype=np.float32)
            color_dict[label_id] = rgba
    return color_dict


def _load_per_roi_probs(scored_dir, stem, n_labels):
    """Load per-ROI probabilities from CSV. Returns array of length n_labels+1."""
    csv_path = Path(scored_dir) / 'scored_rois_summary.csv'
    probs = np.zeros(n_labels + 1, dtype=np.float32)  # index 0 = background
    if not csv_path.exists():
        print('WARNING: scored_rois_summary.csv not found — masks shown with uniform color.')
        return probs
    df = pd.read_csv(csv_path)
    fov_df = df[df['fov'] == stem].sort_values('roi_id')
    if fov_df.empty:
        print(f'WARNING: No rows for stem "{stem}" in CSV.')
        return probs
    for _, row in fov_df.iterrows():
        idx = int(row['roi_id'])
        if idx <= n_labels:
            probs[idx] = float(row['cellpose_mean_prob'])
    return probs


def view_fov(stem, scored_dir, data_dir):
    import napari
    from napari.utils.colormaps import direct_colormap

    scored_dir = Path(scored_dir)
    data_dir   = Path(data_dir)

    mean_path     = data_dir / f'{stem}_mc_mean.tif'
    if not mean_path.exists():
        mean_path = data_dir / f'{stem}_mean.tif'
    cellprob_path = scored_dir / f'{stem}_roi_cellprob.tif'
    masks_path    = scored_dir / f'{stem}_all_s2p_masks.tif'

    for p in (mean_path, cellprob_path, masks_path):
        if not p.exists():
            raise FileNotFoundError(
                f'Missing: {p}\n'
                'Run score_suite2p_rois.py (activity-only) or '
                'build_union_rois.py (union pipeline) first to generate scored outputs.'
            )

    mean_img     = tifffile.imread(mean_path).astype(np.float32)
    cellprob_map = tifffile.imread(cellprob_path).astype(np.float32)
    label_img    = tifffile.imread(masks_path)

    n_labels = int(label_img.max())
    per_roi_probs = _load_per_roi_probs(scored_dir, stem, n_labels)
    color_dict = _build_color_dict(per_roi_probs)

    print(f'FOV: {stem}')
    print(f'  {n_labels} Suite2p ROIs loaded')
    print(f'  Cellpose prob range: {per_roi_probs[1:].min():.3f}–{per_roi_probs[1:].max():.3f}')
    print(f'  ROIs with prob > 0.5: {(per_roi_probs[1:] > 0.5).sum()}')

    viewer = napari.Viewer(title=f'ROI Scores — {stem}')
    viewer.add_image(mean_img, name='mean', colormap='gray')
    viewer.add_image(
        cellprob_map, name='cellpose_prob_heatmap',
        colormap='hot', opacity=0.4, blending='additive',
    )
    viewer.add_labels(
        label_img, name='all_s2p_rois',
        colormap=direct_colormap(color_dict),
    )

    print('\nNapari viewer open.')
    print('  Green masks = high Cellpose neuron confidence')
    print('  Red masks   = low confidence (Suite2p detected, Cellpose uncertain)')
    print('  Hover a mask to see its label ID, then cross-reference scored_rois_summary.csv')
    napari.run()


def main():
    ap = argparse.ArgumentParser(
        description='Napari viewer: Suite2p ROIs colored by Cellpose probability')
    ap.add_argument('--stem', default=None,
                    help='FOV stem name (e.g. T1_221209_..._PRE-002). '
                         'If omitted, lists available stems and opens the first.')
    ap.add_argument('--scored_dir', default=None,
                    help='Directory with scored TIFs + CSV (default: data/scored)')
    ap.add_argument('--data_dir', default=None,
                    help='Directory with *_mean.tif files (default: data/annotated)')
    args = ap.parse_args()

    scored_dir = Path(args.scored_dir) if args.scored_dir else SCORED_DIR
    data_dir   = Path(args.data_dir)   if args.data_dir   else DATA_DIR

    if args.stem:
        stem = args.stem
    else:
        cellprob_tifs = sorted(scored_dir.glob('*_roi_cellprob.tif'))
        if not cellprob_tifs:
            raise FileNotFoundError(
                f'No scored TIFs found in {scored_dir}. '
                'Run score_suite2p_rois.py first.'
            )
        stems = [p.name.replace('_roi_cellprob.tif', '') for p in cellprob_tifs]
        print(f'Available FOVs ({len(stems)} total):')
        for s in stems:
            print(f'  {s}')
        stem = stems[0]
        print(f'\nOpening: {stem}')

    view_fov(stem, scored_dir, data_dir)


if __name__ == '__main__':
    main()
