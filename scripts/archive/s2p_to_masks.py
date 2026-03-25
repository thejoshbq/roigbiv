#!/usr/bin/env python3
"""
ROI G. Biv — Convert Suite2p stat.npy footprints to uint16 labeled masks.
Produces masks in the same format as Cellpose output for consensus matching.

Usage:
  python s2p_to_masks.py --s2p_dir suite2p_workspace/output/suite2p/plane0
  python s2p_to_masks.py --s2p_dir ... --ref_mask data/masks/FOV1_masks.tif
  python s2p_to_masks.py --config configs/pipeline.yaml --s2p_dir ...
"""

import argparse
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


def s2p_stat_to_mask(stat_path, iscell_path, ops_path, min_prob=0.5,
                     ref_shape=None):
    """
    Convert Suite2p stat.npy to a uint16 labeled mask image.

    Parameters
    ----------
    stat_path : Path
        Path to stat.npy (list of ROI dicts with ypix, xpix keys).
    iscell_path : Path
        Path to iscell.npy (n_rois, 2) — col0=binary, col1=probability.
    ops_path : Path
        Path to ops.npy (dict with Ly, Lx fields for image dimensions).
    min_prob : float
        Minimum iscell probability to include an ROI.
    ref_shape : tuple or None
        If provided, center-crop or pad the output to match this (H, W).
        Use this to align with Cellpose masks of a different spatial extent.

    Returns
    -------
    mask : np.ndarray
        uint16 labeled mask of shape (Ly, Lx) or ref_shape.
    n_rois : int
        Number of ROIs included in the mask.
    """
    stat   = np.load(stat_path, allow_pickle=True)
    iscell = np.load(iscell_path)
    ops    = np.load(ops_path, allow_pickle=True).item()

    Ly, Lx = ops['Ly'], ops['Lx']
    mask = np.zeros((Ly, Lx), dtype=np.uint16)

    label = 0
    for i, s in enumerate(stat):
        if iscell[i, 0] < 1 or iscell[i, 1] < min_prob:
            continue
        label += 1
        ypix = s['ypix']
        xpix = s['xpix']
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        mask[ypix[valid], xpix[valid]] = label

    # Align to reference shape if provided
    if ref_shape is not None and (Ly, Lx) != tuple(ref_shape):
        mask = _align_to_shape(mask, ref_shape)

    return mask, label


def _align_to_shape(mask, target_shape):
    """Center-crop or center-pad mask to match target_shape."""
    src_h, src_w = mask.shape
    tgt_h, tgt_w = target_shape
    out = np.zeros((tgt_h, tgt_w), dtype=mask.dtype)

    # Compute overlap region (center-aligned)
    y_off_src = max(0, (src_h - tgt_h) // 2)
    x_off_src = max(0, (src_w - tgt_w) // 2)
    y_off_tgt = max(0, (tgt_h - src_h) // 2)
    x_off_tgt = max(0, (tgt_w - src_w) // 2)

    copy_h = min(src_h - y_off_src, tgt_h - y_off_tgt)
    copy_w = min(src_w - x_off_src, tgt_w - x_off_tgt)

    out[y_off_tgt:y_off_tgt + copy_h,
        x_off_tgt:x_off_tgt + copy_w] = mask[y_off_src:y_off_src + copy_h,
                                              x_off_src:x_off_src + copy_w]

    pct_diff = abs(src_h * src_w - tgt_h * tgt_w) / max(src_h * src_w, 1) * 100
    if pct_diff > 5:
        print(f'  WARNING: Suite2p dims ({src_h}x{src_w}) differ from reference '
              f'({tgt_h}x{tgt_w}) by {pct_diff:.1f}%')

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s2p_dir', required=True,
                    help='Suite2p plane output dir (contains stat.npy, iscell.npy, ops.npy)')
    ap.add_argument('--ref_mask', default=None,
                    help='Reference Cellpose mask to match spatial dimensions')
    ap.add_argument('--out_dir', default=None,
                    help='Output directory (default: same as s2p_dir)')
    ap.add_argument('--min_prob', type=float, default=None,
                    help='Minimum iscell probability (overrides config)')
    ap.add_argument('--config', default=None,
                    help='Path to pipeline YAML config')
    args = ap.parse_args()

    cfg = load_config(args.config)
    cons_cfg = cfg.get('consensus', {})
    min_prob = args.min_prob or cons_cfg.get('s2p_min_iscell_prob', 0.5)

    s2p_dir = Path(args.s2p_dir)
    stat_path   = s2p_dir / 'stat.npy'
    iscell_path = s2p_dir / 'iscell.npy'
    ops_path    = s2p_dir / 'ops.npy'

    for p in (stat_path, iscell_path, ops_path):
        if not p.exists():
            raise FileNotFoundError(f'Missing Suite2p output: {p}')

    ref_shape = None
    if args.ref_mask:
        ref = tifffile.imread(args.ref_mask)
        ref_shape = ref.shape[:2]

    print(f'Converting Suite2p stat → mask (min_prob={min_prob})')
    mask, n_rois = s2p_stat_to_mask(stat_path, iscell_path, ops_path,
                                     min_prob=min_prob, ref_shape=ref_shape)

    out_dir = Path(args.out_dir) if args.out_dir else s2p_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 's2p_masks.tif'
    tifffile.imwrite(out_path, mask)
    print(f'  {n_rois} ROIs → {out_path}')

if __name__ == '__main__':
    main()
