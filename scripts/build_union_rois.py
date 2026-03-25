#!/usr/bin/env python3
"""
ROI G. Biv — Build union ROI set from activity-based + anatomy-based Suite2p runs,
then score each union ROI with Cellpose neuron probability.

Two Suite2p detection passes are merged:
  - Activity pass (anatomical_only=0): ROIs with spatiotemporal fluorescence correlation
  - Anatomy pass (anatomical_only=1):  ROIs detected from the mean image alone

IoU matching (Hungarian algorithm) classifies each union ROI:
  GOLD   = found by both passes (IoU > threshold)     — highest confidence
  SILVER = anatomy-only (morphologically neuron-shaped, silent in this recording)
  BRONZE = activity-only (active but not anatomically prominent)

Outputs (compatible with view_scored_rois.py):
  {stem}_all_s2p_masks.tif   — uint16 union mask (all tiers)
  {stem}_roi_cellprob.tif    — float32 Cellpose probability heatmap
  scored_rois_summary.csv    — per-ROI: tier, iou_score, s2p probs, cellpose prob

Usage:
  python build_union_rois.py
  python build_union_rois.py --diameter 17 --out_dir data/union_rois
  python build_union_rois.py --no_vcorr --iou_threshold 0.2
"""
import csv
from pathlib import Path
import numpy as np
import tifffile
from cellpose import models

from match_rois import match_and_tier, build_consensus_mask

from config import BASE_DIR

ACTIVITY_DIR = BASE_DIR / 'suite2p_workspace' / 'output'
ANATOMY_DIR  = BASE_DIR / 'suite2p_workspace' / 'output_anatomy'
DATA_DIR     = BASE_DIR / 'data' / 'annotated'
OUT_DIR      = BASE_DIR / 'data' / 'union_rois'
MODEL_PATH   = BASE_DIR / 'models' / 'deployed' / 'current_model'
MODELS_DIR   = BASE_DIR / 'models' / 'checkpoints' / 'models'


def _get_cellprob(flows, Ly, Lx):
    """Extract per-pixel cell probability from Cellpose flows output.
    Handles version differences: tries flows[2] then flows[1].
    Applies sigmoid if values fall outside [0, 1] (raw logits).
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


def _stat_to_mask(stat, Ly, Lx):
    """Convert Suite2p stat.npy (all ROIs, no iscell filter) to uint16 label image."""
    mask = np.zeros((Ly, Lx), dtype=np.uint16)
    for i, s in enumerate(stat):
        ypix = s['ypix']
        xpix = s['xpix']
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        mask[ypix[valid], xpix[valid]] = i + 1  # 1-indexed
    return mask


def build_union(activity_dir, anatomy_dir, data_dir, out_dir,
                model_path, diameter, use_vcorr, iou_threshold):
    activity_dir = Path(activity_dir)
    anatomy_dir  = Path(anatomy_dir)
    data_dir     = Path(data_dir)
    out_dir      = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = [1, 2] if use_vcorr else [0, 0]
    model = models.CellposeModel(gpu=True, pretrained_model=str(model_path))

    # Find FOVs present in both runs
    activity_fovs = {d.name for d in activity_dir.iterdir() if d.is_dir()}
    anatomy_fovs  = {d.name for d in anatomy_dir.iterdir()  if d.is_dir()}
    common_fovs   = sorted(activity_fovs & anatomy_fovs)

    if not common_fovs:
        print(f'WARNING: No FOVs found in both {activity_dir} and {anatomy_dir}')
        return

    all_rows = []
    n_processed = 0

    for stem in common_fovs:
        act_plane = activity_dir / stem / 'suite2p' / 'plane0'
        ana_plane = anatomy_dir  / stem / 'suite2p' / 'plane0'

        required = ('stat.npy', 'iscell.npy', 'ops.npy')
        if not all((act_plane / f).exists() for f in required):
            continue
        if not all((ana_plane / f).exists() for f in required):
            continue

        mean_path = data_dir / f'{stem}_mc_mean.tif'
        if not mean_path.exists():
            mean_path = data_dir / f'{stem}_mean.tif'
        if not mean_path.exists():
            print(f'  WARNING: No mean.tif for {stem}, skipping')
            continue

        print(f'Processing {stem} ...')

        # Load Suite2p outputs
        act_stat   = np.load(act_plane / 'stat.npy',   allow_pickle=True)
        act_iscell = np.load(act_plane / 'iscell.npy')
        act_ops    = np.load(act_plane / 'ops.npy',    allow_pickle=True).item()
        ana_stat   = np.load(ana_plane / 'stat.npy',   allow_pickle=True)
        ana_iscell = np.load(ana_plane / 'iscell.npy')
        ana_ops    = np.load(ana_plane / 'ops.npy',    allow_pickle=True).item()

        Ly, Lx = act_ops['Ly'], act_ops['Lx']
        if (ana_ops['Ly'], ana_ops['Lx']) != (Ly, Lx):
            print(f'  WARNING: Dimension mismatch between runs for {stem}, skipping')
            continue

        # Convert stat.npy → uint16 masks (all ROIs, no filter)
        act_mask = _stat_to_mask(act_stat, Ly, Lx)
        ana_mask = _stat_to_mask(ana_stat, Ly, Lx)

        # IoU matching: anatomy plays 'cp_mask' role, activity plays 's2p_mask' role
        records = match_and_tier(ana_mask, act_mask,
                                 iou_threshold=iou_threshold,
                                 s2p_iscell=act_iscell)

        n_gold   = sum(1 for r in records if r['tier'] == 'GOLD')
        n_silver = sum(1 for r in records if r['tier'] == 'SILVER')
        n_bronze = sum(1 for r in records if r['tier'] == 'BRONZE')
        print(f'  Activity: {len(act_stat)} ROIs  Anatomy: {len(ana_stat)} ROIs  '
              f'→ GOLD={n_gold} SILVER={n_silver} BRONZE={n_bronze}')

        # Union mask: all tiers combined
        union_mask = build_consensus_mask(ana_mask, act_mask, records,
                                          tiers=('GOLD', 'SILVER', 'BRONZE'))

        # Run Cellpose → cell probability map
        img = tifffile.imread(mean_path).astype(np.float32)
        if use_vcorr:
            vcorr_path = data_dir / f'{stem}_mc_vcorr.tif'
            if vcorr_path.exists():
                vcorr = tifffile.imread(vcorr_path).astype(np.float32)
                img = np.stack([img, vcorr], axis=-1)
            else:
                print(f'  WARNING: No Vcorr for {stem}, using zero 2nd channel')
                img = np.stack([img, np.zeros_like(img)], axis=-1)

        _, flows, _ = model.eval(
            img, diameter=diameter, channels=channels,
            cellprob_threshold=-6, normalize=True,
        )
        cellprob_map, prob_idx = _get_cellprob(flows, Ly, Lx)
        print(f'  Cellpose prob map: flows[{prob_idx}], '
              f'range {cellprob_map.min():.3f}–{cellprob_map.max():.3f}')

        # Build per-record lookup for anatomy iscell probs
        # records use anatomy labels (cellpose_label) and activity labels (s2p_label)
        ana_prob_by_label  = {i + 1: float(ana_iscell[i, 1])
                              for i in range(len(ana_iscell))}
        act_prob_by_label  = {i + 1: float(act_iscell[i, 1])
                              for i in range(len(act_iscell))}
        act_cell_by_label  = {i + 1: int(act_iscell[i, 0])
                              for i in range(len(act_iscell))}

        # Score each union ROI by its pixels in the union mask
        prob_img = np.zeros((Ly, Lx), dtype=np.float32)

        for new_id, rec in enumerate(records, start=1):
            ypix, xpix = np.where(union_mask == new_id)
            mean_prob = float(cellprob_map[ypix, xpix].mean()) if len(ypix) > 0 else 0.0
            prob_img[ypix, xpix] = mean_prob

            ana_lbl = rec['cellpose_label']  # anatomy label (or -1 if BRONZE)
            act_lbl = rec['s2p_label']       # activity label (or -1 if SILVER)

            all_rows.append({
                'fov':                stem,
                'roi_id':             new_id,
                'tier':               rec['tier'],
                'iou_score':          rec['iou_score'],
                'activity_iscell':    act_cell_by_label.get(act_lbl, -1),
                'activity_s2p_prob':  act_prob_by_label.get(act_lbl, -1.0),
                'anatomy_s2p_prob':   ana_prob_by_label.get(ana_lbl, -1.0),
                'cellpose_mean_prob': round(mean_prob, 5),
            })

        tifffile.imwrite(out_dir / f'{stem}_all_s2p_masks.tif', union_mask)
        tifffile.imwrite(out_dir / f'{stem}_roi_cellprob.tif',  prob_img)
        n_processed += 1

    csv_path = out_dir / 'scored_rois_summary.csv'
    if all_rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
    print(f'\nProcessed {n_processed} FOVs. Summary: {csv_path}')


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Build union ROI set (activity + anatomy) and score with Cellpose')
    ap.add_argument('--activity_dir', default=None,
                    help='Suite2p output from activity pass '
                         '(default: suite2p_workspace/output)')
    ap.add_argument('--anatomy_dir', default=None,
                    help='Suite2p output from anatomy pass '
                         '(default: suite2p_workspace/output_anatomy)')
    ap.add_argument('--model', default=None,
                    help='Checkpoint name or path (default: models/deployed/current_model)')
    ap.add_argument('--diameter', type=float, default=17,
                    help='Cell diameter for Cellpose rescaling (default: 17)')
    ap.add_argument('--data_dir', default=None,
                    help='Directory with *_mean.tif + *_vcorr.tif (default: data/annotated)')
    ap.add_argument('--out_dir', default=None,
                    help='Output directory (default: data/union_rois)')
    ap.add_argument('--iou_threshold', type=float, default=0.3,
                    help='IoU threshold for GOLD matching (default: 0.3)')
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

    activity_dir = Path(args.activity_dir) if args.activity_dir else ACTIVITY_DIR
    anatomy_dir  = Path(args.anatomy_dir)  if args.anatomy_dir  else ANATOMY_DIR
    data_dir     = Path(args.data_dir)     if args.data_dir     else DATA_DIR
    out_dir      = Path(args.out_dir)      if args.out_dir      else OUT_DIR

    print(f'Activity dir: {activity_dir}')
    print(f'Anatomy dir:  {anatomy_dir}')
    print(f'Model:        {model_path}')
    print(f'Diameter:     {args.diameter}')
    print(f'IoU threshold: {args.iou_threshold}')
    print(f'Vcorr 2nd channel: {use_vcorr}')

    build_union(activity_dir, anatomy_dir, data_dir, out_dir,
                model_path, args.diameter, use_vcorr, args.iou_threshold)


if __name__ == '__main__':
    main()
