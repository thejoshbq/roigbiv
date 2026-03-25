#!/usr/bin/env python3
"""
ROI G. Biv — Consensus ROI matching between Suite2p and Cellpose masks.
Computes pairwise IoU, runs Hungarian matching, and classifies ROIs into
GOLD (both agree), SILVER (Cellpose-only), and BRONZE (Suite2p-only) tiers.

Usage:
  python match_rois.py --cp_mask inference/output/FOV1_masks.tif \
                       --s2p_mask suite2p_workspace/output/.../s2p_masks.tif
  python match_rois.py --config configs/pipeline.yaml --cp_mask ... --s2p_mask ...
"""

import argparse, csv
from pathlib import Path
import numpy as np
import tifffile
import yaml
from scipy.optimize import linear_sum_assignment

BASE_DIR = Path.home() / 'Otis-Lab' / 'Projects' / 'roigbiv'

def load_config(config_path=None):
    default = BASE_DIR / 'configs' / 'pipeline.yaml'
    path = Path(config_path) if config_path else default
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def compute_iou_matrix(mask_a, mask_b):
    """
    Compute pairwise IoU between all labeled ROIs in two masks.

    Returns
    -------
    iou : np.ndarray of shape (n_a, n_b)
    labels_a : np.ndarray of unique ROI labels in mask_a
    labels_b : np.ndarray of unique ROI labels in mask_b
    """
    labels_a = np.unique(mask_a[mask_a > 0])
    labels_b = np.unique(mask_b[mask_b > 0])

    if len(labels_a) == 0 or len(labels_b) == 0:
        return np.zeros((len(labels_a), len(labels_b)), dtype=np.float32), labels_a, labels_b

    iou = np.zeros((len(labels_a), len(labels_b)), dtype=np.float32)

    for i, la in enumerate(labels_a):
        pixels_a = mask_a == la
        for j, lb in enumerate(labels_b):
            pixels_b = mask_b == lb
            intersection = np.logical_and(pixels_a, pixels_b).sum()
            if intersection == 0:
                continue
            union = np.logical_or(pixels_a, pixels_b).sum()
            iou[i, j] = intersection / union

    return iou, labels_a, labels_b


def match_and_tier(cp_mask, s2p_mask, iou_threshold=0.3, s2p_iscell=None):
    """
    Match Cellpose and Suite2p ROIs and assign confidence tiers.

    Parameters
    ----------
    cp_mask : np.ndarray
        uint16 Cellpose labeled mask.
    s2p_mask : np.ndarray
        uint16 Suite2p labeled mask (from s2p_to_masks.py).
    iou_threshold : float
        Minimum IoU to consider a match as GOLD.
    s2p_iscell : np.ndarray or None
        Suite2p iscell array (n_rois, 2) for probability metadata.

    Returns
    -------
    roi_records : list of dicts
        Per-ROI metadata with tier classification.
    """
    iou_matrix, cp_labels, s2p_labels = compute_iou_matrix(cp_mask, s2p_mask)
    records = []

    # Hungarian matching (maximize IoU → minimize negative IoU)
    matched_cp = set()
    matched_s2p = set()
    gold_pairs = []

    if iou_matrix.size > 0:
        cost = -iou_matrix
        row_idx, col_idx = linear_sum_assignment(cost)

        for r, c in zip(row_idx, col_idx):
            score = iou_matrix[r, c]
            if score >= iou_threshold:
                gold_pairs.append((r, c, score))
                matched_cp.add(r)
                matched_s2p.add(c)

    # GOLD: both methods agree
    roi_id = 0
    for r, c, score in gold_pairs:
        roi_id += 1
        cp_lbl  = int(cp_labels[r])
        s2p_lbl = int(s2p_labels[c])
        cy, cx  = _centroid(cp_mask, cp_lbl)
        area    = int((cp_mask == cp_lbl).sum())
        s2p_prob = float(s2p_iscell[s2p_lbl - 1, 1]) if s2p_iscell is not None and s2p_lbl - 1 < len(s2p_iscell) else -1.0
        records.append({
            'roi_id': roi_id, 'tier': 'GOLD', 'iou_score': round(score, 4),
            'cellpose_label': cp_lbl, 's2p_label': s2p_lbl,
            'centroid_y': cy, 'centroid_x': cx, 'area_px': area,
            's2p_iscell_prob': round(s2p_prob, 4),
        })

    # SILVER: Cellpose-only
    for i, cp_lbl in enumerate(cp_labels):
        if i in matched_cp:
            continue
        roi_id += 1
        cp_lbl = int(cp_lbl)
        cy, cx = _centroid(cp_mask, cp_lbl)
        area   = int((cp_mask == cp_lbl).sum())
        records.append({
            'roi_id': roi_id, 'tier': 'SILVER', 'iou_score': 0.0,
            'cellpose_label': cp_lbl, 's2p_label': -1,
            'centroid_y': cy, 'centroid_x': cx, 'area_px': area,
            's2p_iscell_prob': -1.0,
        })

    # BRONZE: Suite2p-only
    for j, s2p_lbl in enumerate(s2p_labels):
        if j in matched_s2p:
            continue
        roi_id += 1
        s2p_lbl = int(s2p_lbl)
        cy, cx  = _centroid(s2p_mask, s2p_lbl)
        area    = int((s2p_mask == s2p_lbl).sum())
        s2p_prob = float(s2p_iscell[s2p_lbl - 1, 1]) if s2p_iscell is not None and s2p_lbl - 1 < len(s2p_iscell) else -1.0
        records.append({
            'roi_id': roi_id, 'tier': 'BRONZE', 'iou_score': 0.0,
            'cellpose_label': -1, 's2p_label': s2p_lbl,
            'centroid_y': cy, 'centroid_x': cx, 'area_px': area,
            's2p_iscell_prob': round(s2p_prob, 4),
        })

    return records


def build_consensus_mask(cp_mask, s2p_mask, records, tiers=('GOLD', 'SILVER')):
    """
    Build a uint16 consensus mask from matched records.

    Parameters
    ----------
    cp_mask : np.ndarray
        Cellpose labeled mask.
    s2p_mask : np.ndarray
        Suite2p labeled mask.
    records : list of dicts
        Output of match_and_tier().
    tiers : tuple of str
        Which tiers to include in the output mask.

    Returns
    -------
    consensus : np.ndarray
        uint16 labeled mask with contiguous ROI IDs.
    """
    shape = cp_mask.shape
    consensus = np.zeros(shape, dtype=np.uint16)
    new_id = 0

    for rec in records:
        if rec['tier'] not in tiers:
            continue
        new_id += 1
        if rec['tier'] in ('GOLD', 'SILVER'):
            # Use Cellpose boundary (better spatial precision)
            consensus[cp_mask == rec['cellpose_label']] = new_id
        else:
            # BRONZE: only Suite2p pixels available
            consensus[s2p_mask == rec['s2p_label']] = new_id

    return consensus


def save_diagnostics(mean_img_path, cp_mask, s2p_mask, records, out_dir):
    """Save 3-panel diagnostic overlay figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print('  matplotlib not available, skipping diagnostics')
        return

    mean_img = tifffile.imread(mean_img_path).astype(np.float32)
    p_lo, p_hi = np.percentile(mean_img, [1, 99])
    mean_img = np.clip((mean_img - p_lo) / max(p_hi - p_lo, 1), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: GOLD outlines (green)
    axes[0].imshow(mean_img, cmap='gray')
    axes[0].set_title('GOLD (both agree)')
    for rec in records:
        if rec['tier'] == 'GOLD':
            _draw_contour(axes[0], cp_mask, rec['cellpose_label'], 'lime')

    # Panel 2: SILVER (yellow) + BRONZE (red) outlines
    axes[1].imshow(mean_img, cmap='gray')
    axes[1].set_title('SILVER (Cellpose) + BRONZE (Suite2p)')
    for rec in records:
        if rec['tier'] == 'SILVER':
            _draw_contour(axes[1], cp_mask, rec['cellpose_label'], 'yellow')
        elif rec['tier'] == 'BRONZE':
            _draw_contour(axes[1], s2p_mask, rec['s2p_label'], 'red')

    # Panel 3: IoU histogram for matched pairs
    gold_ious = [r['iou_score'] for r in records if r['tier'] == 'GOLD']
    axes[2].set_title('GOLD pair IoU distribution')
    if gold_ious:
        axes[2].hist(gold_ious, bins=20, range=(0, 1), color='green', alpha=0.7)
        axes[2].axvline(x=np.median(gold_ious), color='black', linestyle='--',
                        label=f'median={np.median(gold_ious):.2f}')
        axes[2].legend()
    axes[2].set_xlabel('IoU')
    axes[2].set_ylabel('Count')

    for ax in axes[:2]:
        ax.axis('off')

    n_gold   = sum(1 for r in records if r['tier'] == 'GOLD')
    n_silver = sum(1 for r in records if r['tier'] == 'SILVER')
    n_bronze = sum(1 for r in records if r['tier'] == 'BRONZE')
    fig.suptitle(f'GOLD={n_gold}  SILVER={n_silver}  BRONZE={n_bronze}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / 'diagnostic.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def _centroid(mask, label):
    """Return (y, x) centroid of a labeled ROI."""
    ys, xs = np.where(mask == label)
    if len(ys) == 0:
        return -1, -1
    return int(np.mean(ys)), int(np.mean(xs))


def _draw_contour(ax, mask, label, color):
    """Draw ROI contour on a matplotlib axis."""
    from scipy.ndimage import binary_dilation
    roi = mask == label
    dilated = binary_dilation(roi, iterations=1)
    contour = dilated & ~roi
    ys, xs = np.where(contour)
    if len(ys) > 0:
        ax.scatter(xs, ys, c=color, s=0.3, alpha=0.8, marker='.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cp_mask', required=True,
                    help='Cellpose uint16 mask TIFF')
    ap.add_argument('--s2p_mask', required=True,
                    help='Suite2p uint16 mask TIFF (from s2p_to_masks.py)')
    ap.add_argument('--s2p_iscell', default=None,
                    help='Suite2p iscell.npy for probability metadata')
    ap.add_argument('--mean_img', default=None,
                    help='Mean projection TIFF for diagnostic overlay')
    ap.add_argument('--stem', default=None,
                    help='FOV stem for output naming')
    ap.add_argument('--out_dir', default=None,
                    help='Output directory (default: inference/consensus)')
    ap.add_argument('--iou_threshold', type=float, default=None)
    ap.add_argument('--config', default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cons_cfg = cfg.get('consensus', {})
    iou_threshold = args.iou_threshold or cons_cfg.get('iou_threshold', 0.3)
    default_tiers = [t.upper() for t in cons_cfg.get('default_tiers', ['gold', 'silver'])]

    out_dir = Path(args.out_dir) if args.out_dir else BASE_DIR / cfg.get('paths', {}).get('consensus_output', 'inference/consensus')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load masks
    cp_mask  = tifffile.imread(args.cp_mask).astype(np.uint16)
    s2p_mask = tifffile.imread(args.s2p_mask).astype(np.uint16)

    # Verify shapes match
    if cp_mask.shape != s2p_mask.shape:
        print(f'WARNING: Shape mismatch — Cellpose {cp_mask.shape} vs Suite2p {s2p_mask.shape}')
        print('  Consider using s2p_to_masks.py --ref_mask to align dimensions')

    # Load optional iscell
    s2p_iscell = None
    if args.s2p_iscell and Path(args.s2p_iscell).exists():
        s2p_iscell = np.load(args.s2p_iscell)

    # Match and tier
    print(f'Matching ROIs (IoU threshold={iou_threshold})')
    records = match_and_tier(cp_mask, s2p_mask, iou_threshold, s2p_iscell)

    n_gold   = sum(1 for r in records if r['tier'] == 'GOLD')
    n_silver = sum(1 for r in records if r['tier'] == 'SILVER')
    n_bronze = sum(1 for r in records if r['tier'] == 'BRONZE')
    print(f'  GOLD={n_gold}  SILVER={n_silver}  BRONZE={n_bronze}')

    # Resolve stem for naming
    stem = args.stem or Path(args.cp_mask).stem.replace('_masks', '')

    # Save default consensus mask (GOLD + SILVER)
    consensus = build_consensus_mask(cp_mask, s2p_mask, records, tiers=tuple(default_tiers))
    tifffile.imwrite(out_dir / f'{stem}_consensus_masks.tif', consensus)
    print(f'  Default mask ({"+".join(default_tiers)}): {out_dir / f"{stem}_consensus_masks.tif"}')

    # Save all-tiers mask
    all_tiers = build_consensus_mask(cp_mask, s2p_mask, records,
                                     tiers=('GOLD', 'SILVER', 'BRONZE'))
    tifffile.imwrite(out_dir / f'{stem}_all_tiers_masks.tif', all_tiers)

    # Save per-ROI CSV
    csv_path = out_dir / f'{stem}_consensus.csv'
    fields = ['roi_id', 'tier', 'iou_score', 'cellpose_label', 's2p_label',
              'centroid_y', 'centroid_x', 'area_px', 's2p_iscell_prob']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    print(f'  CSV: {csv_path}')

    # Diagnostic overlay
    if args.mean_img:
        diag_dir = out_dir / 'diagnostics'
        diag_dir.mkdir(parents=True, exist_ok=True)
        save_diagnostics(args.mean_img, cp_mask, s2p_mask, records,
                         diag_dir / stem)
        print(f'  Diagnostics: {diag_dir / stem}')

if __name__ == '__main__':
    main()
