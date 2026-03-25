"""
ROI G. Biv — IoU-based consensus matching between two Suite2p detection passes.

Computes pairwise IoU, runs Hungarian optimal assignment, and classifies each
union ROI into a confidence tier:
  GOLD   — found by both passes (IoU ≥ threshold)
  SILVER — anatomy-pass only   (morphologically neuron-shaped; silent during recording)
  BRONZE — activity-pass only  (was active; not anatomically prominent in mean image)
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_matrix(mask_a: np.ndarray, mask_b: np.ndarray):
    """
    Compute pairwise IoU between all labeled ROIs in two uint16 masks.

    Returns
    -------
    iou : np.ndarray, shape (n_a, n_b)
    labels_a : np.ndarray — unique non-zero labels in mask_a
    labels_b : np.ndarray — unique non-zero labels in mask_b
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


def match_and_tier(cp_mask: np.ndarray, s2p_mask: np.ndarray,
                   iou_threshold: float = 0.3,
                   s2p_iscell: np.ndarray = None) -> list:
    """
    Match ROIs between two masks and assign confidence tiers.

    In the union-ROI context, ``cp_mask`` is the anatomy-pass mask and
    ``s2p_mask`` is the activity-pass mask — the naming is inherited from the
    original match_rois.py where Cellpose filled the 'cp' role.

    Parameters
    ----------
    cp_mask : np.ndarray
        uint16 labeled mask (anatomy pass plays this role in union building).
    s2p_mask : np.ndarray
        uint16 labeled mask (activity pass).
    iou_threshold : float
        Minimum IoU for a GOLD match.
    s2p_iscell : np.ndarray or None
        Suite2p iscell array (n_rois, 2) for probability metadata.

    Returns
    -------
    records : list of dicts
        One dict per ROI with keys: roi_id, tier, iou_score, cellpose_label,
        s2p_label, centroid_y, centroid_x, area_px, s2p_iscell_prob.
    """
    iou_matrix, cp_labels, s2p_labels = compute_iou_matrix(cp_mask, s2p_mask)
    records = []
    matched_cp = set()
    matched_s2p = set()
    gold_pairs = []

    if iou_matrix.size > 0:
        row_idx, col_idx = linear_sum_assignment(-iou_matrix)
        for r, c in zip(row_idx, col_idx):
            score = iou_matrix[r, c]
            if score >= iou_threshold:
                gold_pairs.append((r, c, score))
                matched_cp.add(r)
                matched_s2p.add(c)

    roi_id = 0

    # GOLD
    for r, c, score in gold_pairs:
        roi_id += 1
        cp_lbl  = int(cp_labels[r])
        s2p_lbl = int(s2p_labels[c])
        cy, cx  = _centroid(cp_mask, cp_lbl)
        area    = int((cp_mask == cp_lbl).sum())
        s2p_prob = (float(s2p_iscell[s2p_lbl - 1, 1])
                    if s2p_iscell is not None and s2p_lbl - 1 < len(s2p_iscell)
                    else -1.0)
        records.append({
            "roi_id": roi_id, "tier": "GOLD", "iou_score": round(score, 4),
            "cellpose_label": cp_lbl, "s2p_label": s2p_lbl,
            "centroid_y": cy, "centroid_x": cx, "area_px": area,
            "s2p_iscell_prob": round(s2p_prob, 4),
        })

    # SILVER (cp_mask only)
    for i, cp_lbl in enumerate(cp_labels):
        if i in matched_cp:
            continue
        roi_id += 1
        cp_lbl = int(cp_lbl)
        cy, cx = _centroid(cp_mask, cp_lbl)
        area   = int((cp_mask == cp_lbl).sum())
        records.append({
            "roi_id": roi_id, "tier": "SILVER", "iou_score": 0.0,
            "cellpose_label": cp_lbl, "s2p_label": -1,
            "centroid_y": cy, "centroid_x": cx, "area_px": area,
            "s2p_iscell_prob": -1.0,
        })

    # BRONZE (s2p_mask only)
    for j, s2p_lbl in enumerate(s2p_labels):
        if j in matched_s2p:
            continue
        roi_id += 1
        s2p_lbl = int(s2p_lbl)
        cy, cx  = _centroid(s2p_mask, s2p_lbl)
        area    = int((s2p_mask == s2p_lbl).sum())
        s2p_prob = (float(s2p_iscell[s2p_lbl - 1, 1])
                    if s2p_iscell is not None and s2p_lbl - 1 < len(s2p_iscell)
                    else -1.0)
        records.append({
            "roi_id": roi_id, "tier": "BRONZE", "iou_score": 0.0,
            "cellpose_label": -1, "s2p_label": s2p_lbl,
            "centroid_y": cy, "centroid_x": cx, "area_px": area,
            "s2p_iscell_prob": round(s2p_prob, 4),
        })

    return records


def build_consensus_mask(cp_mask: np.ndarray, s2p_mask: np.ndarray,
                         records: list,
                         tiers: tuple = ("GOLD", "SILVER")) -> np.ndarray:
    """
    Build a uint16 consensus mask from matched ROI records.

    GOLD and SILVER ROIs use the anatomy (cp_mask) boundary for better spatial
    precision. BRONZE ROIs use the activity (s2p_mask) pixels.

    Parameters
    ----------
    cp_mask : np.ndarray  — anatomy-pass mask (Cellpose role)
    s2p_mask : np.ndarray — activity-pass mask
    records : list        — output of match_and_tier()
    tiers : tuple         — which tiers to include

    Returns
    -------
    consensus : np.ndarray, dtype=uint16
    """
    consensus = np.zeros(cp_mask.shape, dtype=np.uint16)
    new_id = 0
    for rec in records:
        if rec["tier"] not in tiers:
            continue
        new_id += 1
        if rec["tier"] in ("GOLD", "SILVER"):
            consensus[cp_mask == rec["cellpose_label"]] = new_id
        else:
            consensus[s2p_mask == rec["s2p_label"]] = new_id
    return consensus


def _centroid(mask: np.ndarray, label: int) -> tuple:
    ys, xs = np.where(mask == label)
    if len(ys) == 0:
        return -1, -1
    return int(np.mean(ys)), int(np.mean(xs))
