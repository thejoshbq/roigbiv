"""IoU-based ROI matching for evaluation.

Adapted from scripts/diagnostic_compare.py:iou_match().
Matching threshold: IoU >= 0.3 (configurable). Greedy one-to-one assignment,
highest-IoU pairs claimed first.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchResult:
    """One-to-one greedy IoU matches between GT and predicted label images.

    Terminology mirrors the caller's perspective: GT is the reference,
    pred is what the pipeline or baseline produced.

    Attributes
    ----------
    tp : list of (gt_label, pred_label, iou)
    fp : pred labels with no GT match
    fn : GT labels with no pred match
    min_iou : threshold used for this result
    """

    tp: list[tuple[int, int, float]] = field(default_factory=list)
    fp: list[int] = field(default_factory=list)
    fn: list[int] = field(default_factory=list)
    min_iou: float = 0.3

    @property
    def n_tp(self) -> int:
        return len(self.tp)

    @property
    def n_fp(self) -> int:
        return len(self.fp)

    @property
    def n_fn(self) -> int:
        return len(self.fn)


def iou_match(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    min_iou: float = 0.3,
) -> MatchResult:
    """Greedy IoU matching between a GT label image and a predicted label image.

    Parameters
    ----------
    gt_labels, pred_labels : (H, W) integer arrays; 0 = background.
    min_iou : minimum IoU for a valid match (default 0.3).

    Returns
    -------
    MatchResult with tp = [(gt_label, pred_label, iou), ...],
    fp = unmatched pred labels, fn = unmatched GT labels.
    """
    gt_labels = np.asarray(gt_labels)
    pred_labels = np.asarray(pred_labels)

    ids_gt = np.unique(gt_labels)
    ids_gt = ids_gt[ids_gt != 0]
    ids_pred = np.unique(pred_labels)
    ids_pred = ids_pred[ids_pred != 0]

    masks_gt = {int(i): (gt_labels == i) for i in ids_gt}
    masks_pred = {int(i): (pred_labels == i) for i in ids_pred}
    area_gt = {i: int(m.sum()) for i, m in masks_gt.items()}
    area_pred = {i: int(m.sum()) for i, m in masks_pred.items()}

    pairs: list[tuple[int, int, float]] = []
    for ig, mg in masks_gt.items():
        best_iou = 0.0
        best_ip = None
        for ip, mp in masks_pred.items():
            inter = int((mg & mp).sum())
            if inter == 0:
                continue
            union = area_gt[ig] + area_pred[ip] - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_ip = ip
        if best_ip is not None and best_iou >= min_iou:
            pairs.append((ig, best_ip, best_iou))

    pairs.sort(key=lambda p: -p[2])
    used_gt, used_pred = set(), set()
    tp: list[tuple[int, int, float]] = []
    for ig, ip, iou in pairs:
        if ig in used_gt or ip in used_pred:
            continue
        used_gt.add(ig)
        used_pred.add(ip)
        tp.append((ig, ip, iou))

    fp = [int(i) for i in ids_pred if int(i) not in used_pred]
    fn = [int(i) for i in ids_gt if int(i) not in used_gt]
    return MatchResult(tp=tp, fp=fp, fn=fn, min_iou=min_iou)
