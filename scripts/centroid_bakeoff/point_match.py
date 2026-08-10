"""Point-matching metric for the centroid bake-off.

Shape precedent: ``roigbiv.eval.match.MatchResult``/``iou_match`` (greedy IoU
matching between label masks). Centroids have no mask/IoU, so this uses
Euclidean distance instead, and Hungarian assignment (globally optimal)
instead of greedy claiming — greedy can mismatch in a crowded FOV where two
predictions are each other's second-best match; Hungarian doesn't.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# Cost assigned to a GT/pred pair farther apart than max_distance — large
# enough that the solver never "prefers" such a pair over leaving both
# unmatched, but finite so the cost matrix has no inf/nan.
_FORBIDDEN_COST = 1.0e6


@dataclass
class PointMatchResult:
    """One-to-one Hungarian matches between ground-truth and predicted centroids.

    Attributes
    ----------
    tp : list of (gt_idx, pred_idx, distance_px)
    fp : predicted indices with no GT match
    fn : GT indices with no predicted match
    max_distance : the gating threshold used for this result
    """

    tp: list[tuple[int, int, float]] = field(default_factory=list)
    fp: list[int] = field(default_factory=list)
    fn: list[int] = field(default_factory=list)
    max_distance: float = 0.0

    @property
    def n_tp(self) -> int:
        return len(self.tp)

    @property
    def n_fp(self) -> int:
        return len(self.fp)

    @property
    def n_fn(self) -> int:
        return len(self.fn)

    @property
    def precision(self) -> Optional[float]:
        denom = self.n_tp + self.n_fp
        return (self.n_tp / denom) if denom > 0 else None

    @property
    def recall(self) -> Optional[float]:
        denom = self.n_tp + self.n_fn
        return (self.n_tp / denom) if denom > 0 else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def mean_localization_error(self) -> Optional[float]:
        """Mean Euclidean distance (px) over true-positive pairs only."""
        if not self.tp:
            return None
        return float(np.mean([d for _, _, d in self.tp]))

    def to_dict(self) -> dict:
        return {
            "n_tp": self.n_tp, "n_fp": self.n_fp, "n_fn": self.n_fn,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "mean_localization_error": self.mean_localization_error,
            "max_distance": self.max_distance,
        }


def match_points(
    gt: np.ndarray, pred: np.ndarray, max_distance: float,
) -> PointMatchResult:
    """Hungarian assignment on the (N_gt, N_pred) Euclidean distance matrix.

    Pairs farther apart than ``max_distance`` are forbidden in the cost matrix
    (so the solver never "forces" an implausible match just to balance it),
    then any surviving assigned pair exceeding the threshold is discarded
    post-hoc as a defensive check.

    Parameters
    ----------
    gt, pred : (N, 2) arrays of (y, x) centroids.
    max_distance : maximum center-to-center distance (px) for a valid match.
    """
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 2)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)

    if len(gt) == 0 or len(pred) == 0:
        return PointMatchResult(
            tp=[], fp=list(range(len(pred))), fn=list(range(len(gt))),
            max_distance=max_distance,
        )

    D = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=-1)
    cost = np.where(D <= max_distance, D, _FORBIDDEN_COST)
    row_ind, col_ind = linear_sum_assignment(cost)

    tp: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if D[r, c] <= max_distance:
            tp.append((int(r), int(c), float(D[r, c])))
            used_pred.add(int(c))

    used_gt = {t[0] for t in tp}
    fp = [i for i in range(len(pred)) if i not in used_pred]
    fn = [i for i in range(len(gt)) if i not in used_gt]
    return PointMatchResult(tp=tp, fp=fp, fn=fn, max_distance=max_distance)
