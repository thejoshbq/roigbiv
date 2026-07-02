"""Object-level IoU-based ROI matching for the benchmark harness (issue #30, roadmap item A6).

Matches a predicted label image against a manual ground-truth label image via
optimal (Hungarian) bipartite assignment on IoU (scipy.optimize.linear_sum_assignment),
rather than the greedy one-to-one heuristic used elsewhere in the repo.

Distinct from roigbiv.eval, which is an earlier, ad-hoc dict-based scoring
harness (stratified_metrics, iou_match) for the sequential pipeline's own QC
loop. This module is the formalized matcher for the separate roigbiv-bench
roadmap (Milestone A) and does not depend on or modify roigbiv.eval.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from roigbiv.benchmark.metrics import DetectionMetrics


@dataclass
class MatchResult:
    """One-to-one optimal IoU matches between GT and predicted label images.

    Terminology mirrors the caller's perspective: GT is the reference,
    pred is what the pipeline or baseline produced. Unlike roigbiv.eval.match's
    MatchResult, assignment is by global-optimal (Hungarian) IoU maximization
    rather than greedy claiming.

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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict (tp tuples become lists)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "MatchResult":
        """Deserialize from a dict produced by to_dict(), rebuilding tp as tuples."""
        return cls(
            tp=[tuple(t) for t in payload.get("tp", [])],
            fp=list(payload.get("fp", [])),
            fn=list(payload.get("fn", [])),
            min_iou=payload.get("min_iou", 0.3),
        )


def _iou_matrix(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a dense IoU matrix between all GT and predicted labels.

    Parameters
    ----------
    gt_labels, pred_labels : (H, W) integer arrays; 0 = background.

    Returns
    -------
    ids_gt : (n_gt,) sorted nonzero unique labels from gt_labels.
    ids_pred : (n_pred,) sorted nonzero unique labels from pred_labels.
    M : (n_gt, n_pred) float IoU matrix; M[i, j] = IoU(ids_gt[i], ids_pred[j]).

    Raises
    ------
    ValueError
        If gt_labels and pred_labels have different shapes — they must be
        two label images of the same FOV.
    """
    gt_labels = np.asarray(gt_labels)
    pred_labels = np.asarray(pred_labels)

    if gt_labels.shape != pred_labels.shape:
        raise ValueError(
            f"gt_labels and pred_labels must have the same shape, "
            f"got {gt_labels.shape} and {pred_labels.shape}"
        )

    ids_gt = np.unique(gt_labels)
    ids_gt = ids_gt[ids_gt != 0]
    ids_pred = np.unique(pred_labels)
    ids_pred = ids_pred[ids_pred != 0]

    masks_gt = {int(i): (gt_labels == i) for i in ids_gt}
    masks_pred = {int(i): (pred_labels == i) for i in ids_pred}
    area_gt = {i: int(m.sum()) for i, m in masks_gt.items()}
    area_pred = {i: int(m.sum()) for i, m in masks_pred.items()}

    M = np.zeros((len(ids_gt), len(ids_pred)), dtype=float)
    for i, ig in enumerate(ids_gt):
        mg = masks_gt[int(ig)]
        for j, ip in enumerate(ids_pred):
            mp = masks_pred[int(ip)]
            inter = int((mg & mp).sum())
            if inter == 0:
                continue
            union = area_gt[int(ig)] + area_pred[int(ip)] - inter
            M[i, j] = inter / union if union > 0 else 0.0

    return ids_gt, ids_pred, M


def _assign(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
) -> tuple[list[tuple[int, int, float]], np.ndarray, np.ndarray]:
    """Compute the optimal one-to-one assignment via the Hungarian algorithm.

    Parameters
    ----------
    gt_labels, pred_labels : (H, W) integer arrays; 0 = background.

    Returns
    -------
    assignment : list of (gt_label, pred_label, iou) chosen to maximize total
        IoU. Length is min(n_gt, n_pred); may include iou == 0 pairs if the
        optimal solver had no better option for an unbalanced matrix.
    ids_gt, ids_pred : sorted nonzero label arrays used to index M.
    """
    ids_gt, ids_pred, M = _iou_matrix(gt_labels, pred_labels)

    if ids_gt.size == 0 or ids_pred.size == 0:
        return [], ids_gt, ids_pred

    # Minimizing cost == maximizing IoU. Zero-overlap pairs already land at
    # cost 1.0, no worse than any real overlap (cost < 1.0), so the solver
    # naturally avoids them whenever a better pairing exists — no separate
    # sentinel cost is needed even when padding an unbalanced matrix.
    cost = 1.0 - M
    row_ind, col_ind = linear_sum_assignment(cost)

    assignment = [
        (int(ids_gt[r]), int(ids_pred[c]), float(M[r, c]))
        for r, c in zip(row_ind, col_ind)
    ]
    return assignment, ids_gt, ids_pred


def _split(
    assignment: list[tuple[int, int, float]],
    ids_gt: np.ndarray,
    ids_pred: np.ndarray,
    min_iou: float,
) -> MatchResult:
    """Partition an assignment into tp/fp/fn at a given IoU threshold.

    Parameters
    ----------
    assignment : list of (gt_label, pred_label, iou) from _assign.
    ids_gt, ids_pred : sorted unique label arrays (from _assign).
    min_iou : threshold; iou >= min_iou is a match, else the pair splits into
        an fn (gt) and an fp (pred).

    Returns
    -------
    MatchResult with tp/fp/fn partitioned at min_iou. Labels that never
    appeared in `assignment` at all (unbalanced gt/pred counts) are also
    added to fn/fp respectively.
    """
    tp: list[tuple[int, int, float]] = []
    fp: list[int] = []
    fn: list[int] = []
    assigned_gt = set()
    assigned_pred = set()

    for ig, ip, iou in assignment:
        assigned_gt.add(ig)
        assigned_pred.add(ip)
        if iou >= min_iou:
            tp.append((ig, ip, iou))
        else:
            fn.append(ig)
            fp.append(ip)

    fn.extend(int(i) for i in ids_gt if int(i) not in assigned_gt)
    fp.extend(int(i) for i in ids_pred if int(i) not in assigned_pred)

    return MatchResult(tp=tp, fp=fp, fn=fn, min_iou=min_iou)


def match(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    min_iou: float = 0.3,
) -> MatchResult:
    """Optimal one-to-one IoU matching between a GT and predicted label image.

    Parameters
    ----------
    gt_labels, pred_labels : (H, W) integer arrays; 0 = background.
    min_iou : minimum IoU for a valid match (default 0.3).

    Returns
    -------
    MatchResult with tp = [(gt_label, pred_label, iou), ...],
    fp = unmatched pred labels, fn = unmatched GT labels.
    """
    assignment, ids_gt, ids_pred = _assign(gt_labels, pred_labels)
    return _split(assignment, ids_gt, ids_pred, min_iou)


def match_at_thresholds(
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
) -> dict[float, MatchResult]:
    """Compute matches at multiple IoU thresholds from a single optimal assignment.

    The optimal assignment (which pred pairs with which gt) is
    threshold-independent — only the final tp/fp/fn split depends on
    min_iou — so the Hungarian solve runs once and is reused for every
    threshold.

    Parameters
    ----------
    gt_labels, pred_labels : (H, W) integer arrays; 0 = background.
    thresholds : IoU thresholds to report (default 0.3, 0.5, 0.7).

    Returns
    -------
    dict mapping threshold -> MatchResult.
    """
    assignment, ids_gt, ids_pred = _assign(gt_labels, pred_labels)
    return {t: _split(assignment, ids_gt, ids_pred, t) for t in thresholds}


def detection_metrics(result: MatchResult) -> DetectionMetrics:
    """Compute detection quality metrics from a MatchResult.

    Bridges roigbiv.benchmark.matcher.MatchResult into the pure data model
    roigbiv.benchmark.metrics.DetectionMetrics (which defers metric
    computation logic to "the matcher").

    Parameters
    ----------
    result : MatchResult to summarize.

    Returns
    -------
    DetectionMetrics with false_positive_count/false_negative_count always
    set, and precision/recall/f1/mean_iou/median_iou set to None where their
    denominators are undefined (zero predictions, zero GT, or zero matches).
    """
    n_tp, n_fp, n_fn = result.n_tp, result.n_fp, result.n_fn

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else None
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    ious = [iou for _, _, iou in result.tp]
    mean_iou = float(np.mean(ious)) if ious else None
    median_iou = float(np.median(ious)) if ious else None

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou=mean_iou,
        median_iou=median_iou,
        false_positive_count=n_fp,
        false_negative_count=n_fn,
    )


def save_match_result(result: MatchResult, path: str | Path) -> Path:
    """Serialize a single MatchResult to JSON at the given path.

    Parameters
    ----------
    result : MatchResult to save.
    path : full file path (not a directory). Parent directories are created
        as needed.

    Returns
    -------
    Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


def save_match_results(results: dict[float, MatchResult], path: str | Path) -> Path:
    """Serialize a {threshold: MatchResult} dict to a single JSON file.

    Structured as {"thresholds": {str(threshold): result.to_dict(), ...}}
    since JSON object keys must be strings.

    Parameters
    ----------
    results : mapping of threshold -> MatchResult, typically the output of
        match_at_thresholds.
    path : full file path. Parent directories are created as needed.

    Returns
    -------
    Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"thresholds": {str(t): r.to_dict() for t, r in results.items()}}
    path.write_text(json.dumps(payload, indent=2))
    return path
