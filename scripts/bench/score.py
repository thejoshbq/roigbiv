"""Centroid scoring: recall and precision per FOV and per lens, and inter-rater agreement.

A predicted centroid matches a ground-truth centroid when they are within
``max_distance`` pixels, one-to-one (Hungarian), using
``centroid_bakeoff.point_match.match_points``. The default distance is half
the FOV's soma diameter, measured from the annotation's ROI areas.
"""
from __future__ import annotations

import itertools
import math
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from bench.schema import FovScore, GroupScore, RaterPair
from centroid_bakeoff.point_match import match_points

DEFAULT_DISTANCE_FRACTION = 0.5


def soma_diameter_px(areas_px: Union[np.ndarray, Sequence[float]]) -> float:
    """Equivalent-circle diameter of the median ROI area: ``2 * sqrt(median / pi)``.

    ``nan`` entries (POINT ROIs) are dropped first. Raise ``ValueError`` when
    nothing is left or any remaining area is not positive.
    """
    areas = np.asarray(areas_px, dtype=float).ravel()
    areas = areas[~np.isnan(areas)]
    if areas.size == 0:
        raise ValueError("no ROI areas left after dropping nan")
    if np.any(areas <= 0):
        raise ValueError("ROI areas must be positive")
    return 2.0 * math.sqrt(float(np.median(areas)) / math.pi)


def max_distance_px(soma_diameter: float, fraction: float = DEFAULT_DISTANCE_FRACTION) -> float:
    """``fraction * soma_diameter``. Raise ``ValueError`` if either is not positive."""
    if not soma_diameter > 0 or not fraction > 0:
        raise ValueError("soma_diameter and fraction must be positive")
    return fraction * soma_diameter


def score_fov(
    fov_id: str, lens: str, gt: np.ndarray, pred: np.ndarray, max_distance: float,
) -> FovScore:
    """Match ``pred`` against ``gt`` (both ``(N, 2)`` ``(y, x)``) with ``match_points``."""
    result = match_points(gt, pred, max_distance)
    return FovScore(
        fov_id=fov_id,
        lens=lens,
        n_gt=len(np.asarray(gt).reshape(-1, 2)),
        n_pred=len(np.asarray(pred).reshape(-1, 2)),
        n_tp=result.n_tp,
        max_distance=max_distance,
    )


def aggregate(scores: Sequence[FovScore], by: str = "lens") -> dict[str, GroupScore]:
    """Group FOV scores and sum their counts.

    ``by="lens"`` keys the result by each score's ``lens``; ``by="all"`` gives
    one group keyed ``"all"``. Any other value raises ``ValueError``.
    ``macro_recall`` is the mean of per-FOV ``recall`` over the group's FOVs
    with ``n_gt > 0``, ``None`` when there are none. An empty ``scores`` gives
    ``{}``.
    """
    if by not in ("lens", "all"):
        raise ValueError(f"by must be 'lens' or 'all', got {by!r}")

    groups: dict[str, list[FovScore]] = {}
    for s in scores:
        key = s.lens if by == "lens" else "all"
        groups.setdefault(key, []).append(s)

    out: dict[str, GroupScore] = {}
    for key, members in groups.items():
        recalls = [s.recall for s in members if s.n_gt > 0]
        out[key] = GroupScore(
            group=key,
            n_fovs=len(members),
            n_gt=sum(s.n_gt for s in members),
            n_pred=sum(s.n_pred for s in members),
            n_tp=sum(s.n_tp for s in members),
            macro_recall=float(np.mean(recalls)) if recalls else None,
        )
    return out


def inter_rater(
    fov_id: str, raters: Mapping[str, np.ndarray], max_distance: float,
) -> list[RaterPair]:
    """Every unordered pair of raters, as ``RaterPair(rater_a < rater_b)``.

    Pairs are listed in sorted order of ``(rater_a, rater_b)``. ``n_tp`` is
    ``match_points(raters[a], raters[b], max_distance).n_tp``. Fewer than two
    raters gives ``[]``.
    """
    pairs: list[RaterPair] = []
    for a, b in itertools.combinations(sorted(raters), 2):
        pts_a = np.asarray(raters[a]).reshape(-1, 2)
        pts_b = np.asarray(raters[b]).reshape(-1, 2)
        result = match_points(pts_a, pts_b, max_distance)
        pairs.append(RaterPair(
            fov_id=fov_id,
            rater_a=a,
            rater_b=b,
            n_a=len(pts_a),
            n_b=len(pts_b),
            n_tp=result.n_tp,
            max_distance=max_distance,
        ))
    return pairs


def inter_rater_ceiling(pairs: Sequence[RaterPair]) -> Optional[float]:
    """Mean ``f1`` over the pairs whose ``f1`` is not ``None``; ``None`` if none."""
    f1s = [p.f1 for p in pairs if p.f1 is not None]
    return float(np.mean(f1s)) if f1s else None
