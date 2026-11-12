"""Centroid scoring: recall and precision per FOV and per lens, and inter-rater agreement.

A predicted centroid matches a ground-truth centroid when they are within
``max_distance`` pixels, one-to-one (Hungarian), using
``centroid_bakeoff.point_match.match_points``. The default distance is half
the FOV's soma diameter, measured from the annotation's ROI areas.
"""
# contract stub: U2 replaces the function bodies below; signatures are frozen.
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np

from bench.schema import FovScore, GroupScore, RaterPair

DEFAULT_DISTANCE_FRACTION = 0.5


def soma_diameter_px(areas_px: Union[np.ndarray, Sequence[float]]) -> float:
    """Equivalent-circle diameter of the median ROI area: ``2 * sqrt(median / pi)``.

    ``nan`` entries (POINT ROIs) are dropped first. Raise ``ValueError`` when
    nothing is left or any remaining area is not positive.
    """
    return 0.0


def max_distance_px(soma_diameter: float, fraction: float = DEFAULT_DISTANCE_FRACTION) -> float:
    """``fraction * soma_diameter``. Raise ``ValueError`` if either is not positive."""
    return 0.0


def score_fov(
    fov_id: str, lens: str, gt: np.ndarray, pred: np.ndarray, max_distance: float,
) -> FovScore:
    """Match ``pred`` against ``gt`` (both ``(N, 2)`` ``(y, x)``) with ``match_points``."""
    return FovScore(fov_id=fov_id, lens=lens, n_gt=0, n_pred=0, n_tp=0, max_distance=max_distance)


def aggregate(scores: Sequence[FovScore], by: str = "lens") -> dict[str, GroupScore]:
    """Group FOV scores and sum their counts.

    ``by="lens"`` keys the result by each score's ``lens``; ``by="all"`` gives
    one group keyed ``"all"``. Any other value raises ``ValueError``.
    ``macro_recall`` is the mean of per-FOV ``recall`` over the group's FOVs
    with ``n_gt > 0``, ``None`` when there are none. An empty ``scores`` gives
    ``{}``.
    """
    return {}


def inter_rater(
    fov_id: str, raters: Mapping[str, np.ndarray], max_distance: float,
) -> list[RaterPair]:
    """Every unordered pair of raters, as ``RaterPair(rater_a < rater_b)``.

    Pairs are listed in sorted order of ``(rater_a, rater_b)``. ``n_tp`` is
    ``match_points(raters[a], raters[b], max_distance).n_tp``. Fewer than two
    raters gives ``[]``.
    """
    return []


def inter_rater_ceiling(pairs: Sequence[RaterPair]) -> Optional[float]:
    """Mean ``f1`` over the pairs whose ``f1`` is not ``None``; ``None`` if none."""
    return None
