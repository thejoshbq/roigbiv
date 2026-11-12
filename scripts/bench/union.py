"""Union candidate pool: how much of the ground truth any proposer finds.

Every proposer (source) contributes raw ``(y, x)`` candidates. The pool is
their concatenation, with no merging. Pool recall bounds what any decider
choosing among the candidates can reach; unique recall per source says which
proposers earn their place (keep bar: 0.02, plan decision D13).
"""
# contract stub: U5 replaces the function bodies below; signatures are frozen.
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from bench.schema import UnionScore


def union_score(
    fov_id: str, gt: np.ndarray, sources: Mapping[str, np.ndarray], max_distance: float,
) -> UnionScore:
    """Score one FOV's pool against ``gt``.

    - ``n_candidates[s]``: rows of ``sources[s]`` (each ``(N, 2)``, may be empty).
    - ``pool_n_tp``: ``match_points(gt, concatenation of all sources, max_distance).n_tp``.
    - ``covered[s]``: ``match_points(gt, sources[s], max_distance).n_tp``.
    - ``unique[s]``: GT indices matched in ``match_points(gt, sources[s])``
      that are matched by no other source's own ``match_points`` call.

    Every dict has one key per source, in the order of ``sources``.
    """
    return UnionScore(
        fov_id=fov_id, n_gt=0, n_candidates={}, pool_n_tp=0, covered={}, unique={},
        max_distance=max_distance,
    )


def aggregate_union(scores: Sequence[UnionScore]) -> dict:
    """Micro-averaged summary over FOVs.

    Returns ``{"n_fovs", "n_gt", "pool_recall", "pool_precision",
    "candidates_per_fov", "recall": {s: ..}, "unique_recall": {s: ..}}`` where
    ``pool_recall = sum(pool_n_tp) / sum(n_gt)``, ``pool_precision =
    sum(pool_n_tp) / sum(n_candidates_total)``, ``candidates_per_fov`` is the
    mean of ``n_candidates_total``, ``recall[s] = sum(covered[s]) / sum(n_gt)``
    and ``unique_recall[s] = sum(unique[s]) / sum(n_gt)``. Sources are the
    union of every score's sources (a FOV without a source counts 0 for it).
    A ratio with a zero denominator is ``None``. An empty ``scores`` gives
    ``{"n_fovs": 0, "n_gt": 0, "pool_recall": None, "pool_precision": None,
    "candidates_per_fov": None, "recall": {}, "unique_recall": {}}``.
    """
    return {}
