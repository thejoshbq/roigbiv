"""Union candidate pool: how much of the ground truth any proposer finds.

Every proposer (source) contributes raw ``(y, x)`` candidates. The pool is
their concatenation, with no merging. Pool recall bounds what any decider
choosing among the candidates can reach; unique recall per source says which
proposers earn their place (keep bar: 0.02, plan decision D13).
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from bench.schema import UnionScore
from centroid_bakeoff.point_match import match_points


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
    gt = np.asarray(gt, dtype=float).reshape(-1, 2)
    arrays = {s: np.asarray(c, dtype=float).reshape(-1, 2) for s, c in sources.items()}
    pool = np.concatenate(list(arrays.values())) if arrays else np.zeros((0, 2))

    matched = {}
    covered = {}
    for s, cand in arrays.items():
        res = match_points(gt, cand, max_distance)
        matched[s] = {g for g, _, _ in res.tp}
        covered[s] = res.n_tp

    unique = {
        s: len(matched[s] - set().union(*(matched[t] for t in matched if t != s)))
        for s in arrays
    }

    return UnionScore(
        fov_id=fov_id,
        n_gt=len(gt),
        n_candidates={s: len(c) for s, c in arrays.items()},
        pool_n_tp=match_points(gt, pool, max_distance).n_tp,
        covered=covered,
        unique=unique,
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
    n_fovs = len(scores)
    n_gt = sum(s.n_gt for s in scores)
    pool_tp = sum(s.pool_n_tp for s in scores)
    n_cand = sum(s.n_candidates_total for s in scores)
    sources = list(dict.fromkeys(k for s in scores for k in s.n_candidates))

    def ratio(num: float, den: float) -> Optional[float]:
        return num / den if den else None

    return {
        "n_fovs": n_fovs,
        "n_gt": n_gt,
        "pool_recall": ratio(pool_tp, n_gt),
        "pool_precision": ratio(pool_tp, n_cand),
        "candidates_per_fov": n_cand / n_fovs if n_fovs else None,
        "recall": {
            src: ratio(sum(s.covered.get(src, 0) for s in scores), n_gt) for src in sources
        },
        "unique_recall": {
            src: ratio(sum(s.unique.get(src, 0) for s in scores), n_gt) for src in sources
        },
    }
