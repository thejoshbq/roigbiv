"""Sweep infrastructure — vary each detector's operating-point knob(s) and
score every resulting point against ground truth, instead of reporting one
fixed-threshold precision/recall pair.

Two strategies, chosen per-detector by whether the knob is a post-hoc filter
on an already-detected candidate set (cheap: one detection call, N free
re-matches) or a structural parameter that changes what gets detected at all
(expensive: one detection call per grid point). See
``detectors/{opencv_blob,cellpose_centroid,suite2p_centroid}.py`` docstrings
for which knob is which and why.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from centroid_bakeoff.detector import CentroidDetectorInputs, CentroidDetectorResult
from centroid_bakeoff.point_match import PointMatchResult, match_points


@dataclass
class SweepPoint:
    """One (parameter combo, resulting match) sample along a sweep.

    ``centroids`` (the raw predicted points behind this point's match) is kept
    in-memory only — not part of ``to_dict()`` — so callers can re-match a
    method's best point at other ``max_distance`` values without re-running
    the detector (see ``run_centroid_bakeoff.py``'s max_distance sensitivity
    pass).
    """

    params: dict
    match: PointMatchResult
    n_pred: int
    runtime_s: float
    centroids: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return {
            "params": self.params, "n_pred": self.n_pred,
            "runtime_s": self.runtime_s, **self.match.to_dict(),
        }


@dataclass
class SweepResult:
    """All sweep points for one (method, FOV) pair, plus the best-F1 point."""

    method: str
    fov_stem: str
    points: list[SweepPoint] = field(default_factory=list)

    @property
    def best(self) -> Optional[SweepPoint]:
        scored = [p for p in self.points if p.match.f1 is not None]
        if not scored:
            return None
        return max(scored, key=lambda p: p.match.f1)

    def to_dict(self) -> dict:
        best = self.best
        return {
            "method": self.method, "fov_stem": self.fov_stem,
            "points": [p.to_dict() for p in self.points],
            "best": best.to_dict() if best is not None else None,
        }


def filter_by_score(result: CentroidDetectorResult, min_score: float) -> CentroidDetectorResult:
    """Keep only candidates with ``score >= min_score``.

    Returns *result* unchanged if it has no per-candidate scores (nothing to
    filter on) or no candidates at all.
    """
    if result.scores is None or result.n == 0:
        return result
    keep = result.scores >= min_score
    return CentroidDetectorResult(
        centroids=result.centroids[keep],
        scores=result.scores[keep],
        meta={**result.meta, "min_score": float(min_score), "n": int(keep.sum())},
    )


def rescore_sweep(
    result: CentroidDetectorResult,
    gt: np.ndarray,
    max_distance: float,
    thresholds,
    *,
    method: str,
    fov_stem: str,
    param_name: str = "min_score",
) -> SweepResult:
    """Free PR curve from one detection call: vary the post-hoc score cutoff
    and re-match at each threshold. No extra detector invocations.

    Requires ``result.scores`` to be populated — the detector's per-candidate
    confidence is what gets swept (Suite2p's ``iscell`` probability,
    Cellpose's per-mask mean cellprob).
    """
    if result.scores is None:
        raise ValueError(
            f"rescore_sweep needs result.scores populated (method={method!r} "
            "returned no per-candidate confidence to sweep)"
        )
    points: list[SweepPoint] = []
    for thr in thresholds:
        filtered = filter_by_score(result, thr)
        m = match_points(gt, filtered.centroids, max_distance=max_distance)
        points.append(SweepPoint(
            params={param_name: float(thr)}, match=m,
            n_pred=filtered.n, runtime_s=0.0, centroids=filtered.centroids,
        ))
    return SweepResult(method=method, fov_stem=fov_stem, points=points)


def param_grid_sweep(
    detector_factory: Callable[..., object],
    param_grid: dict[str, list],
    inputs: CentroidDetectorInputs,
    gt: np.ndarray,
    max_distance: float,
    *,
    method: str,
    fov_stem: str,
) -> SweepResult:
    """PR curve by re-invoking ``detect()`` once per grid combination.

    ``param_grid`` maps constructor kwarg name -> list of values;
    ``detector_factory(**combo)`` must return a fresh ``CentroidDetector``.
    Used where the knob changes what's structurally detected (OpenCV's blob
    filters, Cellpose's ``cellprob_threshold``, Suite2p's
    ``threshold_scaling``) — a post-hoc filter can't recover candidates that
    were never formed in the first place.
    """
    keys = list(param_grid.keys())
    points: list[SweepPoint] = []
    for values in itertools.product(*(param_grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        det = detector_factory(**combo)
        t0 = time.time()
        result = det.detect(inputs)
        elapsed = time.time() - t0
        m = match_points(gt, result.centroids, max_distance=max_distance)
        points.append(SweepPoint(
            params=combo, match=m, n_pred=result.n, runtime_s=round(elapsed, 2),
            centroids=result.centroids,
        ))
    return SweepResult(method=method, fov_stem=fov_stem, points=points)


def max_distance_sensitivity(
    point: SweepPoint, gt: np.ndarray, distances,
) -> list[PointMatchResult]:
    """Re-match one sweep point's already-detected centroids at several
    ``max_distance`` values — no detector re-invocation, just re-matching.
    Used to check whether a method's ranking survives a different match
    tolerance than the soma-radius default.
    """
    if point.centroids is None:
        raise ValueError("point.centroids is None — nothing to re-match")
    return [match_points(gt, point.centroids, max_distance=d) for d in distances]
