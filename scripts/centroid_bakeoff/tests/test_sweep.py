"""Unit tests for sweep.py — hand-built cases, no detectors/GPU/Suite2p involved."""
from __future__ import annotations

import numpy as np
import pytest

from centroid_bakeoff.detector import CentroidDetectorInputs, CentroidDetectorResult
from centroid_bakeoff.point_match import match_points
from centroid_bakeoff.sweep import (
    SweepPoint, SweepResult, filter_by_score, max_distance_sensitivity,
    param_grid_sweep, rescore_sweep,
)


def _result(centroids, scores):
    return CentroidDetectorResult(
        centroids=np.asarray(centroids, dtype=np.float32).reshape(-1, 2),
        scores=np.asarray(scores, dtype=np.float32),
        meta={"n": len(centroids)},
    )


def test_filter_by_score_keeps_inclusive():
    result = _result([[0, 0], [1, 1], [2, 2]], [0.1, 0.3, 0.5])
    filtered = filter_by_score(result, 0.3)
    assert filtered.n == 2
    np.testing.assert_array_equal(filtered.centroids, [[1, 1], [2, 2]])
    assert filtered.meta["min_score"] == 0.3


def test_filter_by_score_no_scores_passthrough():
    result = CentroidDetectorResult(
        centroids=np.array([[0, 0]], dtype=np.float32), scores=None, meta={},
    )
    assert filter_by_score(result, 0.5) is result


def test_filter_by_score_empty_result():
    result = _result([], [])
    filtered = filter_by_score(result, 0.5)
    assert filtered.n == 0


def test_rescore_sweep_recovers_known_pr_curve():
    # 3 GT points; 3 predictions exactly on top of them with descending scores.
    gt = np.array([[0, 0], [10, 10], [20, 20]], dtype=np.float32)
    result = _result(gt.tolist(), [0.9, 0.5, 0.1])

    sweep = rescore_sweep(
        result, gt, max_distance=1.0, thresholds=[0.0, 0.4, 0.6, 1.0],
        method="stub", fov_stem="fov1",
    )
    by_thr = {p.params["min_score"]: p for p in sweep.points}

    assert by_thr[0.0].match.n_tp == 3      # everything kept -> all 3 match
    assert by_thr[0.4].match.n_tp == 2      # 0.9, 0.5 survive
    assert by_thr[0.6].match.n_tp == 1      # only 0.9 survives
    assert by_thr[1.0].match.n_tp == 0      # nothing survives >= 1.0

    assert sweep.best.params["min_score"] == 0.0
    assert sweep.best.match.f1 == 1.0


def test_rescore_sweep_requires_scores():
    result = CentroidDetectorResult(
        centroids=np.array([[0, 0]], dtype=np.float32), scores=None, meta={},
    )
    with pytest.raises(ValueError):
        rescore_sweep(
            result, np.zeros((0, 2)), max_distance=1.0, thresholds=[0.5],
            method="stub", fov_stem="fov1",
        )


class _StubDetector:
    name = "stub"

    def __init__(self, offset):
        self.offset = offset

    def detect(self, inputs):
        base = np.array([[0, 0], [10, 10]], dtype=np.float32)
        return CentroidDetectorResult(
            centroids=base + self.offset, scores=None, meta={},
        )


def test_param_grid_sweep_calls_factory_once_per_combo():
    calls = []

    def factory(**combo):
        calls.append(combo)
        return _StubDetector(**combo)

    gt = np.array([[0, 0], [10, 10]], dtype=np.float32)
    inputs = CentroidDetectorInputs(summary={}, fov_stem="fov1", shape=(32, 32))

    sweep = param_grid_sweep(
        factory, {"offset": [0.0, 0.5, 5.0]}, inputs, gt, max_distance=1.0,
        method="stub", fov_stem="fov1",
    )

    assert calls == [{"offset": 0.0}, {"offset": 0.5}, {"offset": 5.0}]
    assert len(sweep.points) == 3
    by_offset = {p.params["offset"]: p for p in sweep.points}
    assert by_offset[0.0].match.n_tp == 2     # exact overlap
    assert by_offset[0.5].match.n_tp == 2     # within max_distance
    assert by_offset[5.0].match.n_tp == 0     # too far, no match

    assert sweep.best.params["offset"] in (0.0, 0.5)   # both f1=1.0; max() picks first


def test_sweep_result_best_is_none_when_no_points():
    sweep = SweepResult(method="stub", fov_stem="fov1", points=[])
    assert sweep.best is None


def test_max_distance_sensitivity_rematches_without_rerun():
    gt = np.array([[0, 0], [10, 10]], dtype=np.float32)
    pred = np.array([[0.5, 0.5], [10, 10]], dtype=np.float32)
    m = match_points(gt, pred, max_distance=1.0)
    point = SweepPoint(params={}, match=m, n_pred=2, runtime_s=0.0, centroids=pred)

    results = max_distance_sensitivity(point, gt, distances=[0.1, 1.0, 5.0])
    assert results[0].n_tp == 1     # (0,0)->(0.5,0.5) is ~0.71px, exceeds a 0.1px tolerance
    assert results[1].n_tp == 2
    assert results[2].n_tp == 2


def test_max_distance_sensitivity_requires_centroids():
    m = match_points(np.zeros((0, 2)), np.zeros((0, 2)), max_distance=1.0)
    point = SweepPoint(params={}, match=m, n_pred=0, runtime_s=0.0, centroids=None)
    with pytest.raises(ValueError):
        max_distance_sensitivity(point, np.zeros((0, 2)), distances=[1.0])


def test_sweep_point_to_dict_roundtrip():
    gt = np.array([[0, 0]], dtype=np.float32)
    pred = np.array([[0, 0]], dtype=np.float32)
    m = match_points(gt, pred, max_distance=1.0)
    point = SweepPoint(params={"x": 1}, match=m, n_pred=1, runtime_s=0.1)
    d = point.to_dict()
    assert d["params"] == {"x": 1}
    assert d["n_pred"] == 1
    assert d["n_tp"] == 1
