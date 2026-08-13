"""Unit tests for point_match.match_points — a flagged deviation from the rest
of scripts/'s no-test convention (see the implementation plan): a silent bug
here would corrupt every reported number.

Run standalone: pytest scripts/centroid_bakeoff/
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from centroid_bakeoff.point_match import match_points  # noqa: E402


def test_exact_match():
    gt = np.array([[10.0, 10.0], [50.0, 50.0]])
    pred = np.array([[10.0, 10.0], [50.0, 50.0]])
    r = match_points(gt, pred, max_distance=5.0)
    assert r.n_tp == 2 and r.n_fp == 0 and r.n_fn == 0
    assert r.precision == 1.0 and r.recall == 1.0 and r.f1 == 1.0
    assert r.mean_localization_error == 0.0


def test_one_false_positive():
    gt = np.array([[10.0, 10.0]])
    pred = np.array([[10.0, 10.0], [100.0, 100.0]])
    r = match_points(gt, pred, max_distance=5.0)
    assert r.n_tp == 1 and r.n_fp == 1 and r.n_fn == 0
    assert r.fp == [1]


def test_one_false_negative():
    gt = np.array([[10.0, 10.0], [100.0, 100.0]])
    pred = np.array([[10.0, 10.0]])
    r = match_points(gt, pred, max_distance=5.0)
    assert r.n_tp == 1 and r.n_fp == 0 and r.n_fn == 1
    assert r.fn == [1]


def test_beyond_max_distance_is_unmatched():
    gt = np.array([[10.0, 10.0]])
    pred = np.array([[20.0, 10.0]])  # 10px away
    r = match_points(gt, pred, max_distance=5.0)
    assert r.n_tp == 0 and r.n_fp == 1 and r.n_fn == 1


def test_empty_gt():
    gt = np.zeros((0, 2))
    pred = np.array([[1.0, 1.0], [2.0, 2.0]])
    r = match_points(gt, pred, max_distance=5.0)
    assert r.n_tp == 0 and r.n_fp == 2 and r.n_fn == 0
    assert r.precision == 0.0
    assert r.recall is None  # 0/0 undefined, not 0


def test_empty_pred():
    gt = np.array([[1.0, 1.0]])
    pred = np.zeros((0, 2))
    r = match_points(gt, pred, max_distance=5.0)
    assert r.n_tp == 0 and r.n_fp == 0 and r.n_fn == 1
    assert r.recall == 0.0
    assert r.precision is None


def test_crowded_triple_hungarian_beats_greedy():
    # A, B, C predictions with GT at those exact points except pred order is
    # shuffled so a *greedy* nearest-neighbor matcher (claiming its own best
    # match first, in prediction order) would double-claim one GT and leave
    # another unmatched, while Hungarian finds the perfect assignment.
    gt = np.array([[0.0, 0.0], [0.0, 3.0], [0.0, 6.0]])
    pred = np.array([[0.0, 3.1], [0.0, 0.1], [0.0, 6.1]])
    r = match_points(gt, pred, max_distance=1.0)
    assert r.n_tp == 3 and r.n_fp == 0 and r.n_fn == 0
    matched_gt = {t[0] for t in r.tp}
    matched_pred = {t[1] for t in r.tp}
    assert matched_gt == {0, 1, 2}
    assert matched_pred == {0, 1, 2}


def test_to_dict_roundtrip():
    gt = np.array([[10.0, 10.0]])
    pred = np.array([[10.0, 10.0]])
    r = match_points(gt, pred, max_distance=5.0)
    d = r.to_dict()
    assert d["n_tp"] == 1 and d["precision"] == 1.0
