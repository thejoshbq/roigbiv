"""Accept tests for U7 (``bench.decide``). Written by the contract; workers do not edit them."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.decide import FovCandidates, fit_and_score  # noqa: E402
from centroid_bakeoff.detector import CentroidDetectorResult  # noqa: E402

GRID = np.array([[20, 20], [20, 40], [20, 60], [40, 20], [40, 40], [40, 60]], dtype=float)
CP_SCORES = [2.0, 1.8, 2.2, 1.9, 2.1, 2.0]
S2P_SCORES = [0.9, 0.8, 0.95, 0.85, 0.9, 0.88]
SPURIOUS = np.array([[80, 80], [80, 100], [100, 80], [100, 100]], dtype=float)
SPURIOUS_SCORES = [0.1, 0.15, 0.12, 0.2]


def make(fov_id, gt, offset=(0.0, 0.0), spurious=True, spurious_scores=None, lens="GRIN"):
    """Cellpose finds every cell 0.5 px low; Suite2p finds every cell 0.5 px high plus solo noise."""
    gt = np.asarray(gt, dtype=float) + offset
    cp = CentroidDetectorResult(centroids=(gt + [0.5, 0.0]).astype(np.float32),
                                scores=np.asarray(CP_SCORES[: len(gt)], dtype=np.float32))
    s2p_pts = [gt + [-0.5, 0.0]]
    s2p_scores = list(S2P_SCORES[: len(gt)])
    if spurious:
        s2p_pts.append(SPURIOUS + offset)
        s2p_scores += list(spurious_scores or SPURIOUS_SCORES)
    s2p = CentroidDetectorResult(centroids=np.concatenate(s2p_pts).astype(np.float32),
                                 scores=np.asarray(s2p_scores, dtype=np.float32))
    return FovCandidates(fov_id=fov_id, lens=lens, gt=gt, cellpose=cp, suite2p=s2p, max_distance=4.0)


def dev_set():
    return [make("d1", GRID), make("d2", GRID, offset=(5.0, 3.0))]


def test_fit_on_dev_score_on_test():
    test = [make("t1", GRID, offset=(2.0, 7.0), spurious_scores=[0.1, 0.12, 0.1, 0.12])]
    r = fit_and_score(dev_set(), test)
    assert r.model_trained is True
    assert r.n_train_rows == 2 * (6 + 10)
    assert 0.0 < r.accept_threshold <= 1.0
    assert r.dev_threshold_sweep and all("threshold" in row for row in r.dev_threshold_sweep)
    assert len(r.per_fov) == 1
    s = r.per_fov[0]
    assert (s.fov_id, s.lens, s.n_gt, s.n_pred, s.n_tp, s.max_distance) == ("t1", "GRIN", 6, 6, 6, 4.0)


def test_test_order_and_count():
    test = [make("t2", GRID[:3], spurious=False), make("t1", GRID, offset=(1.0, 1.0), lens="prism")]
    r = fit_and_score(dev_set(), test)
    assert [s.fov_id for s in r.per_fov] == ["t2", "t1"]
    assert r.per_fov[0].n_gt == 3 and r.per_fov[1].lens == "prism"


def test_threshold_does_not_depend_on_test_data():
    a = fit_and_score(dev_set(), [make("t", GRID)])
    moved_gt = make("t", GRID)
    moved_gt = FovCandidates(moved_gt.fov_id, moved_gt.lens, moved_gt.gt[:2], moved_gt.cellpose,
                             moved_gt.suite2p, moved_gt.max_distance)
    b = fit_and_score(dev_set(), [moved_gt])
    assert a.accept_threshold == b.accept_threshold
    assert a.n_train_rows == b.n_train_rows


def test_one_class_dev_is_untrained():
    dev = [make("d1", GRID, spurious=False)]
    r = fit_and_score(dev, [make("t", GRID)])
    assert r.model_trained is False and r.n_train_rows == 12


def test_empty_dev():
    with pytest.raises(ValueError):
        fit_and_score([], [make("t", GRID)])


def test_empty_test():
    r = fit_and_score(dev_set(), [])
    assert r.per_fov == () and r.model_trained is True
