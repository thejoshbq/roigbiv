"""Unit tests for consensus.py — hand-built cases, no detectors/GPU/Suite2p involved."""
from __future__ import annotations

import numpy as np
import pytest

from centroid_bakeoff.consensus import (
    _NO_OPPOSITE_CANDIDATES,
    _mutual_nn_pairs,
    build_candidate_pool,
    collapse_predictions,
    ConsensusFeatures,
    ConsensusModel,
    ConsensusScoreScaler,
    fit_from_labels,
    label_candidate_pool,
    representative_sites,
    scale_pool_features,
)
from centroid_bakeoff.detector import CentroidDetectorResult


def _result(centroids, scores):
    return CentroidDetectorResult(
        centroids=np.asarray(centroids, dtype=np.float32).reshape(-1, 2),
        scores=np.asarray(scores, dtype=np.float32),
        meta={"n": len(centroids)},
    )


# ---------------------------------------------------------------------------
# _mutual_nn_pairs
# ---------------------------------------------------------------------------

def test_mutual_nn_pairs_one_pair_two_solos():
    # cp[0] <-> s2p[0] are mutual nearest neighbors (genuine pair).
    # cp[1] and cp[2] have no s2p partner within range -> solos.
    # s2p[1] and s2p[2] have no cp partner within range -> solos.
    cp = np.array([[0, 0], [50, 50], [80, 80]], dtype=np.float64)
    s2p = np.array([[0.5, 0.5], [60, 60], [90, 90]], dtype=np.float64)

    pairs, solo_cp, solo_s2p = _mutual_nn_pairs(cp, s2p, max_distance=2.0)

    assert pairs == [(0, 0)]
    assert solo_cp == [1, 2]
    assert solo_s2p == [1, 2]


def test_mutual_nn_pairs_not_mutual_stays_solo():
    # cp[0]'s nearest s2p is s2p[0], but s2p[0]'s nearest cp is cp[1]
    # (closer) -> not mutual, both stay solo w.r.t. each other.
    cp = np.array([[0, 0], [0, 1]], dtype=np.float64)
    s2p = np.array([[0, 0.6]], dtype=np.float64)

    pairs, solo_cp, solo_s2p = _mutual_nn_pairs(cp, s2p, max_distance=5.0)

    assert pairs == [(1, 0)]
    assert solo_cp == [0]
    assert solo_s2p == []


def test_mutual_nn_pairs_empty_inputs():
    cp = np.zeros((0, 2))
    s2p = np.array([[0, 0], [1, 1]], dtype=np.float64)
    pairs, solo_cp, solo_s2p = _mutual_nn_pairs(cp, s2p, max_distance=1.0)
    assert pairs == []
    assert solo_cp == []
    assert solo_s2p == [0, 1]


# ---------------------------------------------------------------------------
# build_candidate_pool — feature/sentinel correctness
# ---------------------------------------------------------------------------

def test_pool_no_opposite_candidates_sentinel():
    # Suite2p found nothing anywhere in this FOV.
    cp_result = _result([[0, 0], [10, 10]], [1.0, 2.0])
    s2p_result = _result([], [])

    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)

    assert pool.n == 2
    for f in pool.features:
        assert f.cross_detector_distance == _NO_OPPOSITE_CANDIDATES
        assert f.suite2p_present == 0
        assert f.both_detected == 0


def test_pool_far_but_present_gets_real_distance():
    # Suite2p has a candidate, but it's far from the Cellpose candidate ->
    # real (large) normalized distance, not the sentinel.
    cp_result = _result([[0, 0]], [1.0])
    s2p_result = _result([[100, 100]], [0.9])

    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)

    cp_row = pool.features[pool.origin.tolist().index("cellpose")]
    assert cp_row.cross_detector_distance != _NO_OPPOSITE_CANDIDATES
    assert cp_row.cross_detector_distance > 0
    assert cp_row.suite2p_present == 0
    assert cp_row.both_detected == 0


def test_pool_agreeing_candidates_marked_both_detected():
    cp_result = _result([[0, 0]], [1.0])
    s2p_result = _result([[0.5, 0.5]], [0.8])

    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)

    assert pool.n == 2
    for f in pool.features:
        assert f.both_detected == 1
        assert f.cellpose_present == 1
        assert f.suite2p_present == 1


def test_pool_features_are_raw_unscaled():
    # Cellpose's mean-cellprob score is legitimately negative/unbounded --
    # pooling must not clip/scale it; that's scale_pool_features's job.
    cp_result = _result([[0, 0]], [3.7])
    s2p_result = _result([], [])

    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)
    assert pool.features[0].cellpose_score == pytest.approx(3.7)


def test_scale_pool_features_scales_present_scores_only():
    cp_result = _result([[0, 0]], [4.0])
    s2p_result = _result([[0.5, 0.5]], [0.8])
    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)

    scaler = ConsensusScoreScaler(cellpose_min=0.0, cellpose_max=8.0, suite2p_min=0.0, suite2p_max=1.0)
    scaled = scale_pool_features(pool, scaler)

    cp_row = scaled[pool.origin.tolist().index("cellpose")]
    assert cp_row.cellpose_score == pytest.approx(0.5)   # 4.0 scaled into [0,8] -> 0.5
    assert cp_row.suite2p_score == pytest.approx(0.8)    # opposite score also scaled

    # absence sentinel case: score stays 0.0 (not scaled) when not present.
    cp_only_result = _result([[10, 10]], [4.0])
    empty_result = _result([], [])
    pool2 = build_candidate_pool(cp_only_result, empty_result, max_distance=3.0)
    scaled2 = scale_pool_features(pool2, scaler)
    assert scaled2[0].suite2p_present == 0
    assert scaled2[0].suite2p_score == 0.0


# ---------------------------------------------------------------------------
# label_candidate_pool — the agreement-penalty fix
# ---------------------------------------------------------------------------

def test_label_candidate_pool_agreeing_pair_both_labeled_positive():
    # One real GT cell; both detectors correctly find it (agreeing pair).
    # A naive 1-to-1 Hungarian match on the raw pool would label one row TP
    # and the other a spurious FP -- this must label BOTH rows 1.
    gt = np.array([[10, 10]], dtype=np.float32)
    cp_result = _result([[10, 10]], [1.0])
    s2p_result = _result([[10.2, 10.2]], [0.9])

    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)
    labels = label_candidate_pool(pool, gt, max_distance=3.0)

    assert pool.n == 2
    assert list(labels) == [1, 1]


def test_label_candidate_pool_unmatched_solo_gets_zero():
    gt = np.zeros((0, 2), dtype=np.float32)
    cp_result = _result([[10, 10]], [1.0])
    s2p_result = _result([], [])

    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)
    labels = label_candidate_pool(pool, gt, max_distance=3.0)

    assert list(labels) == [0]


def test_label_candidate_pool_true_solo_matches_gt():
    # Cellpose finds a real cell alone (Suite2p missed it) -> solo site,
    # still matches GT -> label 1.
    gt = np.array([[10, 10]], dtype=np.float32)
    cp_result = _result([[10, 10]], [1.0])
    s2p_result = _result([], [])

    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)
    labels = label_candidate_pool(pool, gt, max_distance=3.0)

    assert list(labels) == [1]


def test_representative_sites_pair_is_midpoint():
    cp_result = _result([[0, 0]], [1.0])
    s2p_result = _result([[2, 0]], [0.9])
    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)

    sites, contributing_rows = representative_sites(pool, max_distance=3.0)

    assert sites.shape == (1, 2)
    np.testing.assert_allclose(sites[0], [1.0, 0.0])
    assert len(contributing_rows[0]) == 2


# ---------------------------------------------------------------------------
# fit_from_labels — mirrors calibration.py's own test convention
# ---------------------------------------------------------------------------

def test_fit_from_labels_single_class_stays_untrained():
    samples = [
        (ConsensusFeatures(0.9, 1, 0.8, 1, 0.0, 1), 1),
        (ConsensusFeatures(0.7, 1, 0.6, 1, 0.1, 1), 1),
    ]
    model = fit_from_labels(samples)
    assert model.trained is False


def test_fit_from_labels_empty_stays_untrained():
    model = fit_from_labels([])
    assert model.trained is False


def test_fit_from_labels_both_classes_trains():
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(20):
        samples.append((ConsensusFeatures(0.9, 1, 0.8, 1, 0.0, 1), 1))
        samples.append((ConsensusFeatures(0.05, 1, 0.02, 0, 2.5, 0), 0))
    model = fit_from_labels(samples)
    assert model.trained is True
    # A confidently-agreeing candidate should score higher than a
    # confidently-lone low-score one, on the just-fitted model.
    p_pos = model.p_consensus(ConsensusFeatures(0.9, 1, 0.8, 1, 0.0, 1))
    p_neg = model.p_consensus(ConsensusFeatures(0.05, 1, 0.02, 0, 2.5, 0))
    assert p_pos > p_neg


# ---------------------------------------------------------------------------
# ConsensusModel / ConsensusScoreScaler persistence
# ---------------------------------------------------------------------------

def test_consensus_model_save_load_roundtrip(tmp_path):
    model = fit_from_labels(
        [(ConsensusFeatures(0.9, 1, 0.8, 1, 0.0, 1), 1)] * 5
        + [(ConsensusFeatures(0.1, 0, 0.1, 1, 2.0, 0), 0)] * 5,
        scaler=ConsensusScoreScaler(cellpose_min=-1.0, cellpose_max=4.0),
    )
    path = tmp_path / "consensus_model.json"
    model.save(path)

    loaded = ConsensusModel.load(path)
    assert loaded.trained == model.trained
    assert loaded.coefs == model.coefs
    assert loaded.scaler.cellpose_min == -1.0
    assert loaded.scaler.cellpose_max == 4.0


def test_consensus_model_load_missing_file_falls_back():
    model = ConsensusModel.load(None)
    assert model.trained is False

    from pathlib import Path
    model2 = ConsensusModel.load(Path("/nonexistent/consensus_model.json"))
    assert model2.trained is False


def test_scaler_fit_clips_held_out_extrapolation():
    scaler = ConsensusScoreScaler.fit([0.0, 1.0, 2.0], [0.1, 0.5])
    assert scaler.scale_cellpose(-5.0) == 0.0    # below train min -> clipped
    assert scaler.scale_cellpose(50.0) == 1.0    # above train max -> clipped
    assert scaler.scale_cellpose(1.0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# collapse_predictions
# ---------------------------------------------------------------------------

def test_collapse_predictions_merges_accepted_agreeing_pair():
    cp_result = _result([[0, 0]], [1.0])
    s2p_result = _result([[0.4, 0.4]], [0.9])
    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)

    p_consensus = np.array([0.9, 0.8], dtype=np.float32)  # both accepted
    centroids, scores = collapse_predictions(pool, p_consensus, accept_threshold=0.5, max_distance=3.0)

    assert centroids.shape == (1, 2)
    np.testing.assert_allclose(centroids[0], [0.2, 0.2])
    assert scores[0] == pytest.approx(0.9)  # max of the pair


def test_collapse_predictions_leaves_two_solos_as_two():
    cp_result = _result([[0, 0]], [1.0])
    s2p_result = _result([[100, 100]], [0.9])  # far apart -> not a pair
    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)

    p_consensus = np.array([0.9, 0.8], dtype=np.float32)
    centroids, scores = collapse_predictions(pool, p_consensus, accept_threshold=0.5, max_distance=3.0)

    assert centroids.shape == (2, 2)


def test_collapse_predictions_below_threshold_dropped():
    cp_result = _result([[0, 0]], [1.0])
    s2p_result = _result([], [])
    pool = build_candidate_pool(cp_result, s2p_result, max_distance=3.0)

    p_consensus = np.array([0.1], dtype=np.float32)
    centroids, scores = collapse_predictions(pool, p_consensus, accept_threshold=0.5, max_distance=3.0)

    assert centroids.shape == (0, 2)
