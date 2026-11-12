"""Scores-only decider: fit the consensus logistic head on dev FOVs, score test FOVs.

Reuses the bake-off's Cellpose + Suite2p consensus code as is:
``build_candidate_pool``, ``label_candidate_pool``, ``ConsensusScoreScaler``,
``scale_pool_features``, ``fit_from_labels``, ``collapse_predictions`` from
``centroid_bakeoff.consensus`` and ``_sweep_accept_thresholds`` and
``_best_f1_threshold`` from ``centroid_bakeoff.fit_consensus``. This is the
baseline a vision-model decider has to beat (gate G2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from bench.schema import DecideResult, FovScore
from centroid_bakeoff.consensus import (
    ConsensusScoreScaler, build_candidate_pool, collapse_predictions, fit_from_labels,
    label_candidate_pool, scale_pool_features,
)
from centroid_bakeoff.detector import CentroidDetectorResult
from centroid_bakeoff.fit_consensus import _best_f1_threshold, _sweep_accept_thresholds
from centroid_bakeoff.point_match import match_points


@dataclass(frozen=True)
class FovCandidates:
    """One FOV's ground truth and its two detectors' raw results."""

    fov_id: str
    lens: str
    gt: np.ndarray
    cellpose: CentroidDetectorResult
    suite2p: CentroidDetectorResult
    max_distance: float


def fit_and_score(dev: Sequence[FovCandidates], test: Sequence[FovCandidates]) -> DecideResult:
    """Fit on ``dev`` only, then score every ``test`` FOV.

    1. Per dev FOV: ``pool = build_candidate_pool(cellpose, suite2p,
       max_distance)``, ``labels = label_candidate_pool(pool, gt,
       max_distance)``.
    2. ``scaler = ConsensusScoreScaler.fit(cp_raw, s2p_raw)`` where ``cp_raw``
       and ``s2p_raw`` are the dev pools' ``raw_score`` rows whose ``origin`` is
       ``"cellpose"`` and ``"suite2p"``, concatenated over dev FOVs.
    3. ``model = fit_from_labels(list of (scaled feature, label) over every dev
       row, scaler=scaler)``, features from ``scale_pool_features(pool, scaler)``.
    4. Dev probabilities ``model.p_consensus(f)`` for every dev row;
       ``sweep = _sweep_accept_thresholds(p_dev, labels_dev)``;
       ``accept_threshold = _best_f1_threshold(sweep)``.
    5. Per test FOV: build its pool, scale with ``model.scaler``, compute
       probabilities, ``centroids, _ = collapse_predictions(pool, p,
       accept_threshold, max_distance)``, then
       ``FovScore(fov_id, lens, n_gt=len(gt), n_pred=len(centroids),
       n_tp=match_points(gt, centroids, max_distance).n_tp, max_distance)``.
       Build the ``FovScore`` directly; do not call ``bench.score``.
    6. Return ``DecideResult(accept_threshold, model.trained, n_train_rows=sum
       of dev pool sizes, per_fov=tuple in test order, dev_threshold_sweep=sweep)``.

    Test labels and test ground truth are never used before step 5's match.
    An empty ``dev`` raises ``ValueError``.
    """
    if not dev:
        raise ValueError("fit_and_score requires at least one dev FOV")

    pools = []
    for f in dev:
        pool = build_candidate_pool(f.cellpose, f.suite2p, f.max_distance)
        pools.append((pool, label_candidate_pool(pool, f.gt, f.max_distance)))

    cp_raw = np.concatenate([p.raw_score[p.origin == "cellpose"] for p, _ in pools])
    s2p_raw = np.concatenate([p.raw_score[p.origin == "suite2p"] for p, _ in pools])
    scaler = ConsensusScoreScaler.fit(cp_raw, s2p_raw)

    feats = []
    labels = []
    for pool, lab in pools:
        feats.extend(scale_pool_features(pool, scaler))
        labels.extend(int(v) for v in lab)
    model = fit_from_labels(list(zip(feats, labels)), scaler=scaler)

    p_dev = np.asarray([model.p_consensus(f) for f in feats], dtype=np.float32)
    sweep = _sweep_accept_thresholds(p_dev, np.asarray(labels, dtype=np.int32))
    accept_threshold = _best_f1_threshold(sweep)

    scores = []
    for f in test:
        pool = build_candidate_pool(f.cellpose, f.suite2p, f.max_distance)
        scaled = scale_pool_features(pool, model.scaler)
        p = (
            np.asarray([model.p_consensus(x) for x in scaled], dtype=np.float32)
            if pool.n else np.zeros(0, dtype=np.float32)
        )
        centroids, _ = collapse_predictions(pool, p, accept_threshold, f.max_distance)
        scores.append(FovScore(
            fov_id=f.fov_id, lens=f.lens, n_gt=len(f.gt), n_pred=len(centroids),
            n_tp=match_points(f.gt, centroids, f.max_distance).n_tp, max_distance=f.max_distance,
        ))

    return DecideResult(
        accept_threshold=accept_threshold,
        model_trained=model.trained,
        n_train_rows=sum(p.n for p, _ in pools),
        per_fov=tuple(scores),
        dev_threshold_sweep=sweep,
    )
