"""Consensus fusion of the Cellpose + Suite2p centroid detectors.

Phase 2's sweep results show the two detectors failing in complementary, not
redundant, ways: Cellpose is precision-heavy and recall-capped by a hard
inference-time gate (``cellprob_threshold`` controls which pixels form a mask
at all — rejected regions never become candidates); Suite2p is more
permissive on recall via ``threshold_scaling`` but noisier and has higher
localization error. This module fuses them with a calibrated logistic head
over cross-detector agreement — mirroring
``roigbiv.registry.calibration.CalibrationModel``'s fit/persist pattern
exactly — rather than a spatial merge, staying philosophically consistent
with Gate 2's "cross-validate, don't blindly blend" precedent
(``roigbiv/pipeline/gate2.py`` validates Stage 2 against Stage 1's traces as a
*feature*, never merges them).

Design summary (see the Phase 3 section of the centroid bake-off plan for the
full rationale):

- ``build_candidate_pool`` — raw union of both detectors' candidates, one row
  per candidate, each carrying its own centroid/score plus a
  ``cross_detector_distance`` feature to the nearest opposite-detector
  candidate. Never a spatial merge before scoring.
- ``label_candidate_pool`` — two-pass labeling via mutual-nearest-neighbor
  pairing, so that two detectors correctly agreeing on one real cell is never
  taught to the model as one TP + one spurious FP (a naive 1-to-1 Hungarian
  match against the raw pool would do exactly that).
- ``collapse_predictions`` — mirrors the labeling fix at output-emission time:
  an accepted agreeing pair emits one centroid, not two near-duplicates.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from centroid_bakeoff.detector import CentroidDetectorInputs, CentroidDetectorResult
from centroid_bakeoff.point_match import match_points

# Real cross-detector distances are always >= 0, so -1.0 is a safe sentinel
# for "the opposite detector found zero candidates anywhere in this FOV" — a
# case distinct from "opposite detector has candidates but none nearby"
# (which gets a real, large normalized-distance value instead). Needed
# because Cellpose's score can be legitimately negative (observed range -0.21
# to 4.59 in the Phase 2 sweep), so 0.0 can't double as an absence sentinel
# the way it safely can for Suite2p's [0,1] iscell range — hence the separate
# *_present indicator columns below.
_NO_OPPOSITE_CANDIDATES = -1.0

# Hand prior: intercept, coef_cellpose_score, coef_cellpose_present,
# coef_suite2p_score, coef_suite2p_present, coef_cross_detector_distance,
# coef_both_detected. Used only until fit_consensus.py produces a trained
# model; mildly favors agreement (both_detected) and own-detector score,
# mildly penalizes cross-detector distance.
DEFAULT_CONSENSUS_COEFS: tuple[float, float, float, float, float, float, float] = (
    -2.0, 1.5, 0.5, 1.5, 0.5, -0.5, 1.0,
)


@dataclass
class ConsensusLogisticCoefs:
    intercept: float
    coef_cellpose_score: float
    coef_cellpose_present: float
    coef_suite2p_score: float
    coef_suite2p_present: float
    coef_cross_detector_distance: float
    coef_both_detected: float


@dataclass
class ConsensusFeatures:
    """Feature vector for one candidate pool row.

    ``cellpose_score``/``suite2p_score`` are min-max scaled to ``[0, 1]``
    (see :class:`ConsensusScoreScaler`) — own score if this row originated
    from that detector, else the nearest opposite-detector candidate's score
    if within ``max_distance``, else ``0.0``. ``cross_detector_distance`` is
    the nearest opposite-detector distance normalized by the FOV's own
    ``max_distance``, clipped to ``[0, 3.0]``, or :data:`_NO_OPPOSITE_CANDIDATES`
    when the opposite detector found nothing anywhere in this FOV.
    """

    cellpose_score: float
    cellpose_present: int
    suite2p_score: float
    suite2p_present: int
    cross_detector_distance: float
    both_detected: int


@dataclass
class ConsensusScoreScaler:
    """Min-max scaler for raw detector scores, fit on training folds only.

    Held-out values are clipped to ``[0, 1]`` after applying the train-fit
    min/max rather than left to extrapolate silently — the same
    never-leak-into-held-out discipline as the LOFO split itself.
    """

    cellpose_min: float = 0.0
    cellpose_max: float = 1.0
    suite2p_min: float = 0.0
    suite2p_max: float = 1.0

    @classmethod
    def fit(cls, cellpose_scores, suite2p_scores) -> "ConsensusScoreScaler":
        def _bounds(arr) -> tuple[float, float]:
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size == 0:
                return 0.0, 1.0
            lo, hi = float(arr.min()), float(arr.max())
            if hi <= lo:
                hi = lo + 1.0
            return lo, hi

        cp_lo, cp_hi = _bounds(cellpose_scores)
        s2p_lo, s2p_hi = _bounds(suite2p_scores)
        return cls(cellpose_min=cp_lo, cellpose_max=cp_hi, suite2p_min=s2p_lo, suite2p_max=s2p_hi)

    def scale_cellpose(self, value: float) -> float:
        span = self.cellpose_max - self.cellpose_min
        return float(np.clip((value - self.cellpose_min) / span, 0.0, 1.0)) if span else 0.0

    def scale_suite2p(self, value: float) -> float:
        span = self.suite2p_max - self.suite2p_min
        return float(np.clip((value - self.suite2p_min) / span, 0.0, 1.0)) if span else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ConsensusScoreScaler":
        return cls(**payload)


@dataclass
class ConsensusModel:
    """Persistent candidate-level logistic model. Mirrors
    ``roigbiv.registry.calibration.CalibrationModel`` structurally: falls back
    to :data:`DEFAULT_CONSENSUS_COEFS` when untrained/missing, which is the
    normal state until :mod:`fit_consensus` produces a fitted artifact.
    """

    coefs: ConsensusLogisticCoefs = field(
        default_factory=lambda: ConsensusLogisticCoefs(*DEFAULT_CONSENSUS_COEFS)
    )
    scaler: ConsensusScoreScaler = field(default_factory=ConsensusScoreScaler)
    trained: bool = False

    def p_consensus(self, features: ConsensusFeatures) -> float:
        z = (
            self.coefs.intercept
            + self.coefs.coef_cellpose_score * float(features.cellpose_score)
            + self.coefs.coef_cellpose_present * float(features.cellpose_present)
            + self.coefs.coef_suite2p_score * float(features.suite2p_score)
            + self.coefs.coef_suite2p_present * float(features.suite2p_present)
            + self.coefs.coef_cross_detector_distance * float(features.cross_detector_distance)
            + self.coefs.coef_both_detected * float(features.both_detected)
        )
        return float(_sigmoid(z))

    def to_dict(self) -> dict:
        return {"coefs": asdict(self.coefs), "scaler": self.scaler.to_dict(), "trained": self.trained}

    @classmethod
    def from_dict(cls, payload: dict) -> "ConsensusModel":
        scaler_payload = payload.get("scaler")
        return cls(
            coefs=ConsensusLogisticCoefs(**payload["coefs"]),
            scaler=ConsensusScoreScaler.from_dict(scaler_payload) if scaler_payload else ConsensusScoreScaler(),
            trained=bool(payload.get("trained", False)),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Optional[Path]) -> "ConsensusModel":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except Exception:
            return cls()


def fit_from_labels(
    samples: Sequence[tuple[ConsensusFeatures, int]],
    scaler: Optional[ConsensusScoreScaler] = None,
) -> ConsensusModel:
    """Fit the consensus logistic from labeled ``(features, label)`` pairs.

    ``features`` are expected already-scaled (via *scaler*, fit by the caller
    on training folds only) — this function does not scale. ``label`` is 0
    (not a real cell) or 1 (real cell, from :func:`label_candidate_pool`).
    Returns an untrained :class:`ConsensusModel` (hand-prior coefs) if the
    sample set lacks both classes, mirroring
    ``roigbiv.registry.calibration.fit_from_labels``'s own behavior.
    """
    model = ConsensusModel(scaler=scaler or ConsensusScoreScaler())
    if not samples:
        return model
    X = np.asarray(
        [
            (
                f.cellpose_score, f.cellpose_present,
                f.suite2p_score, f.suite2p_present,
                f.cross_detector_distance, f.both_detected,
            )
            for f, _ in samples
        ],
        dtype=np.float64,
    )
    y = np.asarray([int(lbl) for _, lbl in samples], dtype=np.int32)
    if len(np.unique(y)) == 2:
        from sklearn.linear_model import LogisticRegression  # lazy

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        model.coefs = ConsensusLogisticCoefs(
            intercept=float(clf.intercept_[0]),
            coef_cellpose_score=float(clf.coef_[0, 0]),
            coef_cellpose_present=float(clf.coef_[0, 1]),
            coef_suite2p_score=float(clf.coef_[0, 2]),
            coef_suite2p_present=float(clf.coef_[0, 3]),
            coef_cross_detector_distance=float(clf.coef_[0, 4]),
            coef_both_detected=float(clf.coef_[0, 5]),
        )
        model.trained = True
    return model


def _sigmoid(z: float) -> float:
    # Numerically stable sigmoid — identical to calibration.py's helper.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


# ---------------------------------------------------------------------------
# Candidate pool
# ---------------------------------------------------------------------------

@dataclass
class CandidatePool:
    """Raw union of Cellpose + Suite2p candidates for one FOV.

    Rows stay aligned across all four attributes — each candidate keeps its
    origin detector's own raw centroid/score; agreement is only ever a
    feature (``ConsensusFeatures.cross_detector_distance``/``both_detected``),
    never a spatial merge, until :func:`collapse_predictions` runs on
    *accepted* rows at output time.
    """

    centroids: np.ndarray                 # (N, 2) float32 (y, x)
    origin: np.ndarray                    # (N,) object array: "cellpose" | "suite2p"
    raw_score: np.ndarray                 # (N,) float32, origin detector's own raw score
    features: list                        # length N, list[ConsensusFeatures]

    @property
    def n(self) -> int:
        return int(self.centroids.shape[0])


def _pool_rows_for_detector(
    own_name: str,
    own_centroids: np.ndarray,
    own_scores_raw: np.ndarray,
    other_tree: Optional[cKDTree],
    other_scores_raw: np.ndarray,
    max_distance: float,
):
    """Builds rows with RAW (unscaled) scores in ``ConsensusFeatures``.

    Scaling is deliberately deferred to :func:`scale_pool_features`, called
    separately at fit/score time with whatever scaler is in effect (a
    per-LOFO-fold train-only scaler during fitting, the persisted model's own
    scaler at inference) — pooling itself must not depend on a scaler, since
    the same raw pool is reused across every LOFO fold with a different
    scaler each time.
    """
    centroids: list[list[float]] = []
    origins: list[str] = []
    raw_scores: list[float] = []
    features: list[ConsensusFeatures] = []

    for i in range(len(own_centroids)):
        pt = own_centroids[i]
        own_raw = float(own_scores_raw[i])

        if other_tree is None:
            norm_dist, present, other_raw = _NO_OPPOSITE_CANDIDATES, 0, 0.0
        else:
            dist, j = other_tree.query(pt, k=1)
            dist = float(dist)
            present = int(dist <= max_distance)
            other_raw = float(other_scores_raw[j]) if present else 0.0
            norm_dist = min(dist / max_distance, 3.0) if max_distance > 0 else dist

        if own_name == "cellpose":
            cp_score, cp_present, s2p_score, s2p_present = own_raw, 1, other_raw, present
        else:
            cp_score, cp_present, s2p_score, s2p_present = other_raw, present, own_raw, 1

        features.append(ConsensusFeatures(
            cellpose_score=cp_score, cellpose_present=cp_present,
            suite2p_score=s2p_score, suite2p_present=s2p_present,
            cross_detector_distance=norm_dist, both_detected=present,
        ))
        centroids.append([float(pt[0]), float(pt[1])])
        origins.append(own_name)
        raw_scores.append(own_raw)

    return centroids, origins, raw_scores, features


def build_candidate_pool(
    cellpose_result: CentroidDetectorResult,
    suite2p_result: CentroidDetectorResult,
    max_distance: float,
) -> CandidatePool:
    """Union both detectors' raw candidates into one pool, one row each.

    Each row computes its nearest opposite-detector distance via a
    :class:`scipy.spatial.cKDTree` (one tree per detector), gated at
    *max_distance* for the ``*_present``/``both_detected`` indicators.
    ``ConsensusFeatures.cellpose_score``/``suite2p_score`` are RAW (unscaled)
    here — call :func:`scale_pool_features` before fitting/scoring.
    """
    cp_centroids = np.asarray(cellpose_result.centroids, dtype=np.float64).reshape(-1, 2)
    s2p_centroids = np.asarray(suite2p_result.centroids, dtype=np.float64).reshape(-1, 2)
    cp_scores = np.asarray(
        cellpose_result.scores if cellpose_result.scores is not None else np.zeros(len(cp_centroids)),
        dtype=np.float64,
    )
    s2p_scores = np.asarray(
        suite2p_result.scores if suite2p_result.scores is not None else np.zeros(len(s2p_centroids)),
        dtype=np.float64,
    )

    cp_tree = cKDTree(cp_centroids) if len(cp_centroids) else None
    s2p_tree = cKDTree(s2p_centroids) if len(s2p_centroids) else None

    cp_rows = _pool_rows_for_detector(
        "cellpose", cp_centroids, cp_scores, s2p_tree, s2p_scores, max_distance,
    )
    s2p_rows = _pool_rows_for_detector(
        "suite2p", s2p_centroids, s2p_scores, cp_tree, cp_scores, max_distance,
    )

    centroids = cp_rows[0] + s2p_rows[0]
    origins = cp_rows[1] + s2p_rows[1]
    raw_scores = cp_rows[2] + s2p_rows[2]
    features = cp_rows[3] + s2p_rows[3]

    return CandidatePool(
        centroids=np.asarray(centroids, dtype=np.float32).reshape(-1, 2),
        origin=np.asarray(origins, dtype=object),
        raw_score=np.asarray(raw_scores, dtype=np.float32),
        features=features,
    )


def scale_pool_features(pool: CandidatePool, scaler: ConsensusScoreScaler) -> list:
    """Returns a new ``list[ConsensusFeatures]`` with ``cellpose_score``/
    ``suite2p_score`` min-max scaled by *scaler* — the raw values in
    ``pool.features`` untouched. Presence flags, ``cross_detector_distance``,
    and ``both_detected`` are copied through unchanged (scaler-independent).
    """
    scaled = []
    for f in pool.features:
        cp = scaler.scale_cellpose(f.cellpose_score) if f.cellpose_present else 0.0
        s2p = scaler.scale_suite2p(f.suite2p_score) if f.suite2p_present else 0.0
        scaled.append(ConsensusFeatures(
            cellpose_score=cp, cellpose_present=f.cellpose_present,
            suite2p_score=s2p, suite2p_present=f.suite2p_present,
            cross_detector_distance=f.cross_detector_distance, both_detected=f.both_detected,
        ))
    return scaled


# ---------------------------------------------------------------------------
# Two-pass labeling — mutual-nearest-neighbor pairing
# ---------------------------------------------------------------------------

def _mutual_nn_pairs(
    cp_centroids: np.ndarray, s2p_centroids: np.ndarray, max_distance: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Mutual-nearest-neighbor pairing within *max_distance*.

    Each point is paired with at most one partner, only if they are each
    other's closest match — not greedy, for the same dense-FOV-mismatch
    reason :func:`~centroid_bakeoff.point_match.match_points` uses Hungarian
    rather than greedy claiming. Returns
    ``(pairs, solo_cellpose_idx, solo_suite2p_idx)``, all indices local to the
    input arrays.
    """
    n_cp, n_s2p = len(cp_centroids), len(s2p_centroids)
    if n_cp == 0 or n_s2p == 0:
        return [], list(range(n_cp)), list(range(n_s2p))

    s2p_tree = cKDTree(s2p_centroids)
    cp_tree = cKDTree(cp_centroids)

    cp_nearest = np.full(n_cp, -1, dtype=int)
    for i in range(n_cp):
        d, j = s2p_tree.query(cp_centroids[i], k=1)
        if d <= max_distance:
            cp_nearest[i] = int(j)

    s2p_nearest = np.full(n_s2p, -1, dtype=int)
    for j in range(n_s2p):
        d, i = cp_tree.query(s2p_centroids[j], k=1)
        if d <= max_distance:
            s2p_nearest[j] = int(i)

    pairs: list[tuple[int, int]] = []
    paired_cp: set[int] = set()
    paired_s2p: set[int] = set()
    for i in range(n_cp):
        j = cp_nearest[i]
        if j >= 0 and s2p_nearest[j] == i:
            pairs.append((i, j))
            paired_cp.add(i)
            paired_s2p.add(j)

    solo_cp = [i for i in range(n_cp) if i not in paired_cp]
    solo_s2p = [j for j in range(n_s2p) if j not in paired_s2p]
    return pairs, solo_cp, solo_s2p


def representative_sites(
    pool: CandidatePool, max_distance: float,
) -> tuple[np.ndarray, list[list[int]]]:
    """One (y, x) site per mutual-NN pair (midpoint) or solo candidate.

    Used only to decide GT correspondence — never as training feature rows.
    Returns ``(sites, contributing_rows)`` where ``contributing_rows[k]``
    lists the pool row indices (2 for a pair, 1 for a solo) that site ``k``
    represents, for propagating that site's match label back onto every raw
    row that contributed to it.
    """
    cp_idx = np.where(pool.origin == "cellpose")[0]
    s2p_idx = np.where(pool.origin == "suite2p")[0]
    cp_centroids = pool.centroids[cp_idx]
    s2p_centroids = pool.centroids[s2p_idx]

    pairs, solo_cp, solo_s2p = _mutual_nn_pairs(cp_centroids, s2p_centroids, max_distance)

    sites: list[list[float]] = []
    contributing_rows: list[list[int]] = []
    for i, j in pairs:
        mid = (cp_centroids[i] + s2p_centroids[j]) / 2.0
        sites.append([float(mid[0]), float(mid[1])])
        contributing_rows.append([int(cp_idx[i]), int(s2p_idx[j])])
    for i in solo_cp:
        sites.append([float(cp_centroids[i][0]), float(cp_centroids[i][1])])
        contributing_rows.append([int(cp_idx[i])])
    for j in solo_s2p:
        sites.append([float(s2p_centroids[j][0]), float(s2p_centroids[j][1])])
        contributing_rows.append([int(s2p_idx[j])])

    return np.asarray(sites, dtype=np.float32).reshape(-1, 2), contributing_rows


def label_candidate_pool(pool: CandidatePool, gt: np.ndarray, max_distance: float) -> np.ndarray:
    """Two-pass labeling: (1) match representative sites against GT for
    correct site-level TP/FP/FN (same denominators the single-detector sweep
    already reports); (2) propagate each site's label to every raw pool row
    that contributed to it — both members of a TP pair get label=1, both
    members of an FP-labeled pair or an unmatched solo get label=0.

    Fixes the bug a naive ``match_points(gt, pool.centroids, ...)`` would
    introduce: Hungarian's 1-to-1 constraint would make one of two agreeing
    rows the TP and mark the *other* as a spurious FP, teaching the model
    that agreement is evidence *against* validity.

    Returns ``(N_pool,) int32`` array aligned with pool rows.
    """
    sites, contributing_rows = representative_sites(pool, max_distance)
    match = match_points(gt, sites, max_distance=max_distance)

    site_label = np.zeros(len(sites), dtype=np.int32)
    for _gt_idx, site_idx, _dist in match.tp:
        site_label[site_idx] = 1

    labels = np.zeros(pool.n, dtype=np.int32)
    for site_idx, rows in enumerate(contributing_rows):
        for r in rows:
            labels[r] = site_label[site_idx]
    return labels


# ---------------------------------------------------------------------------
# Output collapsing
# ---------------------------------------------------------------------------

def collapse_predictions(
    pool: CandidatePool, p_consensus: np.ndarray, accept_threshold: float, max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter to ``p_consensus >= accept_threshold``, then merge mutual-NN
    pairs among *surviving* accepted rows into one emitted centroid each
    (midpoint; score = pair max) — reuses :func:`_mutual_nn_pairs`.

    Without this, an agreeing pair that both clear threshold would emit two
    near-duplicate centroids for one physical cell, double-counting as an FP
    at evaluation time — the same trap :func:`label_candidate_pool` avoids at
    training time, now at output-emission instead.

    Returns ``(centroids (M, 2) float32, scores (M,) float32)``.
    """
    p_consensus = np.asarray(p_consensus)
    accepted = np.where(p_consensus >= accept_threshold)[0]
    if len(accepted) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

    sub_origin = pool.origin[accepted]
    sub_centroids = pool.centroids[accepted]
    sub_scores = p_consensus[accepted]

    cp_local = np.where(sub_origin == "cellpose")[0]
    s2p_local = np.where(sub_origin == "suite2p")[0]
    cp_centroids = sub_centroids[cp_local]
    s2p_centroids = sub_centroids[s2p_local]

    pairs, solo_cp, solo_s2p = _mutual_nn_pairs(cp_centroids, s2p_centroids, max_distance)

    out_centroids: list[list[float]] = []
    out_scores: list[float] = []
    for i, j in pairs:
        mid = (cp_centroids[i] + s2p_centroids[j]) / 2.0
        out_centroids.append([float(mid[0]), float(mid[1])])
        out_scores.append(float(max(sub_scores[cp_local[i]], sub_scores[s2p_local[j]])))
    for i in solo_cp:
        out_centroids.append([float(cp_centroids[i][0]), float(cp_centroids[i][1])])
        out_scores.append(float(sub_scores[cp_local[i]]))
    for j in solo_s2p:
        out_centroids.append([float(s2p_centroids[j][0]), float(s2p_centroids[j][1])])
        out_scores.append(float(sub_scores[s2p_local[j]]))

    return (
        np.asarray(out_centroids, dtype=np.float32).reshape(-1, 2),
        np.asarray(out_scores, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Detector — composes the existing Cellpose/Suite2p detector classes
# ---------------------------------------------------------------------------

class ConsensusCentroidDetector:
    """Runs permissive Cellpose + Suite2p once each, pools, scores every row
    via the fitted model, and collapses accepted rows into final centroids.

    Permissive structural points (``cellprob_threshold=-6.0``,
    ``threshold_scaling=0.5``, ``iscell_threshold=0.0``) are deliberately
    recall-maximizing, not each detector's own best-F1 point — any stricter
    point throws away recall the fusion model can never recover.
    """

    name = "consensus"

    def __init__(
        self,
        model: ConsensusModel,
        cellpose_cfg=None,
        suite2p_work_dir: Optional[Path] = None,
        max_distance: Optional[float] = None,
        accept_threshold: float = 0.5,
        cellprob_threshold: float = -6.0,
        threshold_scaling: float = 0.5,
    ):
        self.model = model
        self.cellpose_cfg = cellpose_cfg
        self.suite2p_work_dir = suite2p_work_dir
        self.max_distance = max_distance
        self.accept_threshold = accept_threshold
        self.cellprob_threshold = cellprob_threshold
        self.threshold_scaling = threshold_scaling

    def detect(self, inputs: CentroidDetectorInputs) -> CentroidDetectorResult:
        from centroid_bakeoff.detectors.cellpose_centroid import CellposeCentroidDetector
        from centroid_bakeoff.detectors.suite2p_centroid import Suite2pCentroidDetector

        t0 = time.time()

        max_distance = self.max_distance
        if max_distance is None:
            soma_scale = inputs.soma_scale
            if soma_scale is not None and getattr(soma_scale, "ok", False):
                max_distance = soma_scale.diameter_med / 2.0
            else:
                max_distance = 6.0  # GRIN-profile fallback, matches run_centroid_bakeoff.py

        cp_det = CellposeCentroidDetector(cfg=self.cellpose_cfg, cellprob_threshold=self.cellprob_threshold)
        cp_result = cp_det.detect(inputs)

        work_dir = self.suite2p_work_dir or (Path("experiments") / "runs" / "_consensus_scratch")
        s2p_det = Suite2pCentroidDetector(
            work_dir=work_dir, iscell_threshold=0.0, lean=True,
            threshold_scaling=self.threshold_scaling,
        )
        s2p_result = s2p_det.detect(inputs)

        pool = build_candidate_pool(cp_result, s2p_result, max_distance)
        scaled_features = scale_pool_features(pool, self.model.scaler)
        p_consensus = np.asarray(
            [self.model.p_consensus(f) for f in scaled_features], dtype=np.float32,
        ) if pool.n else np.zeros(0, dtype=np.float32)
        centroids, scores = collapse_predictions(pool, p_consensus, self.accept_threshold, max_distance)
        elapsed = time.time() - t0

        return CentroidDetectorResult(
            centroids=centroids,
            scores=scores,
            meta={
                "method": self.name,
                "n_raw_cellpose": cp_result.n, "n_raw_suite2p": s2p_result.n,
                "accept_threshold": self.accept_threshold, "model_trained": self.model.trained,
                "cellprob_threshold": self.cellprob_threshold, "threshold_scaling": self.threshold_scaling,
                "n": int(len(centroids)), "runtime_s": round(elapsed, 2),
            },
        )
