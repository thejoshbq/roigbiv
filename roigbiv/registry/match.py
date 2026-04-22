"""v3 FOV matcher — thin wrapper around the ROICaT adapter + calibrated logistic.

The heavy lifting (alignment, embedding, clustering) lives in
:mod:`roigbiv.registry.roicat_adapter`. This module owns the two pieces that
turn a clustering result into a registry decision:

  1. :func:`compute_fov_features` — derive the FOV-level feature vector from
     a :class:`~roigbiv.registry.roicat_adapter.ClusterResult`.
  2. :func:`match_fov` — cluster the query alongside a candidate FOV's
     sessions, derive features, run the calibrated logistic, and branch into
     one of three decisions (``auto_match`` / ``review`` / ``reject``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from roigbiv.registry.calibration import CalibrationModel, FOVFeatures
from roigbiv.registry.roicat_adapter import (
    AdapterConfig,
    ClusterResult,
    SessionInput,
    cluster_sessions,
)

# Default decision thresholds. These remain module-level (and overridable via
# :class:`roigbiv.registry.config.RegistryConfig` env vars) so downstream
# callers that previously imported them keep working.
AUTO_ACCEPT_THRESHOLD: float = 0.9
REVIEW_THRESHOLD: float = 0.5


@dataclass
class FOVMatchResult:
    """Output of :func:`match_fov`.

    Attributes
    ----------
    decision : str
        One of ``"auto_match"``, ``"review"``, ``"reject"``.
    fov_posterior : float
        Calibrated probability in ``[0, 1]``.
    features : FOVFeatures
        The exact feature vector passed to the logistic (useful for logging
        and for populating :file:`registry_match.json`).
    cluster_result : ClusterResult
        The underlying ROICaT output. The orchestrator uses this to map
        query ROIs to cluster labels and to existing global cell IDs.
    query_session_idx : int
        Index of the query session inside ``cluster_result.session_bool``.
        Always equals ``len(candidate_sessions)`` because the query is
        appended last.
    """

    decision: str
    fov_posterior: float
    features: FOVFeatures
    cluster_result: ClusterResult
    query_session_idx: int


def match_fov(
    *,
    query: SessionInput,
    candidate_sessions: list[SessionInput],
    calibration: Optional[CalibrationModel] = None,
    adapter_config: Optional[AdapterConfig] = None,
    accept_threshold: float = AUTO_ACCEPT_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
) -> FOVMatchResult:
    """Cluster ``query`` against ``candidate_sessions`` and decide.

    ``candidate_sessions`` should be all sessions belonging to **one**
    candidate FOV. To evaluate a query against multiple candidate FOVs, call
    :func:`match_fov` once per FOV and keep the highest posterior.
    """
    calibration = calibration or CalibrationModel()
    cfg = adapter_config or AdapterConfig()

    bundle = list(candidate_sessions) + [query]
    query_idx = len(candidate_sessions)

    cluster_result = cluster_sessions(bundle, cfg)
    features = compute_fov_features(cluster_result, query_session_idx=query_idx)
    posterior = calibration.p_same_fov(features)

    if posterior >= accept_threshold:
        decision = "auto_match"
    elif posterior >= review_threshold:
        decision = "review"
    else:
        decision = "reject"

    return FOVMatchResult(
        decision=decision,
        fov_posterior=float(posterior),
        features=features,
        cluster_result=cluster_result,
        query_session_idx=query_idx,
    )


def compute_fov_features(
    result: ClusterResult, *, query_session_idx: int
) -> FOVFeatures:
    """Derive a :class:`FOVFeatures` from a ROICaT :class:`ClusterResult`.

    Undefined / degenerate cases (single-session cluster, zero ROIs) return
    a zero-valued feature vector with ``mean_cluster_cohesion = 0.5`` (the
    neutral prior), so the calibrated logistic still produces a finite
    posterior (typically close to the reject floor).
    """
    labels = result.labels
    session_bool = result.session_bool
    n_sessions = session_bool.shape[1] if session_bool.ndim == 2 else 0

    alignment_quality = float(result.alignment_inlier_rate)

    if n_sessions < 2 or labels.size == 0:
        return FOVFeatures(
            n_shared_clusters=0,
            fraction_query_clustered=0.0,
            alignment_quality=alignment_quality,
            mean_cluster_cohesion=0.5,
        )

    query_col = session_bool[:, query_session_idx]
    # "Candidate" = any non-query session in the bundle.
    candidate_cols = np.delete(session_bool, query_session_idx, axis=1)
    candidate_any = candidate_cols.any(axis=1) if candidate_cols.size else np.zeros_like(query_col)

    n_query_total = int(query_col.sum())
    n_shared_clusters = 0
    n_query_in_shared = 0
    intra_means_accum: list[float] = []

    intra_by_label = _extract_intra_means(result.quality_metrics or {})

    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        members = labels == lbl
        q_hit = bool((members & query_col).any())
        c_hit = bool((members & candidate_any).any())
        if q_hit and c_hit:
            n_shared_clusters += 1
            n_query_in_shared += int((members & query_col).sum())
            if intra_by_label is not None:
                im = intra_by_label.get(int(lbl))
                if im is not None and np.isfinite(im):
                    intra_means_accum.append(float(im))

    fraction_query_clustered = (
        n_query_in_shared / n_query_total if n_query_total > 0 else 0.0
    )
    if intra_means_accum:
        mean_intra = float(np.mean(intra_means_accum))
        mean_cluster_cohesion = max(0.0, min(1.0, 1.0 - mean_intra))
    else:
        mean_cluster_cohesion = 0.5

    return FOVFeatures(
        n_shared_clusters=int(n_shared_clusters),
        fraction_query_clustered=float(fraction_query_clustered),
        alignment_quality=alignment_quality,
        mean_cluster_cohesion=float(mean_cluster_cohesion),
    )


def _extract_intra_means(quality_metrics: dict) -> Optional[dict[int, float]]:
    """Return a ``{cluster_label -> mean intra-cluster distance}`` mapping.

    ROICaT 1.5.5's ``Clusterer.compute_quality_metrics`` stores two parallel
    arrays: ``cluster_labels_unique`` and ``cluster_intra_means``. Zip them
    into a dict keyed by label. Returns ``None`` if the expected keys are
    missing or malformed (callers fall back to the 0.5 neutral cohesion).
    """
    labels_unique = quality_metrics.get("cluster_labels_unique")
    intra_means = quality_metrics.get("cluster_intra_means")
    if labels_unique is None or intra_means is None:
        return None
    try:
        labels_arr = np.asarray(labels_unique, dtype=np.float64).ravel()
        means_arr = np.asarray(intra_means, dtype=np.float64).ravel()
    except Exception:
        return None
    if labels_arr.shape != means_arr.shape:
        return None
    return {int(lbl): float(v) for lbl, v in zip(labels_arr, means_arr)}
