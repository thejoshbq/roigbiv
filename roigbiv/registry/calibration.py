"""Calibrated FOV logistic over ROICaT cluster statistics (v3).

A single logistic head converts four ROICaT-derived features into a
probability that two FOVs represent the same physical field of view:

    z = intercept
      + coef_n_shared_clusters       * n_shared_clusters
      + coef_fraction_query_clustered * fraction_query_clustered
      + coef_alignment_quality       * alignment_quality
      + coef_mean_cluster_cohesion   * mean_cluster_cohesion

    p_same_fov = sigmoid(z)

Replaces the two-level (cell + FOV) calibration used in v2. The cell-level
posterior is no longer needed because ROICaT produces per-ROI cluster
assignments internally — we only aggregate those assignments into FOV-level
evidence here.

Until a labeled cross-session pair set exists, the coefficients are hand
priors (see :data:`DEFAULT_FOV_COEFS`). Target behaviour: ``p_same_fov ≈ 0.9``
when about 60 % of query ROIs join a candidate session's cluster AND
alignment quality ≥ 0.5 AND mean cluster cohesion ≥ 0.5.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# Hand priors: intercept, coef_n_shared, coef_frac_clustered, coef_align,
# coef_cohesion. Explicitly marked as priors; refit from labeled pairs via
# :func:`fit_from_labels` once a dataset exists.
DEFAULT_FOV_COEFS: tuple[float, float, float, float, float] = (-4.0, 0.05, 3.0, 4.0, 3.0)


@dataclass
class FOVLogisticCoefs:
    intercept: float
    coef_n_shared_clusters: float
    coef_fraction_query_clustered: float
    coef_alignment_quality: float
    coef_mean_cluster_cohesion: float


@dataclass
class FOVFeatures:
    """Feature vector fed into the calibrated logistic.

    Attributes
    ----------
    n_shared_clusters : int
        Number of ROICaT clusters containing at least one ROI from the query
        session AND at least one ROI from any candidate-FOV session.
    fraction_query_clustered : float
        ``n_query_in_shared / n_query_total`` — share of query ROIs that
        joined a shared cluster. Scale-invariant across FOV sizes.
    alignment_quality : float
        In ``[0, 1]``. :class:`~roigbiv.registry.roicat_adapter.ClusterResult`'s
        ``alignment_inlier_rate`` — Pearson-to-template proxy for geometric
        alignment methods, true RANSAC inlier rate for deep methods.
    mean_cluster_cohesion : float
        In ``[0, 1]``. ``1 - mean(cluster_intra_distance)`` averaged over
        shared clusters, from ROICaT's quality metrics. Higher = tighter.
    """

    n_shared_clusters: int
    fraction_query_clustered: float
    alignment_quality: float
    mean_cluster_cohesion: float


@dataclass
class CalibrationModel:
    """Persistent FOV-level logistic model.

    Loaded from a JSON file at registry startup (see
    :func:`roigbiv.registry.config.load_calibration`). Falls back to
    :data:`DEFAULT_FOV_COEFS` when no file exists — this is the normal state
    of the system until labeled cross-session pairs are collected.
    """

    fov: FOVLogisticCoefs = field(
        default_factory=lambda: FOVLogisticCoefs(*DEFAULT_FOV_COEFS)
    )
    trained: bool = False

    def p_same_fov(self, features: FOVFeatures) -> float:
        z = (
            self.fov.intercept
            + self.fov.coef_n_shared_clusters * float(features.n_shared_clusters)
            + self.fov.coef_fraction_query_clustered
            * float(features.fraction_query_clustered)
            + self.fov.coef_alignment_quality * float(features.alignment_quality)
            + self.fov.coef_mean_cluster_cohesion
            * float(features.mean_cluster_cohesion)
        )
        return float(_sigmoid(z))

    def to_dict(self) -> dict:
        return {"fov": asdict(self.fov), "trained": self.trained}

    @classmethod
    def from_dict(cls, payload: dict) -> "CalibrationModel":
        return cls(
            fov=FOVLogisticCoefs(**payload["fov"]),
            trained=bool(payload.get("trained", False)),
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Optional[Path]) -> "CalibrationModel":
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
    samples: Sequence[tuple[FOVFeatures, int]],
) -> CalibrationModel:
    """Fit the FOV logistic from labeled ``(features, label)`` pairs.

    ``label`` is 0 (different FOV) or 1 (same FOV). Returns an untrained
    :class:`CalibrationModel` if the sample set lacks both classes.
    """
    if not samples:
        return CalibrationModel()
    X = np.asarray(
        [
            (
                f.n_shared_clusters,
                f.fraction_query_clustered,
                f.alignment_quality,
                f.mean_cluster_cohesion,
            )
            for f, _ in samples
        ],
        dtype=np.float64,
    )
    y = np.asarray([int(lbl) for _, lbl in samples], dtype=np.int32)
    model = CalibrationModel()
    if len(np.unique(y)) == 2:
        from sklearn.linear_model import LogisticRegression  # lazy

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        model.fov = FOVLogisticCoefs(
            intercept=float(clf.intercept_[0]),
            coef_n_shared_clusters=float(clf.coef_[0, 0]),
            coef_fraction_query_clustered=float(clf.coef_[0, 1]),
            coef_alignment_quality=float(clf.coef_[0, 2]),
            coef_mean_cluster_cohesion=float(clf.coef_[0, 3]),
        )
        model.trained = True
    return model


def _sigmoid(z: float) -> float:
    # Numerically stable sigmoid.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)
