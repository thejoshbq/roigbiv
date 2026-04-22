"""Tests for the v3 FOV-level calibrated logistic."""
from __future__ import annotations

import json
from pathlib import Path

from roigbiv.registry.calibration import (
    CalibrationModel,
    DEFAULT_FOV_COEFS,
    FOVFeatures,
    FOVLogisticCoefs,
    fit_from_labels,
)


def _features(*, n_shared=0, frac=0.0, align=0.0, cohesion=0.5) -> FOVFeatures:
    return FOVFeatures(
        n_shared_clusters=n_shared,
        fraction_query_clustered=frac,
        alignment_quality=align,
        mean_cluster_cohesion=cohesion,
    )


def test_default_model_reject_floor_near_zero():
    m = CalibrationModel()
    # No overlap, no alignment — should be well below 0.5.
    p = m.p_same_fov(_features())
    assert 0.0 <= p < 0.3


def test_default_model_accepts_well_clustered_aligned_fov():
    m = CalibrationModel()
    # ~60 % of query ROIs clustered, alignment solid, cohesion solid.
    features = _features(n_shared=40, frac=0.6, align=0.5, cohesion=0.55)
    p = m.p_same_fov(features)
    assert p > 0.9


def test_model_monotonic_in_fraction_clustered():
    m = CalibrationModel()
    lo = m.p_same_fov(_features(frac=0.1, align=0.5, cohesion=0.5, n_shared=5))
    hi = m.p_same_fov(_features(frac=0.8, align=0.5, cohesion=0.5, n_shared=40))
    assert hi > lo


def test_model_save_load_roundtrip(tmp_path: Path):
    path = tmp_path / "calib.json"
    m = CalibrationModel(
        fov=FOVLogisticCoefs(-3.0, 0.1, 4.0, 5.0, 2.5), trained=True
    )
    m.save(path)
    loaded = CalibrationModel.load(path)
    assert loaded.trained is True
    features = _features(n_shared=10, frac=0.5, align=0.5, cohesion=0.5)
    assert abs(loaded.p_same_fov(features) - m.p_same_fov(features)) < 1e-9


def test_model_load_missing_falls_back_to_defaults(tmp_path: Path):
    loaded = CalibrationModel.load(tmp_path / "nope.json")
    assert loaded.trained is False
    # Coefficients match DEFAULT_FOV_COEFS.
    assert loaded.fov.intercept == DEFAULT_FOV_COEFS[0]


def test_fit_from_labels_two_class():
    # Trivially separable dataset — label depends on fraction_query_clustered.
    samples = [
        (_features(n_shared=1, frac=0.05, align=0.1, cohesion=0.5), 0),
        (_features(n_shared=2, frac=0.1, align=0.2, cohesion=0.5), 0),
        (_features(n_shared=30, frac=0.8, align=0.8, cohesion=0.6), 1),
        (_features(n_shared=40, frac=0.9, align=0.7, cohesion=0.7), 1),
    ]
    m = fit_from_labels(samples)
    assert m.trained is True
    # Trained model must push the positives above 0.5 and negatives below it.
    assert m.p_same_fov(samples[0][0]) < 0.5
    assert m.p_same_fov(samples[-1][0]) > 0.5


def test_fit_from_labels_single_class_returns_untrained():
    samples = [
        (_features(frac=0.1), 0),
        (_features(frac=0.2), 0),
    ]
    m = fit_from_labels(samples)
    assert m.trained is False


def test_to_from_dict_roundtrip():
    m = CalibrationModel(fov=FOVLogisticCoefs(-2.0, 0.1, 2.0, 3.0, 1.5), trained=True)
    payload = m.to_dict()
    rebuilt = CalibrationModel.from_dict(json.loads(json.dumps(payload)))
    assert rebuilt.trained is True
    assert rebuilt.fov.intercept == -2.0
    assert rebuilt.fov.coef_mean_cluster_cohesion == 1.5
