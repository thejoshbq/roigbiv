"""Tests for the Phase-5a neuropil-relative baseline-elevation QC feature
(:func:`roigbiv.pipeline.qc_features.compute_neuropil_baseline_elevation`).

5a adds a logged feature only — it must compute, be JSON-serializable, and
change no decision logic. The bulky F_neuropil ndarray must be dropped from
the serialized metadata (like trace_bandpass).
"""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline.qc_features import (
    _stable_baseline,
    compute_neuropil_baseline_elevation,
)
from roigbiv.pipeline.types import PipelineConfig, ROI


def _roi_with_traces(roi_trace: np.ndarray, neu_trace: np.ndarray) -> ROI:
    roi = ROI(
        mask=np.ones((4, 4), dtype=bool),
        label_id=1,
        source_stage=1,
        confidence="high",
        gate_outcome="accept",
    )
    roi.trace = roi_trace.astype(np.float32)
    roi.features["F_neuropil"] = neu_trace.astype(np.float32)
    return roi


def _cfg() -> PipelineConfig:
    # fs=7.5, tonic_baseline_window_s=120 -> 900-frame window; keep traces longer
    return PipelineConfig(fs=7.5)


def test_elevated_roi_is_positive():
    """A soma whose DC baseline sits above its neuropil yields positive elevation."""
    rng = np.random.default_rng(0)
    T = 2000
    neu = 100.0 + rng.normal(0, 2.0, T)          # background ~100
    roi = 150.0 + rng.normal(0, 2.0, T)          # soma DC ~150 (50% above)
    r = _roi_with_traces(roi, neu)
    compute_neuropil_baseline_elevation(r, _cfg())
    elev = r.features["neuropil_baseline_elevation"]
    assert elev > 0.2, elev
    assert r.features["roi_baseline_f0"] > r.features["neuropil_baseline_f0"]


def test_no_elevation_when_roi_matches_neuropil():
    """A cell at background level yields elevation near zero."""
    rng = np.random.default_rng(1)
    T = 2000
    base = 100.0 + rng.normal(0, 2.0, T)
    r = _roi_with_traces(base.copy(), base + rng.normal(0, 2.0, T))
    compute_neuropil_baseline_elevation(r, _cfg())
    assert abs(r.features["neuropil_baseline_elevation"]) < 0.1


def test_phasic_transients_do_not_inflate_baseline():
    """Sparse bright transients on a background-level baseline must NOT read as
    elevated — the low-percentile stable baseline ignores the spikes."""
    rng = np.random.default_rng(2)
    T = 2000
    neu = 100.0 + rng.normal(0, 2.0, T)
    roi = 100.0 + rng.normal(0, 2.0, T)
    roi[::200] += 400.0                            # rare large transients
    r = _roi_with_traces(roi, neu)
    compute_neuropil_baseline_elevation(r, _cfg())
    assert abs(r.features["neuropil_baseline_elevation"]) < 0.1


def test_missing_neuropil_is_safe():
    """No F_neuropil -> feature still present, defaults to 0.0 (no crash)."""
    roi = ROI(mask=np.ones((4, 4), bool), label_id=1, source_stage=1,
              confidence="high", gate_outcome="accept")
    roi.trace = np.full(500, 100.0, np.float32)
    compute_neuropil_baseline_elevation(roi, _cfg())
    assert roi.features["neuropil_baseline_elevation"] == 0.0


def test_feature_is_logged_and_array_dropped():
    """The scalar feature survives JSON serialization; the F_neuropil array
    is dropped (matches the trace_bandpass convention)."""
    rng = np.random.default_rng(3)
    T = 1500
    r = _roi_with_traces(140.0 + rng.normal(0, 2, T), 100.0 + rng.normal(0, 2, T))
    compute_neuropil_baseline_elevation(r, _cfg())
    blob = r.to_serializable()
    assert "neuropil_baseline_elevation" in blob["features"]
    assert "roi_baseline_f0" in blob["features"]
    assert isinstance(blob["features"]["neuropil_baseline_elevation"], float)
    assert "F_neuropil" not in blob["features"]      # bulky array dropped


def test_stable_baseline_short_trace_fallback():
    """Trace shorter than one window -> whole-trace percentile, no crash."""
    tr = np.array([1, 2, 3, 4, 100], dtype=np.float32)
    val = _stable_baseline(tr, window_frames=900, percentile=10)
    assert val == float(np.percentile(tr, 10))
