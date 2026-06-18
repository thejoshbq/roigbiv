"""Tests for the Phase-5b tonic accept tier
(:func:`roigbiv.pipeline.gate_tonic_elevation.apply_tonic_accept_tier`).

The tier is OFF by default, promotes ONLY anatomical (source_stage ∈ {1,2})
tonic ROIs above an elevation threshold, never touches Stage-4 ROIs, never
touches rejects, and is strictly additive (records provenance in gate_reasons).
"""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline.gate_tonic_elevation import apply_tonic_accept_tier
from roigbiv.pipeline.types import PipelineConfig, ROI


def _roi(source_stage, activity_type, gate_outcome, confidence, elevation):
    r = ROI(
        mask=np.ones((3, 3), bool),
        label_id=1,
        source_stage=source_stage,
        confidence=confidence,
        gate_outcome=gate_outcome,
    )
    r.activity_type = activity_type
    r.features["neuropil_baseline_elevation"] = float(elevation)
    return r


def _cfg(enabled=True, thr=0.5):
    c = PipelineConfig(fs=7.5)
    c.tonic_accept_tier = enabled
    c.tonic_accept_min_elevation = thr
    return c


def test_off_by_default_is_noop():
    cfg = PipelineConfig(fs=7.5)
    assert cfg.tonic_accept_tier is False
    r = _roi(1, "tonic", "flag", "requires_review", 2.0)
    n = apply_tonic_accept_tier([r], cfg)
    assert n == 0
    assert r.gate_outcome == "flag"


def test_promotes_anatomical_tonic_above_threshold():
    r = _roi(1, "tonic", "flag", "requires_review", 1.2)
    n = apply_tonic_accept_tier([r], _cfg(thr=0.5))
    assert n == 1
    assert r.gate_outcome == "accept"
    assert r.confidence == "high"
    assert any("tonic_accept_tier" in s for s in r.gate_reasons)


def test_below_threshold_not_promoted():
    r = _roi(2, "tonic", "flag", "moderate", 0.2)
    n = apply_tonic_accept_tier([r], _cfg(thr=0.5))
    assert n == 0
    assert r.gate_outcome == "flag"


def test_stage4_tonic_never_touched():
    """Stage-4 tonics keep their requires_review contract regardless of elevation."""
    r = _roi(4, "tonic", "flag", "requires_review", 3.0)
    n = apply_tonic_accept_tier([r], _cfg(thr=0.5))
    assert n == 0
    assert r.gate_outcome == "flag"
    assert r.confidence == "requires_review"


def test_non_tonic_never_touched():
    r = _roi(1, "phasic", "flag", "moderate", 3.0)
    n = apply_tonic_accept_tier([r], _cfg(thr=0.5))
    assert n == 0
    assert r.gate_outcome == "flag"


def test_reject_never_promoted():
    r = _roi(1, "tonic", "reject", "requires_review", 3.0)
    n = apply_tonic_accept_tier([r], _cfg(thr=0.5))
    assert n == 0
    assert r.gate_outcome == "reject"


def test_already_accepted_high_conf_not_counted():
    """An ROI already auto-accepted with high confidence isn't 'promoted' (no
    review-routing change), so it doesn't inflate the promoted count."""
    r = _roi(1, "tonic", "accept", "high", 3.0)
    n = apply_tonic_accept_tier([r], _cfg(thr=0.5))
    assert n == 0
    assert r.gate_outcome == "accept"


def test_missing_elevation_defaults_zero():
    r = ROI(mask=np.ones((3, 3), bool), label_id=1, source_stage=1,
            confidence="requires_review", gate_outcome="flag")
    r.activity_type = "tonic"          # no neuropil_baseline_elevation feature
    n = apply_tonic_accept_tier([r], _cfg(thr=0.5))
    assert n == 0                       # elev defaults 0.0 < 0.5
    assert r.gate_outcome == "flag"


def test_mixed_population_counts_only_qualifying():
    rois = [
        _roi(1, "tonic", "flag", "requires_review", 1.5),   # promote (flag)
        _roi(2, "tonic", "accept", "moderate", 0.9),        # promote (review via conf)
        _roi(4, "tonic", "flag", "requires_review", 2.0),   # stage4 -> skip
        _roi(1, "phasic", "flag", "moderate", 2.0),         # non-tonic -> skip
        _roi(1, "tonic", "flag", "requires_review", 0.1),   # below thr -> skip
    ]
    # fix label_ids unique (not required by logic, but realistic)
    for i, r in enumerate(rois):
        r.label_id = i + 1
    n = apply_tonic_accept_tier(rois, _cfg(thr=0.5))
    assert n == 2
    assert rois[0].gate_outcome == "accept"
    assert rois[1].gate_outcome == "accept"
    assert rois[2].gate_outcome == "flag"
    assert rois[3].gate_outcome == "flag"
    assert rois[4].gate_outcome == "flag"
