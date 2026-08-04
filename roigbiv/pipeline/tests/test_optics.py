"""Tests for optics auto-adaptation (pre-foundation prior + scale derivation).

Pins the frame-size classifier bands, the GRIN/PRISM derivation golden values,
synthetic soma-scale recovery, and the plausibility cross-check.
"""
from __future__ import annotations

import numpy as np
import pytest

from roigbiv.pipeline.optics import (
    GRIN_MAX_DIM,
    PRISM_MIN_DIM,
    SomaScale,
    classify_optics_prior,
    derive_scale_params,
    measure_soma_scale,
    scale_plausible,
)


# ── classify_optics_prior ────────────────────────────────────────────────────

def test_classify_grin_512_high():
    p = classify_optics_prior((1000, 512, 512))
    assert p.profile_name == "grin"
    assert p.confidence == "high"
    assert p.max_dim == 512


def test_classify_prism_1024_high():
    p = classify_optics_prior((1000, 1024, 1024))
    assert p.profile_name == "prism"
    assert p.confidence == "high"


def test_classify_ambiguous_band_is_generic_low():
    mid = (GRIN_MAX_DIM + PRISM_MIN_DIM) // 2
    p = classify_optics_prior((1000, mid, mid))
    assert p.profile_name == "generic"
    assert p.confidence == "low"


def test_classify_bad_shape_is_generic_low():
    assert classify_optics_prior(None).profile_name == "generic"
    assert classify_optics_prior((10,)).confidence == "low"


def test_classify_2d_shape_supported():
    assert classify_optics_prior((512, 512)).profile_name == "grin"


def test_pixel_size_contradiction_demotes_confidence():
    # 512² → grin by size, but a coarse pixel size contradicts → low confidence.
    p = classify_optics_prior((1000, 512, 512), {"pixel_size_um": 2.0})
    assert p.profile_name == "grin"        # size pick stands
    assert p.confidence == "low"           # but flagged for confirmation
    assert p.pixel_size_um == 2.0


def test_pixel_size_consistent_keeps_high():
    p = classify_optics_prior((1000, 512, 512), {"pixel_size_um": 0.8})
    assert p.confidence == "high"


# ── derive_scale_params (golden) ─────────────────────────────────────────────

def test_derive_grin_scale_conservative():
    # GRIN-like measured scale (d≈12).
    s = SomaScale(diameter_med=12, diameter_p5=9, diameter_p95=16,
                  area_med=113, area_p5=70, area_p95=200, n_somata=20, ok=True)
    d = derive_scale_params(s)
    assert d["diameter"] == 12
    assert 30 <= d["min_area"] <= 90       # near tuned GRIN 80, conservative
    assert d["tile_norm_blocksize"] == 128
    assert d["spatial_pool_radius"] == 6
    assert d["roi_stamp_radius"] == 6       # tracks spatial_pool_radius's formula


def test_derive_min_area_has_hard_floor():
    # A sparse FOV of tiny somata must not open Gate 1 below the noise floor.
    s = SomaScale(diameter_med=4, diameter_p5=3, diameter_p95=6,
                  area_med=30, area_p5=30, area_p95=40, n_somata=6, ok=True)
    assert derive_scale_params(s)["min_area"] == 30


def test_derive_prism_scale_in_validated_band():
    # PRISM-like measured scale (d≈56), grounded by measure_prism_scale.py.
    s = SomaScale(diameter_med=56, diameter_p5=40, diameter_p95=70,
                  area_med=2480, area_p5=1500, area_p95=3350,
                  n_somata=20, ok=True)
    d = derive_scale_params(s)
    assert d["diameter"] == 56
    assert 900 <= d["min_area"] <= 1100    # inside validated 900..9000 band
    assert 5000 <= d["max_area"] <= 9000
    assert d["tile_norm_blocksize"] == 256
    assert d["gate1_merge_peak_min_separation"] == 28
    assert d["roi_stamp_radius"] == 28       # tracks spatial_pool_radius's formula


# ── measure_soma_scale (synthetic) ───────────────────────────────────────────

def _planted(n: int, radius: int, size: int = 256, sep: int = 50) -> np.ndarray:
    """A mean_M-like image with n Gaussian blobs of known radius on a grid."""
    img = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cols = int(np.ceil(np.sqrt(n)))
    for i in range(n):
        cy = sep + (i // cols) * sep
        cx = sep + (i % cols) * sep
        if cy >= size - sep or cx >= size - sep:
            continue
        img += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * radius ** 2))
    return img


def test_measure_recovers_planted_scale():
    img = _planted(n=12, radius=6)
    s = measure_soma_scale(img)
    assert s.ok
    assert s.n_somata >= 5
    # equivalent diameter of a ~radius-6 blob should land in a sane band
    assert 6 <= s.diameter_med <= 24


def test_measure_sparse_is_not_ok():
    img = _planted(n=2, radius=6)
    s = measure_soma_scale(img)
    assert not s.ok
    assert s.n_somata < 5


def test_measure_total_on_garbage():
    assert measure_soma_scale(np.zeros((4, 4), np.float32)).ok is False
    assert measure_soma_scale("not an image").ok is False
    assert measure_soma_scale(np.zeros((4,), np.float32)).ok is False


# ── scale_plausible ──────────────────────────────────────────────────────────

def test_plausible_rejects_not_ok():
    assert not scale_plausible(SomaScale(ok=False), None)


def test_plausible_rejects_bimodal():
    s = SomaScale(diameter_med=20, diameter_p5=5, diameter_p95=60,
                  n_somata=20, ok=True)
    assert not scale_plausible(s, None)


def test_plausible_rejects_prism_profile_with_tiny_somata():
    s = SomaScale(diameter_med=12, diameter_p5=9, diameter_p95=16,
                  n_somata=20, ok=True)
    assert not scale_plausible(s, "prism")


def test_plausible_accepts_consistent():
    s = SomaScale(diameter_med=56, diameter_p5=40, diameter_p95=70,
                  n_somata=20, ok=True)
    assert scale_plausible(s, "prism")
    assert scale_plausible(s, None)        # no profile → only sparsity/bimodal gates


# ── _apply_auto_scale (run-level integration) ────────────────────────────────

def test_apply_auto_scale_derives_and_respects_pins():
    from types import SimpleNamespace

    from roigbiv.pipeline.run import _apply_auto_scale
    from roigbiv.pipeline.types import PipelineConfig

    fov = SimpleNamespace(mean_M=_planted(n=16, radius=8), dog_map=None)
    cfg = PipelineConfig(profile="generic", auto_scale=True,
                         diameter=56, min_area=900, max_area=9000,
                         explicit_fields=("max_area",))
    _apply_auto_scale(cfg, fov)
    assert cfg.auto_adapt["scale_ok"] is True
    assert cfg.diameter != 56                       # derived (was profile default)
    assert cfg.max_area == 9000                     # pinned → untouched
    assert "diameter" in cfg.auto_adapt["applied"]
    assert "max_area" not in cfg.auto_adapt["applied"]


def test_apply_auto_scale_sparse_keeps_fallback():
    from types import SimpleNamespace

    from roigbiv.pipeline.run import _apply_auto_scale
    from roigbiv.pipeline.types import PipelineConfig

    fov = SimpleNamespace(mean_M=_planted(n=2, radius=8), dog_map=None)
    cfg = PipelineConfig(profile="generic", auto_scale=True,
                         diameter=56, min_area=900, max_area=9000)
    _apply_auto_scale(cfg, fov)
    assert cfg.auto_adapt["scale_ok"] is False
    assert cfg.diameter == 56 and cfg.min_area == 900   # untouched fallback


# ── resume fingerprint interaction ───────────────────────────────────────────

def _dummy_tif(tmp_path):
    p = tmp_path / "x.tif"
    p.write_bytes(b"not-a-real-tif-but-stat-able")
    return p


def test_fingerprint_invariant_to_autoscale_derivation(tmp_path):
    from roigbiv.pipeline.resume import compute_cfg_fingerprint
    from roigbiv.pipeline.types import PipelineConfig

    tif = _dummy_tif(tmp_path)
    cfg = PipelineConfig(profile="prism", auto_scale=True,
                         min_area=900, max_area=9000, diameter=56)
    fp_before = compute_cfg_fingerprint(cfg, tif)
    # Simulate the post-foundation derivation mutating unpinned derived fields.
    cfg.diameter, cfg.min_area, cfg.max_area = 40, 1234, 4321
    cfg.auto_adapt = {"scale_ok": True, "applied": {"diameter": 40}}
    assert compute_cfg_fingerprint(cfg, tif) == fp_before   # still resumable


def test_fingerprint_changes_with_profile(tmp_path):
    from roigbiv.pipeline.resume import compute_cfg_fingerprint
    from roigbiv.pipeline.types import PipelineConfig

    tif = _dummy_tif(tmp_path)
    grin = PipelineConfig(profile="grin")
    prism = PipelineConfig(profile="prism", auto_scale=True)
    assert compute_cfg_fingerprint(grin, tif) != compute_cfg_fingerprint(prism, tif)


def test_fingerprint_keeps_user_pinned_field(tmp_path):
    from roigbiv.pipeline.resume import compute_cfg_fingerprint
    from roigbiv.pipeline.types import PipelineConfig

    tif = _dummy_tif(tmp_path)
    a = PipelineConfig(profile="prism", auto_scale=True,
                       explicit_fields=("min_area",), min_area=900)
    b = PipelineConfig(profile="prism", auto_scale=True,
                       explicit_fields=("min_area",), min_area=1500)
    # Pinned field stays in the fingerprint → a changed explicit flag is detected.
    assert compute_cfg_fingerprint(a, tif) != compute_cfg_fingerprint(b, tif)


def test_changed_cfg_fields_and_confirm_resume_scope():
    from roigbiv.pipeline.resume import (
        _CONFIRM_RESUME_FIELDS,
        _changed_cfg_fields,
    )
    from roigbiv.pipeline.types import PipelineConfig

    base = PipelineConfig(profile="generic", fs=7.5)
    # Only the profile changed → an optics-only change (bypass allowed).
    snap = base.summary_for_log()
    confirmed = PipelineConfig(profile="prism", fs=7.5)
    changed = _changed_cfg_fields(snap, confirmed)
    assert "profile" in changed
    assert changed <= _CONFIRM_RESUME_FIELDS        # bypass would be allowed

    # An unrelated fs edit is NOT in the allowed set → bypass must be refused.
    other = PipelineConfig(profile="prism", fs=30.0)
    changed2 = _changed_cfg_fields(snap, other)
    assert "fs" in changed2
    assert not (changed2 <= _CONFIRM_RESUME_FIELDS)


def test_fingerprint_grin_autoscale_excludes_derived_fields(tmp_path):
    from roigbiv.pipeline.resume import compute_cfg_fingerprint
    from roigbiv.pipeline.types import PipelineConfig

    tif = _dummy_tif(tmp_path)
    # GRIN + auto_scale=True (the default): derived fields are excluded, so
    # different min_area values produce identical fingerprints.
    a = PipelineConfig(profile="grin", auto_scale=True, min_area=80)
    b = PipelineConfig(profile="grin", auto_scale=True, min_area=120)
    assert compute_cfg_fingerprint(a, tif) == compute_cfg_fingerprint(b, tif)

    # GRIN + auto_scale=False: derived fields stay in the fingerprint.
    c = PipelineConfig(profile="grin", auto_scale=False, min_area=80)
    d = PipelineConfig(profile="grin", auto_scale=False, min_area=120)
    assert compute_cfg_fingerprint(c, tif) != compute_cfg_fingerprint(d, tif)


# ── _estimate_diameter_px regression (refactored onto measure_soma_scale) ────

def test_estimate_diameter_px_regression():
    from roigbiv.pipeline.stage1 import _estimate_diameter_px

    d = _estimate_diameter_px(_planted(n=16, radius=8))
    assert d is not None and 8 <= d <= 40
    assert _estimate_diameter_px(_planted(n=2, radius=8)) is None


# ── pause-to-confirm decision (Phase 3) ──────────────────────────────────────

def _cfg_with_adapt(**kw):
    from roigbiv.pipeline.types import PipelineConfig
    return PipelineConfig(**kw)


def test_confirmation_none_for_explicit_profile():
    from roigbiv.pipeline.run import _optics_confirmation_decision
    cfg = _cfg_with_adapt(profile="prism", auto_adapt={})   # no prior → explicit
    assert _optics_confirmation_decision(cfg) is None


def test_confirmation_none_when_confident_and_scale_ok():
    from roigbiv.pipeline.run import _optics_confirmation_decision
    cfg = _cfg_with_adapt(
        profile="prism", auto_scale=True,
        auto_adapt={"prior": {"confidence": "high", "reasons": []}, "scale_ok": True})
    assert _optics_confirmation_decision(cfg) is None


def test_confirmation_pauses_on_low_confidence_prior():
    from roigbiv.pipeline.run import _optics_confirmation_decision
    cfg = _cfg_with_adapt(
        profile="generic", auto_scale=True,
        auto_adapt={"prior": {"confidence": "low", "reasons": ["ambiguous size"]},
                    "scale_ok": True})
    d = _optics_confirmation_decision(cfg)
    assert d is not None
    assert d["candidate_profile"] == "generic"
    assert "auto" in d["choices"]          # confirm offers the full profile list


def test_confirmation_pauses_on_scale_failure_even_if_confident():
    from roigbiv.pipeline.run import _optics_confirmation_decision
    cfg = _cfg_with_adapt(
        profile="prism", auto_scale=True,
        auto_adapt={"prior": {"confidence": "high", "reasons": []},
                    "scale_ok": False, "n_somata": 2})
    d = _optics_confirmation_decision(cfg)
    assert d is not None
    assert any("scale" in r for r in d["reasons"])


def test_confirmation_grin_pauses_on_scale_failure():
    from roigbiv.pipeline.run import _optics_confirmation_decision
    # grin + auto_scale=True: scale failure now triggers a pause (GRIN is a full
    # participant in auto_scale since _AUTO_SCALE_PROFILES includes it).
    cfg = _cfg_with_adapt(
        profile="grin", auto_scale=True,
        auto_adapt={"prior": {"confidence": "high", "reasons": []}, "scale_ok": False})
    d = _optics_confirmation_decision(cfg)
    assert d is not None
    assert any("scale" in r for r in d["reasons"])

def test_confirmation_grin_no_pause_when_auto_scale_off():
    from roigbiv.pipeline.run import _optics_confirmation_decision
    # grin + auto_scale=False: scale_failed is False regardless → no pause.
    cfg = _cfg_with_adapt(
        profile="grin", auto_scale=False,
        auto_adapt={"prior": {"confidence": "high", "reasons": []}, "scale_ok": False})
    assert _optics_confirmation_decision(cfg) is None


def test_sentinel_writer(tmp_path):
    import json

    from roigbiv.pipeline.run import _write_optics_confirmation_sentinel
    cfg = _cfg_with_adapt(profile="generic")
    decision = {"candidate_profile": "generic", "confidence": "low",
                "reasons": ["ambiguous"], "choices": ["auto", "grin", "prism"],
                "n_somata": 3, "soma_diameter_med": None}
    p = _write_optics_confirmation_sentinel(tmp_path, cfg, decision)
    obj = json.loads(p.read_text())
    assert obj["mode"] == "needs_optics_confirmation"
    assert obj["candidate_profile"] == "generic"
    assert obj["choices"] == ["auto", "grin", "prism"]


def test_build_auto_workspace_overrides_resolves_and_enables_scale(tmp_path):
    import tifffile

    from roigbiv.pipeline.run import build_auto_workspace_overrides
    # A 1024² stack → prism categoricals; auto_scale on; nothing pinned.
    tif = tmp_path / "fov_mc.tif"
    tifffile.imwrite(tif, np.zeros((3, 1024, 1024), np.uint16),
                     photometric="minisblack")   # genuine 3-page grayscale stack
    ov = build_auto_workspace_overrides([tif], {"fs": 7.5, "tau": 1.0})
    assert ov["profile"] == "prism"
    assert ov["channels"] == (0, 0)            # prism categorical from the bundle
    assert ov["auto_scale"] is True
    assert ov["assume_optics"] is False
    assert ov["explicit_fields"] == ()
    assert ov["auto_adapt"]["prior"]["confidence"] == "high"
