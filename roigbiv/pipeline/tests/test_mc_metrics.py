"""Guards for the shared motion-correction quality metrics.

Shared by ``scripts/bench_motion_correction.py`` (offline backend A/B) and
``roigbiv.ui.services.pipeline_runner`` (live per-FOV metrics panel) — both
must score the exact same way.
"""
import numpy as np
import pytest

from roigbiv.pipeline.mc_metrics import (
    banding_score,
    compute_metrics,
    contrast_rms,
    grad_anisotropy,
    grad_energy,
    lap_var,
    lap_var_smooth,
    tenengrad,
)


def _sharp_checkerboard(n: int = 64) -> np.ndarray:
    a = np.indices((n, n)).sum(axis=0) % 2
    return a.astype(np.float32)


def _flat(n: int = 64) -> np.ndarray:
    return np.full((n, n), 5.0, dtype=np.float32)


def test_compute_metrics_returns_all_headline_keys():
    m = compute_metrics(_sharp_checkerboard())
    for key in ("lap_var_smooth", "lap_var", "grad_energy", "tenengrad",
                "grad_anisotropy_xy", "banding_score", "contrast_rms"):
        assert key in m
        assert isinstance(m[key], float)


def test_sharp_image_scores_higher_lap_var_than_flat():
    # A checkerboard has real edges; a flat image has none — sharpness metrics
    # must separate them, or the metric isn't measuring what it claims to.
    sharp = lap_var(_sharp_checkerboard())
    flat = lap_var(_flat())
    assert sharp > flat


def test_flat_image_has_near_zero_contrast():
    assert contrast_rms(_flat()) == pytest.approx(0.0, abs=1e-6)


def test_banding_score_detects_horizontal_stripes():
    # Pure horizontal banding: rows alternate value, no vertical structure.
    n = 64
    banded = np.tile(np.arange(n).reshape(n, 1) % 2, (1, n)).astype(np.float32)
    isotropic = _sharp_checkerboard(n)
    assert banding_score(banded) > banding_score(isotropic)


def test_grad_anisotropy_near_one_for_isotropic_noise():
    rng = np.random.default_rng(0)
    img = rng.normal(size=(128, 128)).astype(np.float32)
    assert grad_anisotropy(img) == pytest.approx(1.0, rel=0.2)


def test_lap_var_smooth_suppresses_pixel_noise_vs_raw_lap_var():
    # Pre-smoothing should pull down the sharpness reading on pure noise (no
    # real cell-edge structure) relative to the unsmoothed Laplacian variance.
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(64, 64)).astype(np.float32)
    assert lap_var_smooth(noise) < lap_var(noise)


def test_tenengrad_and_grad_energy_score_sharp_higher_than_flat():
    # Both are nonnegative by construction (sums of squared gradients) — the
    # discriminating property that could actually regress is that they track
    # sharpness, same as lap_var.
    sharp, flat = _sharp_checkerboard(), _flat()
    assert tenengrad(sharp) > tenengrad(flat)
    assert grad_energy(sharp) > grad_energy(flat)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
