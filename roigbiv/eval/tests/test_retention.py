"""Tests for residual-retention diagnostics (``roigbiv/eval/retention.py``).

Pins the §3.3 absorption metric used to gate Phase A and drive Phase-K
k-selection: a soma whose brightness lives in the residual ``S`` scores
``r_S → 1``; one absorbed into the background ``L`` scores ``r_S → 0``.
"""
from __future__ import annotations

import numpy as np

from roigbiv.eval.retention import (
    count_vcorr_maxima,
    mask_retention,
    retention_summary,
)


def _disk_mask(H, W, cy, cx, r):
    yy, xx = np.ogrid[:H, :W]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def test_fully_retained_soma_scores_one():
    """All soma brightness in S, none in L → r_S ≈ 1 (k well-calibrated)."""
    H = W = 64
    mask = _disk_mask(H, W, 32, 32, 6)
    mean_S = np.zeros((H, W), np.float32)
    mean_L = np.zeros((H, W), np.float32)
    mean_S[mask] = 10.0
    assert abs(mask_retention(mean_S, mean_L, mask) - 1.0) < 1e-6


def test_fully_absorbed_soma_scores_zero():
    """All soma brightness in L, none in S → r_S ≈ 0 (k-too-high, PRISM k=30)."""
    H = W = 64
    mask = _disk_mask(H, W, 32, 32, 6)
    mean_S = np.zeros((H, W), np.float32)
    mean_L = np.zeros((H, W), np.float32)
    mean_L[mask] = 10.0
    assert abs(mask_retention(mean_S, mean_L, mask) - 0.0) < 1e-6


def test_half_retained_soma_scores_half():
    H = W = 64
    mask = _disk_mask(H, W, 32, 32, 6)
    mean_S = np.zeros((H, W), np.float32)
    mean_L = np.zeros((H, W), np.float32)
    mean_S[mask] = 4.0
    mean_L[mask] = 4.0
    assert abs(mask_retention(mean_S, mean_L, mask) - 0.5) < 1e-6


def test_empty_mask_is_nan():
    H = W = 16
    z = np.zeros((H, W), np.float32)
    assert np.isnan(mask_retention(z, z, np.zeros((H, W), bool)))


def test_zero_brightness_denominator_is_nan():
    """mean_S + mean_L ≈ 0 over the mask (no soma there) → NaN, not div-by-zero."""
    H = W = 16
    mask = _disk_mask(H, W, 8, 8, 3)
    z = np.zeros((H, W), np.float32)
    assert np.isnan(mask_retention(z, z, mask))


def test_retention_summary_frac_pass_and_stats():
    """Three somata: one retained, one absorbed, one half → frac_pass(τ=0.5)=2/3."""
    H = W = 96
    m1 = _disk_mask(H, W, 20, 20, 6)   # retained
    m2 = _disk_mask(H, W, 20, 70, 6)   # absorbed
    m3 = _disk_mask(H, W, 70, 45, 6)   # half
    mean_S = np.zeros((H, W), np.float32)
    mean_L = np.zeros((H, W), np.float32)
    mean_S[m1] = 10.0
    mean_L[m2] = 10.0
    mean_S[m3] = 5.0
    mean_L[m3] = 5.0
    out = retention_summary(mean_S, mean_L, [m1, m2, m3], tau_retain=0.5)
    assert out["n_scored"] == 3
    assert abs(out["frac_pass"] - 2 / 3) < 1e-6      # m1 (1.0) and m3 (0.5) pass
    assert abs(out["r_S_median"] - 0.5) < 1e-6
    assert abs(out["r_S_min"] - 0.0) < 1e-6


def test_count_vcorr_maxima_finds_planted_hotspots():
    """Three well-separated smooth bumps → 3 local maxima.

    Uses Gaussian bumps (not flat disks) since real Vcorr soma hotspots are
    smooth — a flat plateau would expose multiple equal maxima, which is a
    fixture artifact, not the metric's behavior.
    """
    from scipy.ndimage import gaussian_filter

    H = W = 128
    pts = np.zeros((H, W), np.float32)
    for cy, cx in [(30, 30), (30, 90), (90, 60)]:
        pts[cy, cx] = 1.0
    vcorr = gaussian_filter(pts, sigma=3.0)
    vcorr = vcorr / vcorr.max() * 0.8 + 0.1   # scale into a plausible Vcorr range
    n = count_vcorr_maxima(vcorr, min_distance=8, threshold_rel=0.3)
    assert n == 3, f"expected 3 hotspots, got {n}"
