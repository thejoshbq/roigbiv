"""Tests for the Phase-2 PMD spatiotemporal denoiser (roigbiv/pipeline/pmd.py).

Covers: (1) it actually denoises (recovers a low-rank signal from noise),
(2) the returned object honors the ResidualView dense read contract, and
(3) the dense base propagates through ``with_source`` so Stage 4 inherits it.
CPU-forced for determinism.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from roigbiv.pipeline.residual import ResidualView
from roigbiv.pipeline.types import PipelineConfig
from roigbiv.pipeline.pmd import pmd_denoise_view, pmd_denoise_to_memmap


T, H, W = 200, 48, 48
RANK = 3


def _cfg() -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.use_pmd_denoise = True
    cfg.force_cpu = True            # deterministic, no GPU dependency in tests
    cfg.pmd_patch_size = 32
    cfg.pmd_patch_overlap = 8
    cfg.pmd_max_rank = 20
    cfg.pmd_rank_margin = 0.0
    return cfg


def _lowrank_signal_plus_noise(seed: int = 3):
    rng = np.random.RandomState(seed)
    # Temporally-smooth low-rank signal so lag-1 diffs are small (signal) vs the
    # i.i.d. noise the MP-edge rank selector is meant to reject.
    t = np.linspace(0, 6 * np.pi, T)
    temporal = np.stack(
        [np.sin(t), np.cos(0.5 * t), np.sin(0.25 * t + 1.0)], axis=1
    ).astype(np.float32)                                   # (T, RANK)
    spatial = rng.randn(RANK, H * W).astype(np.float32)    # (RANK, P)
    signal = (temporal @ spatial).reshape(T, H, W).astype(np.float32)
    signal *= 5.0 / signal.std()
    noise = rng.randn(T, H, W).astype(np.float32) * 1.0
    return signal, signal + noise


def test_pmd_recovers_lowrank_signal():
    signal, noisy = _lowrank_signal_plus_noise()
    view = ResidualView.from_dense(noisy)
    with tempfile.TemporaryDirectory() as td:
        out_view = pmd_denoise_view(view, Path(td), _cfg(), gpu=False)
        denoised = out_view.read_chunk(0, T)               # (T, H, W)

    mse_raw = float(np.mean((noisy - signal) ** 2))
    mse_pmd = float(np.mean((denoised - signal) ** 2))
    assert np.isfinite(denoised).all()
    # PMD should meaningfully reduce the distance to the clean signal.
    assert mse_pmd < 0.7 * mse_raw, f"mse_pmd={mse_pmd:.4f} not < 0.7*mse_raw={mse_raw:.4f}"


def test_pmd_view_dense_and_shape_preserved():
    _, noisy = _lowrank_signal_plus_noise(seed=11)
    view = ResidualView.from_dense(noisy)
    with tempfile.TemporaryDirectory() as td:
        out_view = pmd_denoise_view(view, Path(td), _cfg(), gpu=False)
        assert out_view.shape == (T, H, W)
        assert out_view._dense is not None                 # dense-backed read path
        # read primitives are self-consistent across access patterns
        chunk = out_view.read_chunk(10, 20)                # (10, H, W)
        rows = out_view.read_rows(0, 8)                    # (T, 8, W)
        np.testing.assert_allclose(chunk, np.asarray(out_view._dense[10:20]), rtol=0, atol=0)
        np.testing.assert_allclose(rows[10:20], chunk[:, 0:8, :], rtol=0, atol=0)


def test_pmd_dense_base_propagates_through_with_source():
    """The Stage-3 subtraction advances the view via with_source; the dense PMD
    base must survive so Stage 4 reads (PMD residual − sources)."""
    _, noisy = _lowrank_signal_plus_noise(seed=5)
    view = ResidualView.from_dense(noisy)
    with tempfile.TemporaryDirectory() as td:
        pmd_view = pmd_denoise_view(view, Path(td), _cfg(), gpu=False)
        base = pmd_view.read_chunk(0, T).copy()

        # One rank-1 source over a small pixel set, as the subtraction would add.
        flat_idx = np.arange(0, 50, dtype=np.int64)
        W_design = np.ones((1, flat_idx.size), dtype=np.float32)
        traces = np.ones((1, T), dtype=np.float32) * 2.0
        advanced = pmd_view.with_source(flat_idx, W_design, traces, stage_idx=3)

        assert advanced._dense is not None                 # dense carried forward
        out = advanced.read_chunk(0, T).reshape(T, H * W)
        expected = base.reshape(T, H * W).copy()
        expected[:, flat_idx] -= traces.T @ W_design       # (T, P)
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-4)


def test_pmd_overlapadd_noiseless_reproduces_input_nonaligned():
    """Decisive overlap-add correctness check with dims NOT aligned to patch/stride.

    With a noiseless low-rank signal and ample max_rank, every patch reconstructs
    near-exactly, so the normalized overlap-add must reproduce the input across
    the WHOLE field — including y-band and x-tile overlap rows. A stride/weight
    bug would show as ~2× amplification in overlap zones.
    """
    Hn, Wn, Tn, rank = 100, 70, 120, 2          # 100,70 not multiples of ps=32 / stride=24
    rng = np.random.RandomState(1)
    t = np.linspace(0, 4 * np.pi, Tn)
    temporal = np.stack([np.sin(t), np.cos(0.3 * t)], axis=1).astype(np.float32)
    spatial = rng.randn(rank, Hn * Wn).astype(np.float32)
    signal = (temporal @ spatial).reshape(Tn, Hn, Wn).astype(np.float32)

    cfg = _cfg()
    cfg.pmd_max_rank = 10
    view = ResidualView.from_dense(signal)
    with tempfile.TemporaryDirectory() as td:
        out_view = pmd_denoise_view(view, Path(td), cfg, gpu=False)
        out = out_view.read_chunk(0, Tn)
    assert np.isfinite(out).all()
    # Per-pixel relative error must be uniformly small everywhere (no overlap seams).
    denom = np.abs(signal).mean() + 1e-6
    assert np.max(np.abs(out - signal)) / denom < 0.05, (
        f"max rel err {np.max(np.abs(out - signal))/denom:.4f} — overlap-add seam/weight bug?"
    )


def test_pmd_memmap_full_coverage_no_nan():
    """Overlap-add normalization must cover every pixel (no NaN/zero-weight)."""
    _, noisy = _lowrank_signal_plus_noise(seed=2)
    view = ResidualView.from_dense(noisy)
    with tempfile.TemporaryDirectory() as td:
        mm_path = Path(td) / "pmd.dat"
        pmd_denoise_to_memmap(view, mm_path, (T, H, W), _cfg(), gpu=False)
        out = np.memmap(str(mm_path), dtype=np.float32, mode="r", shape=(T, H, W))
        assert np.isfinite(np.asarray(out)).all()
