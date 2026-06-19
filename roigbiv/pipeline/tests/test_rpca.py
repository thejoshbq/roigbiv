"""Tests for the robust low-rank + sparse Foundation background (``rpca.py``).

All tests run on CPU (``force_cpu=True``) so they need no GPU. They pin:
  - recovery of a known low-rank + sparse split,
  - the core regression: a bright *constant* (tonic) pixel survives into the
    residual under RPCA but is absorbed into L under the plain truncated SVD,
  - the post-hoc factoring matches ``_binned_svd_gpu``'s output convention
    (the svd_factors.npz contract),
  - determinism, the CPU path, and the GoDec fallback.
"""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline import rpca
from roigbiv.pipeline.foundation import _binned_svd_gpu


def _lowrank(rng, T_bin, N_pix, rank):
    return (rng.randn(T_bin, rank) @ rng.randn(rank, N_pix)).astype(np.float32)


def test_rpca_recovers_lowrank_plus_sparse():
    rng = np.random.RandomState(0)
    T_bin, N_pix, rank = 80, 64, 2
    L_true = _lowrank(rng, T_bin, N_pix, rank)
    S_true = np.zeros((T_bin, N_pix), dtype=np.float32)
    spikes = rng.choice(T_bin * N_pix, size=40, replace=False)
    S_true.flat[spikes] = rng.uniform(20, 40, size=40).astype(np.float32)
    M = L_true + S_true

    L, S = rpca.robust_lowrank_sparse(M, max_rank=4, force_cpu=True, max_iter=200)

    rel_err = np.linalg.norm(L - L_true) / np.linalg.norm(L_true)
    assert rel_err < 0.1, f"L recovery rel err too high: {rel_err}"
    # The recovered sparse support should mostly overlap the planted spikes.
    recovered = set(np.flatnonzero(np.abs(S).ravel() > 1.0).tolist())
    planted = set(spikes.tolist())
    overlap = len(recovered & planted) / len(planted)
    assert overlap > 0.7, f"sparse support overlap too low: {overlap}"


def test_rpca_retains_sparse_source_that_svd_absorbs():
    """The proximate root cause, as a unit test.

    A spatially-sparse, temporally-distinct bright source (a high-amplitude
    transient at one pixel) dominates the energy, so the plain top-k SVD pulls it
    into a leading component — it vanishes from the residual. RPCA peels it into
    the sparse term instead, so it survives in the residual ``M − L``.

    (Note: plain RPCA does *not* solve the *tonic*/constant-in-time regime — a
    constant single pixel is genuinely rank-1. That is the sub-step-2 TV-prior
    job; sub-step 1 fixes absorption of sparse, temporally-distinct sources and
    restores residual structure.)
    """
    rng = np.random.RandomState(1)
    T_bin, N_pix = 80, 64
    background = _lowrank(rng, T_bin, N_pix, 2) * 0.5      # weak slow background
    p_star = 37
    source = np.zeros(T_bin, dtype=np.float32)
    source[[10, 11, 30, 55, 56]] = 200.0                  # sparse, high transients
    M = background.copy()
    M[:, p_star] += source

    # Plain truncated SVD path (k components).
    k = 3
    U, S, V = _binned_svd_gpu(M, n_svd=k, force_cpu=True)
    US_k = (U[:, :k] * S[:k][None, :]).astype(np.float32)
    L_svd = (V[:, :k] @ US_k.T)                            # (T_bin, N_pix)
    resid_svd_p = (M - L_svd)[:, p_star]

    # RPCA path.
    L_rpca, _ = rpca.robust_lowrank_sparse(M, max_rank=k, force_cpu=True, max_iter=200)
    resid_rpca_p = (M - L_rpca)[:, p_star]

    src_norm = np.linalg.norm(source)
    retained_svd = np.linalg.norm(resid_svd_p) / src_norm
    retained_rpca = np.linalg.norm(resid_rpca_p) / src_norm

    # SVD absorbs most of the source; RPCA keeps it in the residual.
    assert retained_svd < 0.4, f"expected SVD to absorb source, retained {retained_svd}"
    assert retained_rpca > 0.7, f"expected RPCA to retain source, retained {retained_rpca}"
    assert retained_rpca > 2 * retained_svd


def test_factor_from_robust_L_matches_svd_format():
    rng = np.random.RandomState(2)
    T_bin, N_pix = 60, 48
    L_bin = _lowrank(rng, T_bin, N_pix, 3)
    n_svd = 8

    U, S, V = rpca._factor_from_robust_L(L_bin, n_svd, force_cpu=True)
    Uo, So, Vo = _binned_svd_gpu(L_bin, n_svd, force_cpu=True)

    # Same orientation/shape/dtype as the SVD path.
    assert U.shape == Uo.shape == (N_pix, n_svd)
    assert V.shape == Vo.shape == (T_bin, n_svd)
    assert S.shape == So.shape == (n_svd,)
    assert U.dtype == V.dtype == S.dtype == np.float32
    # Reconstruction matches L_bin (L ≈ V diag(S) U.T — pixel-indexed U).
    recon = V @ (U * S[None, :]).T
    rel = np.linalg.norm(recon - L_bin) / np.linalg.norm(L_bin)
    assert rel < 1e-3, f"factor reconstruction rel err {rel}"


def test_rpca_determinism():
    rng = np.random.RandomState(3)
    M = _lowrank(rng, 50, 40, 2) + 0.1 * rng.randn(50, 40).astype(np.float32)
    L1, S1 = rpca.robust_lowrank_sparse(M, max_rank=4, force_cpu=True, seed=7)
    L2, S2 = rpca.robust_lowrank_sparse(M, max_rank=4, force_cpu=True, seed=7)
    assert np.array_equal(L1, L2)
    assert np.array_equal(S1, S2)


def test_rpca_cpu_fallback_runs():
    rng = np.random.RandomState(4)
    M = _lowrank(rng, 40, 32, 2)
    L, S = rpca.robust_lowrank_sparse(M, max_rank=3, force_cpu=True)
    assert L.shape == M.shape and S.shape == M.shape
    assert np.isfinite(L).all() and np.isfinite(S).all()


def test_rpca_godec_fallback():
    rng = np.random.RandomState(5)
    M = _lowrank(rng, 60, 50, 2) + 0.05 * rng.randn(60, 50).astype(np.float32)
    L, S = rpca.robust_lowrank_sparse(
        M, max_rank=3, force_cpu=True, method="godec", max_iter=50)
    assert L.shape == M.shape and S.shape == M.shape
    rel = np.linalg.norm(M - L - S) / np.linalg.norm(M)
    assert rel < 0.5, f"godec residual too large: {rel}"


# ── Adaptive GPU-memory binning ─────────────────────────────────────────────

def test_estimate_bin_frames_none_free_bytes_is_passthrough():
    """CPU run / no probe (free_bytes=None) leaves the requested target alone."""
    assert rpca.estimate_rpca_bin_frames(1_048_576, 2000, None) == 2000
    assert rpca.estimate_rpca_bin_frames(1_048_576, 50, None) == 50


def test_estimate_bin_frames_caps_large_fov():
    """A 1024×1024 FOV on an 8 GB-free card is capped below the request but
    above the floor (exercises the memory ceiling, not the MIN clamp)."""
    n_pix = 1024 * 1024
    free = 8 * 1024**3  # 8 GB free
    target = rpca.estimate_rpca_bin_frames(n_pix, 2000, free)
    # mem ceiling = 0.6 * free / (6 copies * N_pix * 4 bytes)
    expected = int(0.6 * free // (rpca._IALM_LIVE_COPIES * n_pix * 4))
    assert target == expected
    assert rpca.MIN_RPCA_FRAMES < target < 2000
    # The sized matrix must fit the live-copy budget within the fraction.
    assert rpca._IALM_LIVE_COPIES * target * n_pix * 4 <= 0.6 * free


def test_estimate_bin_frames_small_fov_unbound():
    """A small FOV with plenty of memory keeps the requested target (cap inert)."""
    n_pix = 256 * 256
    free = 12 * 1024**3
    assert rpca.estimate_rpca_bin_frames(n_pix, 2000, free) == 2000


def test_estimate_bin_frames_respects_floor():
    """A tiny memory budget still floors at MIN_RPCA_FRAMES, not below."""
    n_pix = 2048 * 2048
    free = 256 * 1024**2  # 256 MB — way too small
    target = rpca.estimate_rpca_bin_frames(n_pix, 2000, free)
    assert target == rpca.MIN_RPCA_FRAMES


def test_estimate_bin_frames_never_exceeds_request():
    """The cap is a min() with the request — more memory never inflates it."""
    n_pix = 256 * 256
    free = 64 * 1024**3
    assert rpca.estimate_rpca_bin_frames(n_pix, 500, free) == 500


def test_free_gpu_bytes_force_cpu_is_none():
    """force_cpu short-circuits the probe so no GPU ceiling is applied."""
    assert rpca.free_gpu_bytes(force_cpu=True) is None
