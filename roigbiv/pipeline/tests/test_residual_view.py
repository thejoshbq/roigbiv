"""Numerical-equivalence tests for the lazy :class:`ResidualView`.

Proves the on-demand reconstruction (``S = M − L − Σsources``) matches a densely
materialized oracle computed with the same arithmetic the old streaming write
used (``foundation.compute_background_separation`` + ``subtraction.subtract_sources``).
This is the safety net for the virtual-residual refactor: if these pass, every
downstream consumer reading the view sees the same values the old ``.dat`` held
(within float32 tolerance — matmul reduction order differs slightly).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from roigbiv.pipeline.residual import ResidualView, SourceLayer

T, LY, LX = 120, 16, 20
N_PIX = LY * LX
N_SVD = 8
K = 4

RTOL, ATOL = 1e-3, 1e-4


def _make_substrate(td: Path):
    rng = np.random.RandomState(7)
    movie = rng.randint(-50, 800, size=(T, LY, LX)).astype(np.int16)
    data_bin = td / "data.bin"
    movie.tofile(str(data_bin))

    U = rng.randn(N_PIX, N_SVD).astype(np.float32)
    S = (rng.rand(N_SVD).astype(np.float32) + 0.2) * 40.0
    V_bin = rng.randn(T, N_SVD).astype(np.float32)
    bin_size = 1
    svd_path = td / "svd_factors.npz"
    np.savez(str(svd_path), U=U, S=S, V_bin=V_bin,
             bin_size=np.int32(bin_size), T=np.int32(T))
    return data_bin, svd_path, U, S, V_bin, bin_size, movie


def _oracle_S0(movie, U, S):
    """Dense S₀ = M − L, L = US_k @ V_k_full.T (bin_size=1 ⇒ V_full = V_bin)."""
    US_k = (U[:, :K] * S[:K][None, :]).astype(np.float32)   # (N_pix, K)
    V_k = _V.astype(np.float32)[:, :K]                       # (T, K)
    L = (V_k @ US_k.T)                                       # (T, N_pix)
    M = movie.reshape(T, N_PIX).astype(np.float32)
    return (M - L).reshape(T, LY, LX)


def test_read_chunk_matches_oracle():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_bin, svd_path, U, S, V_bin, bin_size, movie = _make_substrate(td)
        global _V
        _V = V_bin
        S0 = _oracle_S0(movie, U, S)

        view = ResidualView.from_factors(
            data_bin, U, S, V_bin, bin_size, (T, LY, LX), K,
        )
        # temporal chunks
        for t0, t1 in [(0, 30), (30, 90), (90, T)]:
            got = view.read_chunk(t0, t1)
            assert np.allclose(got, S0[t0:t1], rtol=RTOL, atol=ATOL), \
                f"read_chunk mismatch at [{t0}:{t1}]"
        # spatial bands → (T, h, Lx)
        for y0, y1 in [(0, 5), (5, 16)]:
            got = view.read_rows(y0, y1)
            assert np.allclose(got, S0[:, y0:y1, :], rtol=RTOL, atol=ATOL), \
                f"read_rows mismatch at rows [{y0}:{y1}]"
        # arbitrary pixels → (T, P)
        ys = np.array([1, 7, 7, 12])
        xs = np.array([2, 3, 19, 0])
        got = view.read_pixels(ys, xs)
        assert np.allclose(got, S0[:, ys, xs], rtol=RTOL, atol=ATOL), \
            "read_pixels mismatch"


def _oracle_subtract(S_dense, flat_idx, W_design, traces):
    """Dense S − Σ_i w_i·c_i over union pixels — mirrors subtract_sources."""
    out = S_dense.reshape(T, N_PIX).copy()
    out[:, flat_idx] -= (traces.T @ W_design)   # (T,N)x(N,P) → (T,P)
    return out.reshape(T, LY, LX)


def test_with_source_layer_matches_oracle():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data_bin, svd_path, U, S, V_bin, bin_size, movie = _make_substrate(td)
        global _V
        _V = V_bin
        S0 = _oracle_S0(movie, U, S)

        view0 = ResidualView.from_factors(
            data_bin, U, S, V_bin, bin_size, (T, LY, LX), K,
        )

        rng = np.random.RandomState(11)
        # Two stacked source layers (stages 1 and 2).
        S_oracle = S0
        view = view0
        for stage in (1, 2):
            P = 9
            flat_idx = np.sort(rng.choice(N_PIX, size=P, replace=False)).astype(np.int64)
            N = 3
            W_design = rng.rand(N, P).astype(np.float32)
            traces = rng.randn(N, T).astype(np.float32)
            S_oracle = _oracle_subtract(S_oracle, flat_idx, W_design, traces)
            view = view.with_source(flat_idx, W_design, traces, stage_idx=stage)

        # Equivalence across all three read primitives after 2 layers.
        assert np.allclose(view.read_chunk(0, T), S_oracle, rtol=RTOL, atol=ATOL), \
            "stacked read_chunk mismatch"
        assert np.allclose(view.read_rows(0, LY), S_oracle, rtol=RTOL, atol=ATOL), \
            "stacked read_rows mismatch"
        ys = np.array([0, 5, 10, 15])
        xs = np.array([0, 9, 4, 19])
        assert np.allclose(view.read_pixels(ys, xs), S_oracle[:, ys, xs],
                           rtol=RTOL, atol=ATOL), "stacked read_pixels mismatch"


def test_source_layer_roundtrip(tmp_path):
    layer = SourceLayer(
        flat_idx=np.array([1, 4, 9], dtype=np.int64),
        W_design=np.arange(6, dtype=np.float32).reshape(2, 3),
        traces=np.ones((2, T), dtype=np.float32),
        stage_idx=2,
    )
    path = tmp_path / "layer.sources.npz"
    layer.save(path)
    back = SourceLayer.load(path)
    assert np.array_equal(back.flat_idx, layer.flat_idx)
    assert np.array_equal(back.W_design, layer.W_design)
    assert np.array_equal(back.traces, layer.traces)
    assert back.stage_idx == 2


def test_from_dense_with_layer_matches_oracle():
    """from_dense base + a source layer reconstructs the same as the oracle."""
    rng = np.random.RandomState(3)
    dense = rng.randn(T, LY, LX).astype(np.float32)
    P = 7
    flat_idx = np.sort(rng.choice(N_PIX, size=P, replace=False)).astype(np.int64)
    W_design = rng.rand(2, P).astype(np.float32)
    traces = rng.randn(2, T).astype(np.float32)
    oracle = _oracle_subtract(dense, flat_idx, W_design, traces)

    view = ResidualView.from_dense(dense).with_source(flat_idx, W_design, traces)
    assert np.allclose(view.read_chunk(0, T), oracle, rtol=RTOL, atol=ATOL)


def test_rpca_factors_roundtrip_contract():
    """Factors from ``rpca._factor_from_robust_L`` round-trip through the view.

    Proves the RPCA background path emits ``svd_factors.npz`` in a byte-compatible
    format: the ResidualView reconstructs ``M − L`` identically whether built from
    the in-memory factors (``from_factors``) or from a saved/reloaded npz
    (``from_foundation``), matching a dense ``M − L`` oracle.
    """
    from roigbiv.pipeline import rpca

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        rng = np.random.RandomState(5)
        movie = rng.randint(0, 500, size=(T, LY, LX)).astype(np.int16)
        data_bin = td / "data.bin"
        movie.tofile(str(data_bin))

        # A genuinely low-rank L over the binned grid (bin_size = 1 ⇒ T_bin = T).
        L_bin = (rng.randn(T, 3) @ rng.randn(3, N_PIX)).astype(np.float32)
        U, S, V_bin = rpca._factor_from_robust_L(L_bin, n_svd=N_SVD, force_cpu=True)
        bin_size = 1

        US_k = (U[:, :K] * S[:K][None, :]).astype(np.float32)
        L_full = V_bin[:, :K] @ US_k.T                      # (T, N_pix)
        M = movie.reshape(T, N_PIX).astype(np.float32)
        oracle = (M - L_full).reshape(T, LY, LX)

        view = ResidualView.from_factors(
            data_bin, U, S, V_bin, bin_size, (T, LY, LX), K,
        )
        assert np.allclose(view.read_chunk(0, T), oracle, rtol=RTOL, atol=ATOL)

        svd_path = td / "svd_factors.npz"
        np.savez(str(svd_path), U=U, S=S, V_bin=V_bin,
                 bin_size=np.int32(bin_size), T=np.int32(T))
        view2 = ResidualView.from_foundation(data_bin, svd_path, (T, LY, LX), K)
        assert np.allclose(view2.read_chunk(0, T), oracle, rtol=RTOL, atol=ATOL)
        assert np.allclose(view2.read_rows(0, LY), oracle, rtol=RTOL, atol=ATOL)
