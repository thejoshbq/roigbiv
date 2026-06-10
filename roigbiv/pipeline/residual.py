"""Lazy virtual residual for the sequential subtractive pipeline.

The pipeline's residual is a destructive chain ``S → S1 → S2 → S3`` where each
stage subtracts its detected sources from the prior residual. Materializing
each link as a dense ``(T, Ly, Lx)`` float32 memmap costs ~10-19 GB *each* and
peaks at 40-60 GB across the chain — the source of the silent SIGBUS crash when
the disk fills mid-write.

Nothing actually needs the dense array on disk: every link is reconstructible
on demand from artifacts that already exist.

    S_0(p, t) = M(p, t) − L(p, t),       L = US_k @ V_k_full[t].T
    S_n(p, t) = S_{n-1}(p, t) − Σ_i w_i(p)·c_i(t)   for p ∈ source_i (stages ≤ n)

where ``M`` is Suite2p's ``data.bin`` (int16 memmap), ``US_k``/``V_k_full`` come
from ``svd_factors.npz`` (foundation), and each subtraction stage contributes a
:class:`SourceLayer` of (mask pixels, spatial weights, temporal traces).

:class:`ResidualView` holds only small in-RAM arrays (the SVD factors and the
source layers — a few MB each) plus a zero-cost int16 memmap of ``data.bin``,
and reconstructs any temporal chunk / spatial band / pixel set on demand. The
arithmetic mirrors the old materializing code (``subtraction.subtract_sources``)
so reconstructed values match the previously-written ``.dat`` within float32
tolerance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Source layer (one per subtraction stage)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SourceLayer:
    """A single stage's rank-1 source contribution to the residual.

    flat_idx  : (P,) int64 — union of ROI mask pixels (flat row-major indices),
                ascending (from ``np.flatnonzero``), so range tests and
                ``searchsorted`` are valid.
    W_design  : (N, P) float32 — per-ROI spatial weight over the union pixels.
    traces    : (N, T) float32 — per-ROI temporal trace c_i(t).
    stage_idx : the detection stage that produced this layer (1, 2 or 3).
    """
    flat_idx: np.ndarray
    W_design: np.ndarray
    traces: np.ndarray
    stage_idx: int = 0

    def save(self, path: Path) -> None:
        np.savez(
            str(path),
            flat_idx=self.flat_idx.astype(np.int64),
            W_design=self.W_design.astype(np.float32),
            traces=self.traces.astype(np.float32),
            stage_idx=np.int32(self.stage_idx),
        )

    @classmethod
    def load(cls, path: Path) -> "SourceLayer":
        z = np.load(str(path))
        return cls(
            flat_idx=np.asarray(z["flat_idx"], dtype=np.int64),
            W_design=np.asarray(z["W_design"], dtype=np.float32),
            traces=np.asarray(z["traces"], dtype=np.float32),
            stage_idx=int(z["stage_idx"]),
        )


# ─────────────────────────────────────────────────────────────────────────
# Virtual residual view
# ─────────────────────────────────────────────────────────────────────────

class ResidualView:
    """Lazily reconstructs a residual ``(T, Ly, Lx)`` float32 array on demand.

    Construct via :meth:`from_factors` (foundation, in-memory arrays),
    :meth:`from_foundation` (resume, reads ``svd_factors.npz`` + ``data.bin``),
    or :meth:`from_dense` (tests / synthetic residuals). Advance the chain with
    :meth:`with_source`.

    Read primitives (all return ``float32`` and apply source layers in order):
      * :meth:`read_chunk` ``(t0, t1) -> (cs, Ly, Lx)`` — temporal slab.
      * :meth:`read_rows`  ``(y0, y1) -> (T, h, Lx)`` — full-T spatial band.
      * :meth:`read_pixels```(ys, xs) -> (T, P)`` — arbitrary pixel timecourses.
    """

    def __init__(
        self,
        shape: tuple,
        *,
        data_bin_path: Optional[Path] = None,
        US_k: Optional[np.ndarray] = None,
        V_k_full: Optional[np.ndarray] = None,
        sources: Optional[list] = None,
        dense: Optional[np.ndarray] = None,
    ):
        self.shape = tuple(int(s) for s in shape)
        self.T, self.Ly, self.Lx = self.shape
        self.N_pix = self.Ly * self.Lx
        self.data_bin_path = None if data_bin_path is None else Path(data_bin_path)
        self.US_k = US_k                          # (N_pix, k) float32
        self.V_k_full = V_k_full                  # (T, k) float32
        self.k = 0 if US_k is None else int(US_k.shape[1])
        self.sources: list[SourceLayer] = list(sources) if sources else []
        self._dense = dense                       # (T, Ly, Lx) — test/synthetic base
        self._movie = None                        # lazily-opened int16 memmap

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_factors(
        cls,
        data_bin_path: Path,
        U: np.ndarray,
        S: np.ndarray,
        V_bin: np.ndarray,
        bin_size: int,
        shape: tuple,
        k_background: int,
        sources: Optional[list] = None,
    ) -> "ResidualView":
        """Build from in-memory SVD factors (foundation hot path)."""
        from roigbiv.pipeline.foundation import _upsample_V

        T, Ly, Lx = (int(s) for s in shape)
        n_svd = int(U.shape[1])
        k = min(int(k_background), n_svd)
        US_k = (U[:, :k] * S[:k][np.newaxis, :]).astype(np.float32)
        V_full = _upsample_V(V_bin, int(bin_size), T)
        V_k_full = np.ascontiguousarray(V_full[:, :k].astype(np.float32))
        return cls(
            (T, Ly, Lx),
            data_bin_path=data_bin_path,
            US_k=US_k,
            V_k_full=V_k_full,
            sources=sources,
        )

    @classmethod
    def from_foundation(
        cls,
        data_bin_path: Path,
        svd_factors_path: Path,
        shape: tuple,
        k_background: int,
        sources: Optional[list] = None,
    ) -> "ResidualView":
        """Build by loading ``svd_factors.npz`` from disk (resume path)."""
        z = np.load(str(svd_factors_path))
        return cls.from_factors(
            data_bin_path,
            np.asarray(z["U"], dtype=np.float32),
            np.asarray(z["S"], dtype=np.float32),
            np.asarray(z["V_bin"], dtype=np.float32),
            int(z["bin_size"]),
            shape,
            k_background,
            sources=sources,
        )

    @classmethod
    def from_dense(cls, dense: np.ndarray, sources: Optional[list] = None) -> "ResidualView":
        """Wrap an in-memory ``(T, Ly, Lx)`` array (tests / synthetic data)."""
        dense = np.asarray(dense, dtype=np.float32)
        return cls(dense.shape, dense=dense, sources=sources)

    # ── Chain advancement ─────────────────────────────────────────────────

    def with_source(
        self,
        flat_idx: np.ndarray,
        W_design: np.ndarray,
        traces: np.ndarray,
        stage_idx: int = 0,
    ) -> "ResidualView":
        """Return a new view with one more :class:`SourceLayer` appended.

        Cheap — shares the movie/SVD arrays by reference; only the small source
        list grows.
        """
        layer = SourceLayer(
            flat_idx=np.asarray(flat_idx, dtype=np.int64),
            W_design=np.asarray(W_design, dtype=np.float32),
            traces=np.asarray(traces, dtype=np.float32),
            stage_idx=int(stage_idx),
        )
        new = ResidualView(
            self.shape,
            data_bin_path=self.data_bin_path,
            US_k=self.US_k,
            V_k_full=self.V_k_full,
            sources=self.sources + [layer],
            dense=self._dense,
        )
        new._movie = self._movie
        return new

    # ── Internals ─────────────────────────────────────────────────────────

    @property
    def movie(self) -> np.memmap:
        if self._movie is None:
            from roigbiv.pipeline.foundation import _open_data_bin
            self._movie = _open_data_bin(self.data_bin_path, self.Ly, self.Lx)
        return self._movie

    # ── Read primitives ───────────────────────────────────────────────────

    def read_chunk(self, t0: int, t1: int) -> np.ndarray:
        """Reconstruct frames ``[t0, t1)`` → ``(cs, Ly, Lx)`` float32."""
        cs = t1 - t0
        if self._dense is not None:
            out = np.array(self._dense[t0:t1], dtype=np.float32).reshape(cs, self.N_pix)
        else:
            M = np.asarray(self.movie[t0:t1], dtype=np.float32).reshape(cs, self.N_pix)
            L = self.V_k_full[t0:t1] @ self.US_k.T          # (cs, N_pix)
            out = M - L
        for layer in self.sources:
            # out[:, idx] -= traces[:, t0:t1].T @ W_design  ≡ subtract_sources
            sub = layer.traces[:, t0:t1].T @ layer.W_design  # (cs, P)
            out[:, layer.flat_idx] -= sub
        return out.reshape(cs, self.Ly, self.Lx)

    def read_rows(self, y0: int, y1: int) -> np.ndarray:
        """Reconstruct row band ``[y0, y1)`` over all T → ``(T, h, Lx)`` float32.

        Row-major flat layout makes ``[y0:y1]`` a contiguous flat range, so the
        L term is a fast contiguous slice of ``US_k`` and source restriction is
        a range test.
        """
        h = y1 - y0
        N_band = h * self.Lx
        p0 = y0 * self.Lx
        p1 = y1 * self.Lx
        if self._dense is not None:
            out = np.array(self._dense[:, y0:y1, :], dtype=np.float32).reshape(self.T, N_band)
        else:
            M = np.asarray(self.movie[:, y0:y1, :], dtype=np.float32).reshape(self.T, N_band)
            L = self.V_k_full @ self.US_k[p0:p1].T          # (T, N_band)
            out = M - L
        for layer in self.sources:
            sel = (layer.flat_idx >= p0) & (layer.flat_idx < p1)
            if sel.any():
                local = layer.flat_idx[sel] - p0
                sub = layer.traces.T @ layer.W_design[:, sel]   # (T, n_sel)
                out[:, local] -= sub
        return out.reshape(self.T, h, self.Lx)

    def read_pixels(self, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
        """Reconstruct timecourses at pixels ``(ys, xs)`` → ``(T, P)`` float32."""
        ys = np.asarray(ys)
        xs = np.asarray(xs)
        flat = (ys.astype(np.int64) * self.Lx + xs.astype(np.int64))
        if self._dense is not None:
            out = np.asarray(self._dense[:, ys, xs], dtype=np.float32)
        else:
            M = np.asarray(self.movie[:, ys, xs], dtype=np.float32)   # (T, P)
            L = self.V_k_full @ self.US_k[flat].T                     # (T, P)
            out = M - L
        for layer in self.sources:
            sel = np.isin(flat, layer.flat_idx)
            if sel.any():
                pos = np.searchsorted(layer.flat_idx, flat[sel])
                sub = layer.traces.T @ layer.W_design[:, pos]         # (T, n_sel)
                out[:, sel] -= sub
        return out

    def iter_chunks(self, chunk: int):
        """Yield ``(t0, t1, S_chunk)`` over temporal chunks of ``chunk`` frames.

        Drop-in for the old ``_iter_S_chunks`` generator.
        """
        for t0 in range(0, self.T, chunk):
            t1 = min(t0 + chunk, self.T)
            yield t0, t1, self.read_chunk(t0, t1)
