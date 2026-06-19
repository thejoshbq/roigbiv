"""Robust low-rank + sparse background separation for the Foundation.

The default Foundation background is a plain top-*k* truncated SVD
(:func:`roigbiv.pipeline.foundation._binned_svd_gpu`). Its leading components
align with per-pixel **mean brightness**, so the static structural image *and*
the brightest / tonic somata get absorbed into the low-rank ``L`` — leaving the
residual ``mean_S ≈ 0`` and erasing the easiest cells from the substrate the
pipeline detects on.

This module replaces that step (opt-in via ``cfg.background_method == "rpca"``)
with a **robust** decomposition ``M_bin ≈ L_bin + S_bin`` where ``L_bin`` is
genuinely low-rank background and ``S_bin`` is the sparse foreground that a plain
SVD would otherwise pull into ``L``. Primary solver is inexact-ALM Principal
Component Pursuit (IALM-PCP); GoDec is a lighter fallback.

Contract preservation (load-bearing — see ``roigbiv/pipeline/residual.py``):
the residual is *virtual*, reconstructed on demand as ``S = M − L`` from
``data.bin`` + the SVD factors in ``svd_factors.npz``. Because the robust
``L_bin`` is genuinely rank ``≤ max_rank``, we take its **exact** SVD
(:func:`_factor_from_robust_L`) and emit ``(U, S, V_bin)`` in the *identical*
pixel-indexed convention as ``_binned_svd_gpu`` — so ``residual.py``,
``ResidualView``, and the ``svd_factors.npz`` format are untouched. Only *how*
``L`` is estimated changes; the downstream arithmetic is byte-compatible.

Memory: IALM holds ~5 live copies of the binned matrix on the GPU, so the
``rpca_bin_frames`` target alone (purely temporal, N_pix-blind) overflows a 16 GB
card on large FOVs. :func:`estimate_rpca_bin_frames` caps the target against
*currently free* GPU memory so the peak fits regardless of frame size; the
foundation then retries coarser and finally on CPU if the estimate falls short.
The background is slow, so coarser temporal sampling is harmless, and the
``bin_size`` persisted to ``svd_factors.npz`` stays self-consistent.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from roigbiv.pipeline.device import cuda_compute_capable


# ─────────────────────────────────────────────────────────────────────────
# Adaptive GPU-memory binning
# ─────────────────────────────────────────────────────────────────────────
#
# IALM holds ~_IALM_LIVE_COPIES live full-size copies of the binned matrix
# (M, Y, S, L + the X/Z temporaries in :func:`_ialm_pcp`). The plain
# ``rpca_bin_frames`` target is purely temporal and ignores N_pix, so a large
# FOV (e.g. 1024×1024 → ~1M pixels) blows the peak past a 16 GB card. We size
# the temporal-bin target so ``copies × T_bin × N_pix × 4`` stays within a
# fraction of *currently free* GPU memory, then retry coarser / fall back to CPU
# (driven by foundation.compute_background_separation) if the estimate is short.

MIN_RPCA_FRAMES = 150       # floor: IALM still needs enough temporal samples
_IALM_LIVE_COPIES = 6       # worst-case simultaneous full-size tensors
_RPCA_MEM_FRACTION = 0.6    # headroom for fragmentation + the partial-SVD workspace


def free_gpu_bytes(force_cpu: bool = False) -> Optional[int]:
    """Free bytes on the active CUDA device, or ``None`` if RPCA runs on CPU.

    ``None`` means "no GPU memory ceiling applies" — either ``force_cpu``, no
    usable CUDA device, or the probe failed. Callers treat ``None`` as "leave the
    requested bin target unchanged".
    """
    if force_cpu:
        return None
    import torch

    if not (cuda_compute_capable() and torch.cuda.is_available()):
        return None
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def estimate_rpca_bin_frames(
    n_pix: int,
    requested_frames: int,
    free_bytes: Optional[int],
    *,
    bytes_per_elem: int = 4,
) -> int:
    """Cap the RPCA temporal-bin target so the binned matrix fits GPU memory.

    Sizes ``T_bin`` so ``_IALM_LIVE_COPIES × T_bin × n_pix × bytes_per_elem``
    stays within ``_RPCA_MEM_FRACTION`` of ``free_bytes``. The result is
    ``max(MIN_RPCA_FRAMES, min(requested_frames, mem_ceiling))`` — small FOVs
    keep ``requested_frames`` (the ceiling never binds); only large FOVs are
    coarsened. ``free_bytes is None`` (CPU run / no probe) returns
    ``requested_frames`` unchanged.
    """
    requested = int(requested_frames)
    if free_bytes is None or n_pix <= 0:
        return requested
    budget = _RPCA_MEM_FRACTION * float(free_bytes)
    ceiling = int(budget // (_IALM_LIVE_COPIES * int(n_pix) * int(bytes_per_elem)))
    ceiling = max(MIN_RPCA_FRAMES, ceiling)
    return max(MIN_RPCA_FRAMES, min(requested, ceiling))


# ─────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────

def robust_lowrank_sparse(
    M_bin: np.ndarray,
    *,
    max_rank: int,
    lam: Optional[float] = None,
    mu: Optional[float] = None,
    tol: float = 1e-3,
    max_iter: int = 100,
    force_cpu: bool = False,
    allow_cpu_fallback: bool = True,
    seed: int = 0,
    method: str = "ialm",
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose ``M_bin`` (T_bin, N_pix) into robust ``L_bin + S_bin``.

    Parameters
    ----------
    M_bin    : (T_bin, N_pix) float32 — temporally-binned movie (time × pixels).
    max_rank : rank cap for ``L_bin``; keeps it low-rank so the post-hoc SVD in
               :func:`_factor_from_robust_L` is exact and the residual contract
               is preserved. Mirrors ``cfg.k_background``.
    lam      : PCP sparsity weight. ``None`` → ``1/sqrt(max(T_bin, N_pix))``
               (Candès et al. 2011).
    mu       : ALM penalty. ``None`` → ``1.25 / ‖M_bin‖₂`` (spectral norm,
               estimated by a rank-1 randomized SVD).
    tol      : relative constraint-residual ``‖M−L−S‖_F / ‖M‖_F`` stop tol.
    max_iter : hard iteration cap (data-dependent convergence backstop).
    force_cpu, seed : mirror :func:`foundation._binned_svd_gpu`.
    force_cpu : run on system RAM, skipping the GPU entirely.
    allow_cpu_fallback : on a GPU OOM, retry on CPU (``True``) versus re-raising
               so a caller can retry coarser first (``False``). The foundation
               drives a retry-coarser-then-CPU ladder with this; the innermost
               default stays ``True`` for standalone callers.
    method   : ``"ialm"`` (primary PCP) or ``"godec"`` (lighter fallback).

    Returns
    -------
    L_bin, S_bin : both (T_bin, N_pix) float32 on host.
    """
    import torch

    if method not in ("ialm", "godec"):
        raise ValueError(f"Unknown rpca method {method!r}; expected 'ialm' or 'godec'.")

    M_bin = np.ascontiguousarray(M_bin, dtype=np.float32)
    T_bin, N_pix = M_bin.shape
    q = max(1, min(int(max_rank), T_bin - 1, N_pix - 1))
    if lam is None:
        lam = 1.0 / float(np.sqrt(max(T_bin, N_pix)))

    def _run(device: str) -> tuple[np.ndarray, np.ndarray]:
        # Seed before any randomized SVD so the top-q subspace is reproducible
        # run-to-run (same rationale as foundation._binned_svd_gpu:280).
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        M = torch.from_numpy(M_bin).to(device)
        if method == "ialm":
            L, S = _ialm_pcp(M, q, float(lam), mu, float(tol), int(max_iter))
        else:
            L, S = _godec(M, q, float(lam), float(tol), int(max_iter))
        return (L.detach().cpu().numpy().astype(np.float32),
                S.detach().cpu().numpy().astype(np.float32))

    device = "cpu" if force_cpu else ("cuda" if cuda_compute_capable() else "cpu")
    t0 = time.time()
    try:
        L_bin, S_bin = _run(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if device == "cpu" or not allow_cpu_fallback:
            # On CPU there is nowhere to fall back; when the caller drives its
            # own retry ladder it wants the OOM to propagate (the ``finally``
            # below still releases the GPU cache).
            raise
        print(f"  WARN: RPCA on GPU failed ({type(exc).__name__}: {exc}); "
              f"falling back to CPU.", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        L_bin, S_bin = _run("cpu")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"  RPCA ({method}) {M_bin.shape} rank≤{q} in {time.time()-t0:.1f}s "
          f"on {device}", flush=True)
    return L_bin, S_bin


def _factor_from_robust_L(
    L_bin: np.ndarray,
    n_svd: int,
    force_cpu: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact SVD of the robust ``L_bin`` in the ``_binned_svd_gpu`` convention.

    Returns ``(U, S, V)`` with ``L_bin ≈ V @ diag(S) @ U.T``:
      - U (N_pix, n_svd) — spatial components (pixel-indexed),
      - S (n_svd,)       — singular values,
      - V (T_bin, n_svd) — temporal components.

    Identical orientation/dtype to :func:`foundation._binned_svd_gpu` (it factors
    ``L_bin^T`` so ``U`` indexes pixels), so the resulting ``svd_factors.npz`` is
    byte-compatible with the residual view. Cheap because ``L_bin`` is genuinely
    low-rank.
    """
    import torch

    L_bin = np.ascontiguousarray(L_bin, dtype=np.float32)
    T_bin, N_pix = L_bin.shape
    q = max(1, min(int(n_svd), T_bin - 1, N_pix - 1))

    def _svd(device: str):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        A = torch.from_numpy(L_bin.T).to(device)        # (N_pix, T_bin)
        U_t, S_t, V_t = torch.svd_lowrank(A, q=q, niter=4)
        return (U_t.detach().cpu().numpy().astype(np.float32),
                S_t.detach().cpu().numpy().astype(np.float32),
                V_t.detach().cpu().numpy().astype(np.float32))

    device = "cpu" if force_cpu else ("cuda" if cuda_compute_capable() else "cpu")
    try:
        U, S, V = _svd(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        U, S, V = _svd("cpu")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return U, S, V


# ─────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────

def _spectral_norm(M, niter: int = 4) -> float:
    """Approximate ‖M‖₂ (largest singular value) via a rank-1 randomized SVD."""
    import torch
    _, s, _ = torch.svd_lowrank(M, q=1, niter=niter)
    return float(s[0].item())


def _soft_threshold(A, t: float):
    """Element-wise soft-threshold ``sign(A)·max(|A|−t, 0)`` (L1 prox)."""
    import torch
    return torch.sign(A) * torch.clamp(torch.abs(A) - t, min=0.0)


def _ialm_pcp(M, q: int, lam: float, mu: Optional[float],
              tol: float, max_iter: int):
    """Inexact-ALM Principal Component Pursuit (Lin, Chen & Ma 2010).

    Minimizes ``‖L‖_* + λ‖S‖_1  s.t.  L + S = M`` with a partial (rank-``q``)
    singular-value-threshold step, so ``L`` stays rank ``≤ q`` and memory is
    bounded. Returns ``(L, S)`` torch tensors on the input device.
    """
    import torch

    normfro = torch.linalg.norm(M)
    norm2 = _spectral_norm(M)
    norm_inf = float(torch.max(torch.abs(M)).item()) / lam
    J = max(norm2, norm_inf)
    Y = M / J                                  # dual variable
    S = torch.zeros_like(M)
    L = torch.zeros_like(M)

    mu = (1.25 / norm2) if mu is None else float(mu)
    mu_bar = mu * 1e7
    rho = 1.5

    for it in range(max_iter):
        inv_mu = 1.0 / mu
        # ── L update: partial SVT of (M − S + Y/μ) ─────────────────────────
        X = M - S + Y * inv_mu
        U, s, V = torch.svd_lowrank(X, q=q, niter=2)
        del X
        s_t = torch.clamp(s - inv_mu, min=0.0)
        L = (U * s_t) @ V.mT                    # (m, n), rank ≤ q
        # ── S update: L1 prox of (M − L + Y/μ) ─────────────────────────────
        S = _soft_threshold(M - L + Y * inv_mu, lam * inv_mu)
        # ── dual update + convergence ──────────────────────────────────────
        Z = M - L - S
        Y = Y + mu * Z
        err = float((torch.linalg.norm(Z) / normfro).item())
        del Z
        mu = min(mu * rho, mu_bar)
        if err < tol:
            break

    return L, S


def _godec(M, q: int, lam: float, tol: float, max_iter: int):
    """GoDec-style low-rank + sparse fallback (Zhou & Tao 2011).

    Alternates a rank-``q`` projection of ``M − S`` with a magnitude-threshold of
    ``M − L``. Cheaper and fewer knobs than IALM; ``lam`` sets the keep-fraction
    of the sparse term (entries above the ``(1−lam)`` magnitude quantile).
    """
    import torch

    normfro = torch.linalg.norm(M)
    S = torch.zeros_like(M)
    L = torch.zeros_like(M)
    keep_frac = min(max(lam, 1e-4), 0.5)

    for it in range(max_iter):
        # Low-rank projection of (M − S).
        U, s, V = torch.svd_lowrank(M - S, q=q, niter=2)
        L = (U * s) @ V.mT
        # Sparse residual: keep the largest-magnitude entries by quantile.
        R = M - L
        absR = torch.abs(R)
        thr = torch.quantile(absR.flatten()[::257], 1.0 - keep_frac)
        S = torch.where(absR >= thr, R, torch.zeros_like(R))
        err = float((torch.linalg.norm(M - L - S) / normfro).item())
        if err < tol:
            break

    return L, S
