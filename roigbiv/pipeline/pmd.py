"""
ROI G. Biv pipeline — Phase 2: PMD spatiotemporal denoiser (OPTIONAL).

Patch-wise penalized-matrix-decomposition denoising (Buchanan et al. lineage) of
the residual movie that feeds Stages 3 and 4. The denoiser is a low-rank
truncation per overlapping spatial patch: each patch's (T, P) pixel-time matrix
is mean-centered, decomposed by a truncated SVD, and reconstructed from only the
components that rise above the Marchenko–Pastur noise edge. Overlapping patches
are averaged (overlap-add) to suppress block seams.

Why this exists (see docs/phase2_pmd_insertion_point.md):
  - SVD truncation in Foundation denoises the *background* but leaves shot noise
    in the residual; Stage 3 (per-pixel MAD) and Stage 4 (z-scored bandpassed
    residual) must work against that noise. PMD lifts residual SNR spatiotempo-
    rally — the gain faint sparse/tonic detections most need.

Design contract:
  - Reads the input through the ResidualView read primitives only; never mutates
    L+S, the SVD factors, or the reconstruction math.
  - Materializes the denoised movie ONCE to a float32 memmap (PMD is a global
    patch decomposition — it cannot be applied coherently per on-demand read),
    then wraps that memmap as the dense base of a fresh ResidualView. The
    Stage-3 source subtraction carries the dense base forward via
    ``ResidualView.with_source`` (dense=self._dense), so Stage 4 inherits the
    denoised residual automatically.
  - Streams band-by-band with bounded RAM; torch on GPU with CPU fallback.

OFF by default (``cfg.use_pmd_denoise``); enabling it never changes Stage 1/2.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from roigbiv.pipeline.residual import ResidualView


# ─────────────────────────────────────────────────────────────────────────
# Per-patch PMD core (torch)
# ─────────────────────────────────────────────────────────────────────────

def _pmd_denoise_patch(patch_tp, *, max_rank: int, margin: float, torch, device):
    """Low-rank PMD denoise of one patch ``(T, P)`` float32 ndarray → ``(T, P)``.

    Rank is selected by the Marchenko–Pastur upper edge: for a ``T×P`` matrix of
    i.i.d. noise with std σ, the largest noise singular value scales like
    ``σ·(√T + √P)``. Components above ``(1+margin)`` times that edge are treated
    as signal and retained; the rest are discarded as noise.
    """
    T, P = patch_tp.shape
    X = torch.as_tensor(patch_tp, device=device, dtype=torch.float32)
    mu = X.mean(dim=0, keepdim=True)                         # (1, P) per-pixel baseline
    Xc = X - mu

    # Robust per-patch noise std from lag-1 temporal differences:
    # var(diff)/2 ≈ noise var when signal is temporally smooth.
    if T > 1:
        d = Xc[1:] - Xc[:-1]
        sigma = float(torch.sqrt(torch.clamp(torch.median(d * d) * 0.5, min=1e-12)))
    else:
        sigma = float(torch.sqrt(torch.clamp(torch.mean(Xc * Xc), min=1e-12)))

    q = int(min(max_rank, T, P))
    if q < 1:
        return np.broadcast_to(mu.cpu().numpy(), (T, P)).copy()

    # Economy low-rank SVD: Xc ≈ U(T,q) diag(S) V(P,q)^T
    U, S, V = torch.svd_lowrank(Xc, q=q, niter=2)

    mp_edge = sigma * (math.sqrt(T) + math.sqrt(P)) * (1.0 + float(margin))
    keep = int((S > mp_edge).sum().item())
    keep = max(0, min(keep, q))
    if keep == 0:
        return np.broadcast_to(mu.cpu().numpy(), (T, P)).copy()

    Xhat = (U[:, :keep] * S[:keep]) @ V[:, :keep].T + mu
    return Xhat.cpu().numpy().astype(np.float32, copy=False)


# ─────────────────────────────────────────────────────────────────────────
# Streaming driver
# ─────────────────────────────────────────────────────────────────────────

def pmd_denoise_to_memmap(view_in, out_path: Path, shape, cfg, *, gpu: bool = True) -> None:
    """Denoise a residual view into a ``(T, H, W)`` float32 memmap (overlap-add).

    Processes horizontal bands of ``pmd_patch_size`` rows (advancing by
    ``patch_size − overlap``), tiling each band into overlapping patches across
    the width. Reconstructions are accumulated into the output memmap and a
    per-pixel weight map, then normalized — so overlapping estimates are averaged
    and block seams are suppressed. Peak RAM ≈ one band ``(T, ps, W)``.
    """
    from roigbiv.pipeline.diskguard import ensure_free_space

    T, H, W = (int(s) for s in shape)
    ps = int(cfg.pmd_patch_size)
    ov = int(cfg.pmd_patch_overlap)
    ov = max(0, min(ov, ps - 1))
    stride = max(1, ps - ov)

    try:
        import torch
        use_cuda = bool(gpu) and not getattr(cfg, "force_cpu", False) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
    except Exception as exc:  # torch missing — should not happen in this env
        raise RuntimeError(f"PMD denoise requires torch: {exc}")

    band_bytes = T * ps * W * 4
    budget = int(getattr(cfg, "pmd_band_budget_bytes", 1 << 30))
    if band_bytes > budget:
        print(
            f"  PMD: per-band working set {band_bytes/1e9:.2f} GB exceeds budget "
            f"{budget/1e9:.2f} GB (T={T}, patch={ps}, W={W}); proceeding anyway",
            flush=True,
        )

    ensure_free_space(out_path, T * H * W * 4, label="pmd denoise")
    out_mm = np.memmap(str(out_path), dtype=np.float32, mode="w+", shape=(T, H, W))
    out_mm[:] = 0.0
    wmap = np.zeros((H, W), dtype=np.float32)

    # Inclusive coverage of the H/W extents: offsets land on a stride grid and
    # the final offset is clamped so the last patch reaches the border.
    def _offsets(n: int) -> list[int]:
        if n <= ps:
            return [0]
        offs = list(range(0, n - ps + 1, stride))
        if offs[-1] != n - ps:
            offs.append(n - ps)
        return offs

    y_offsets = _offsets(H)
    x_offsets = _offsets(W)

    t0 = time.time()
    n_bands = len(y_offsets)
    for bi, yy in enumerate(y_offsets):
        ye = min(yy + ps, H)
        h = ye - yy
        band = np.asarray(view_in.read_rows(yy, ye), dtype=np.float32)   # (T, h, W)
        band_out = np.zeros((T, h, W), dtype=np.float32)
        band_w = np.zeros((h, W), dtype=np.float32)
        for xx in x_offsets:
            xe = min(xx + ps, W)
            w = xe - xx
            patch = band[:, :, xx:xe].reshape(T, h * w)                  # (T, h*w)
            denoised = _pmd_denoise_patch(
                patch, max_rank=int(cfg.pmd_max_rank),
                margin=float(cfg.pmd_rank_margin), torch=torch, device=device,
            ).reshape(T, h, w)
            band_out[:, :, xx:xe] += denoised
            band_w[:, xx:xe] += 1.0
        # Accumulate band into the global memmap (y-bands overlap → += merges them).
        out_mm[:, yy:ye, :] += band_out
        wmap[yy:ye, :] += band_w
        if use_cuda:
            torch.cuda.empty_cache()
        print(f"  PMD band {bi+1}/{n_bands} (rows {yy}:{ye}) "
              f"{time.time()-t0:.1f}s", flush=True)

    # Normalize overlap-add: divide each pixel by how many patch estimates hit
    # it. The offset grid spans [0,H)×[0,W) with stride ≤ ps, so every pixel is
    # covered by ≥1 patch — a zero weight would be a coverage bug, not a gap.
    # (The division loop chunks rows only to bound RAM; the blocks are disjoint
    # and cover every row exactly once, so each pixel is divided once by its own
    # accumulated weight regardless of the chunk size.)
    assert np.all(wmap >= 1.0), "PMD overlap-add left uncovered pixels"
    for y0 in range(0, H, max(1, ps)):
        y1 = min(y0 + max(1, ps), H)
        out_mm[:, y0:y1, :] /= wmap[None, y0:y1, :]
    out_mm.flush()
    del out_mm


def pmd_denoise_view(view_in, output_dir: Path, cfg, *, gpu: bool = True) -> ResidualView:
    """Materialize a PMD-denoised residual and return it as a dense ResidualView.

    The returned view reconstructs from the denoised memmap via the existing
    ``_dense`` read path (oracle-tested in ``test_residual_view.py``); the engine
    code is unmodified. ``with_source`` carries the dense base forward, so the
    Stage-3 subtraction and Stage 4 inherit the denoised residual.
    """
    shape = tuple(int(s) for s in view_in.shape)
    pmd_dir = Path(output_dir) / "pmd"
    pmd_dir.mkdir(parents=True, exist_ok=True)
    mm_path = pmd_dir / "residual_pmd.dat"

    pmd_denoise_to_memmap(view_in, mm_path, shape, cfg, gpu=gpu)

    denoised = np.memmap(str(mm_path), dtype=np.float32, mode="r", shape=shape)
    # Pass the memmap straight to the constructor (NOT from_dense, which would
    # np.asarray-copy the whole movie into RAM). __init__ stores it as _dense
    # without copying; read_chunk slices then copies → bounded RAM preserved.
    return ResidualView(shape, dense=denoised)
