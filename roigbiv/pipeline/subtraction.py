"""
ROI G. Biv pipeline — Source Subtraction Engine (spec §5).

This is the core custom component of the sequential pipeline. Between every
pair of detection stages, source subtraction removes the fluorescence
contribution of detected ROIs from the residual movie so the next stage
sees a cleaner substrate (addresses Blindspots 1, 2, 5).

Algorithm (spec §5.1):
  Step 1 — spatial profile: w_i(x,y) = mean_t[S(x,y,t)] / max_mask[mean_t S]
  Step 2 — simultaneous trace: at each t, solve S(p,t) ≈ Σ_i w_i(p) c_i(t)
           via least squares over the union of ROI pixels
  Step 3 — rank-1 subtract: S_new(p,t) = S(p,t) - w_i(p) c_i(t) for p ∈ mask_i

Implementation notes (Plan agent A3/A4):
  - Trace estimation uses GPU-chunked normal equations (W.T W + λI)⁻¹ W.T b
    rather than CPU lstsq. This is ~100× faster and memory-bounded by chunk size.
  - NNLS fallback runs ONLY on ROIs flagged by post-hoc validation, and
    only over their LOCAL mask pixels — this reduces each NNLS call from
    30k×150 to ~200×1.
  - Rank-1 subtraction streams through the S memmap per temporal chunk so
    the full (T, H, W) residual never lives in RAM.
  - Writes a new memmap at residual_S1.dat; the original residual_S.dat is
    preserved for validation and future re-subtraction.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.pipeline.types import ROI, PipelineConfig


# ─────────────────────────────────────────────────────────────────────────
# Step 1 — Spatial profiles
# ─────────────────────────────────────────────────────────────────────────

def estimate_spatial_profiles(
    profile_source: np.ndarray,
    rois: list[ROI],
) -> list[np.ndarray]:
    """Compute the normalized spatial profile for each ROI.

    w_i(x,y) = profile_source(x,y) / max_{x,y ∈ mask_i} profile_source(x,y)   if mask_i[x,y]
              0                                                                otherwise

    Profiles peak at 1.0. Pixels outside the mask have w_i = 0.

    Notes on profile_source choice
    ------------------------------
    Spec §5.1 Step 1 calls for `mean_t[S(x,y,t)]` as the weight source. Under
    truncated-SVD-based L+S (our implementation), the top-k components absorb
    per-pixel mean brightness, so mean_t[S] ≈ 0 everywhere with no spatial
    structure. This produces near-random profiles that fit poorly in the
    subsequent least-squares trace step.

    The semantic intent of the spec's weight is "spatial pattern of residual
    activity at each mask pixel." Under our L+S, `std_t[S]` (per-pixel rms)
    preserves this pattern faithfully: active pixels have higher variance
    than neuropil. Callers should therefore pass `std_S` as profile_source.

    Parameters
    ----------
    profile_source : (H, W) float32 — spatial activity map (e.g., std_S or max_S)
    rois           : list of ROI objects with .mask attribute (H, W) bool

    Returns
    -------
    list of (H, W) float32 arrays, one per ROI, same shape as profile_source.
    """
    profiles = []
    for roi in rois:
        w = np.zeros_like(profile_source, dtype=np.float32)
        mask = roi.mask
        vals = profile_source[mask]
        if vals.size == 0:
            profiles.append(w)
            continue
        peak = float(vals.max())
        if peak <= 0:
            # All-negative (rare for std_S but possible for mean_S) — use abs
            peak = float(np.abs(vals).max() + 1e-12)
            w[mask] = np.abs(vals) / peak
        else:
            w[mask] = (vals / peak).astype(np.float32)
        profiles.append(w)
    return profiles


# ─────────────────────────────────────────────────────────────────────────
# Step 2 — Simultaneous trace estimation (GPU, chunked)
# ─────────────────────────────────────────────────────────────────────────

def _build_union_design(
    profiles: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Build the (union_pixels × N_rois) design matrix W.

    Returns
    -------
    W               : (P, N) float32 — design matrix over union pixels
    union_flat_idx  : (P,) int64 — flat indices into (H*W) for the union pixels
    union_yx        : tuple of (y, x) arrays for fancy-indexing into (H, W)
    """
    if not profiles:
        raise ValueError("No profiles given.")
    H, W = profiles[0].shape
    # Union = pixels where any profile > 0
    union_mask = np.zeros((H, W), dtype=bool)
    for p in profiles:
        union_mask |= (p > 0)
    union_flat_idx = np.flatnonzero(union_mask.ravel()).astype(np.int64)
    union_y, union_x = np.where(union_mask)
    P = union_flat_idx.size
    N = len(profiles)
    design = np.zeros((P, N), dtype=np.float32)
    for i, prof in enumerate(profiles):
        design[:, i] = prof.ravel()[union_flat_idx]
    return design, union_flat_idx, (union_y, union_x)


def solve_traces_from_chunks(
    design: np.ndarray,
    T: int,
    chunk_iter,
    cfg: PipelineConfig,
) -> np.ndarray:
    """Shared normal-equations solver for simultaneous trace estimation.

    At each time chunk, solve:
        c_chunk = (W.T W + λI)⁻¹ W.T S_chunk.T

    Parameters
    ----------
    design     : (P, N) float32 — design matrix over union pixels.
    T          : total number of frames (for output allocation).
    chunk_iter : iterable yielding (t0, t1, S_chunk) tuples where
                 S_chunk is (cs, P) float32 — the already-union-indexed
                 residual slab for frames [t0, t1). Callers provide this so
                 the solver is decoupled from dtype and storage (works for
                 float32 residuals, int16 data.bin, in-memory arrays, etc.).
    cfg        : PipelineConfig (uses subtract_ridge_lambda_scale).

    Returns
    -------
    traces : (N, T) float32
    """
    import torch

    N = design.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    W_t = torch.from_numpy(design).to(device)                # (P, N)
    WtW = W_t.T @ W_t                                         # (N, N)
    lam = cfg.subtract_ridge_lambda_scale * (
        WtW.diagonal().sum().item() / max(N, 1)
    )
    WtW_reg = WtW + lam * torch.eye(N, device=device, dtype=WtW.dtype)

    traces = np.empty((N, T), dtype=np.float32)

    for t0, t1, S_chunk in chunk_iter:
        b_t = torch.from_numpy(np.ascontiguousarray(S_chunk.T)).to(device)  # (P, cs)
        rhs = W_t.T @ b_t                                                    # (N, cs)
        c_chunk = torch.linalg.solve(WtW_reg, rhs)                           # (N, cs)
        traces[:, t0:t1] = c_chunk.detach().cpu().numpy().astype(np.float32)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return traces


def estimate_traces_simultaneous(
    residual_S_path: Path,
    shape: tuple,
    profiles: list[np.ndarray],
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Estimate all traces simultaneously via GPU-chunked normal equations.

    Thin adapter around :func:`solve_traces_from_chunks` that reads a float32
    (T, H, W) memmap and yields union-indexed chunks to the shared solver.

    Returns
    -------
    traces          : (N_rois, T) float32
    union_flat_idx  : (P,) int64 for downstream subtraction
    union_yx        : (union_y, union_x) for downstream subtraction
    """
    T, H, L = shape
    design, union_flat_idx, union_yx = _build_union_design(profiles)

    S_mm = np.memmap(str(residual_S_path), dtype=np.float32, mode="r",
                     shape=(T, H, L))
    chunk = int(cfg.subtract_chunk_frames)

    def _iter():
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            cs = t1 - t0
            # Materialize the chunk in RAM first (one sequential read),
            # THEN fancy-index. np.asarray on a matching-dtype memmap returns
            # the memmap view, so the original `[:, union_flat_idx]` was doing
            # random-access reads on the file — disastrous on slow/external
            # storage. Same memmap-view pitfall as the Phase C validator.
            chunk_ram = np.array(S_mm[t0:t1], dtype=np.float32, copy=True)
            S_chunk = chunk_ram.reshape(cs, H * L)[:, union_flat_idx]
            yield t0, t1, S_chunk

    traces = solve_traces_from_chunks(design, T, _iter(), cfg)
    del S_mm
    return traces, union_flat_idx, union_yx


# ─────────────────────────────────────────────────────────────────────────
# Step 3 — Rank-1 subtraction (streaming)
# ─────────────────────────────────────────────────────────────────────────

def subtract_sources(
    residual_S_path: Path,
    residual_out_path: Path,
    shape: tuple,
    profiles: list[np.ndarray],
    traces: np.ndarray,
    chunk: int = 500,
) -> None:
    """Stream rank-1 subtraction from residual_S_path into residual_out_path.

    For each temporal chunk t in [t0, t1):
        residual_out[t] = residual_S[t] - Σ_i w_i[y_i, x_i] * c_i[t]   at mask_i pixels

    We compute this over the union of all ROI pixels in one batched operation
    per chunk using np.einsum over (N_rois, P_union) × (N_rois, cs) → (P_union, cs).

    Parameters
    ----------
    residual_S_path   : input memmap path (float32, shape)
    residual_out_path : output memmap path (will be created, float32, same shape)
    profiles          : list of N (H, W) float32 profile maps
    traces            : (N, T) float32
    """
    T, H, W = shape
    S_in = np.memmap(str(residual_S_path), dtype=np.float32, mode="r", shape=(T, H, W))

    # Create output fresh (no copyfile). On slow drives the previous design
    # spent 8.6 GB of extra write just to copy the input, then did an
    # in-place read-modify-write over the memmap union indices on top — ~34
    # GB of I/O to produce an 8.6 GB output. Stream-read → subtract in RAM →
    # stream-write instead: exactly one 8.6 GB sequential read + one 8.6 GB
    # sequential write per call.
    S_out = np.memmap(str(residual_out_path), dtype=np.float32, mode="w+", shape=(T, H, W))

    # Stack profile weights over union of ALL ROI pixels
    union_mask = np.zeros((H, W), dtype=bool)
    for p in profiles:
        union_mask |= (p > 0)
    union_flat = np.flatnonzero(union_mask.ravel())  # (P,)
    P_union = union_flat.size
    N = len(profiles)

    W_design = np.zeros((N, P_union), dtype=np.float32)
    for i, p in enumerate(profiles):
        W_design[i] = p.ravel()[union_flat]

    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        cs = t1 - t0
        # Materialize chunk in RAM (sequential read from S_in), modify in
        # place, write sequentially to S_out. Matches the same memmap-copy
        # pattern that fixed _validate_streaming and estimate_traces_*.
        chunk_ram = np.array(S_in[t0:t1], dtype=np.float32, copy=True)
        flat = chunk_ram.reshape(cs, H * W)
        sub = W_design.T @ traces[:, t0:t1]                       # (P_union, cs)
        flat[:, union_flat] -= sub.T
        S_out[t0:t1] = chunk_ram

    S_out.flush()
    del S_in, S_out


# ─────────────────────────────────────────────────────────────────────────
# Post-subtraction validation (spec §5.2)
# ─────────────────────────────────────────────────────────────────────────

def _compute_surround_mask(mask: np.ndarray, inner: int, outer: int,
                           exclude: np.ndarray) -> np.ndarray:
    """Annular ring around `mask`, excluding pixels from `exclude` (other ROIs).

    Returns a (H, W) bool mask.
    """
    from scipy.ndimage import binary_dilation
    outer_mask = binary_dilation(mask, iterations=outer)
    inner_mask = binary_dilation(mask, iterations=inner)
    ring = outer_mask & ~inner_mask & ~exclude
    return ring


def _validate_streaming(
    residual_after_path: Path,
    shape: tuple,
    rois: list[ROI],
    traces: np.ndarray,
    cfg: PipelineConfig,
    subset_indices: Optional[list[int]] = None,
) -> dict:
    """Single streaming-pass validator (Phase C.1).

    Replaces the 5N-memmap-scan hotpath of the original ``validate_subtraction``
    with one temporal pass over ``(T, H, W)`` that accumulates float64 moments
    for every ROI simultaneously. Disk reads drop from 5N·(T·n_pix) to one
    sequential pass of the full (T, H, W) float32 memmap; this is the phase
    C.1 win documented in the plan.

    When ``subset_indices`` is provided, only those ROIs (indices into
    ``rois``/``traces``) are validated — used by the post-NNLS re-validate so
    unflagged entries in the first-pass dict can be reused (Phase C.2).

    Returns
    -------
    dict {roi_label_id: {mean_ratio, std_ratio, anticorr_max, pass}}
    """
    T, H, W = shape
    N = len(rois)
    if N == 0 or T <= 0:
        return {}

    indices = list(range(N)) if subset_indices is None else list(subset_indices)
    if not indices:
        return {}

    # Surround exclusion uses the union of *all* ROI masks (not just the subset)
    # so a flagged ROI's annulus still avoids unflagged neighbors. Matches
    # the original lines 353-355.
    all_union = np.zeros((H, W), dtype=bool)
    for r in rois:
        all_union |= r.mask

    mask_flat: dict[int, np.ndarray] = {}
    surround_flat: dict[int, np.ndarray] = {}
    n_m: dict[int, int] = {}
    n_s: dict[int, int] = {}
    empty_mask: set = set()
    for i in indices:
        roi = rois[i]
        mf = np.flatnonzero(roi.mask.ravel()).astype(np.int64)
        mask_flat[i] = mf
        n_m[i] = int(mf.size)
        if mf.size == 0:
            empty_mask.add(i)
            surround_flat[i] = np.empty(0, dtype=np.int64)
            n_s[i] = 0
            continue
        others = all_union & ~roi.mask
        ring = _compute_surround_mask(
            roi.mask,
            inner=cfg.annulus_inner_buffer,
            outer=cfg.annulus_outer_radius,
            exclude=others,
        )
        sf = np.flatnonzero(ring.ravel()).astype(np.int64)
        surround_flat[i] = sf
        n_s[i] = int(sf.size)

    # Trace moments depend only on `traces` — precompute once outside the loop.
    sum_b: dict[int, float] = {}
    sum_b2: dict[int, float] = {}
    for i in indices:
        tr64 = traces[i].astype(np.float64, copy=False)
        sum_b[i] = float(tr64.sum())
        sum_b2[i] = float((tr64 * tr64).sum())

    # Float64 accumulators — required to avoid catastrophic cancellation across
    # ~10^6 samples (8624 frames × ~200 mask pixels on the reference FOV).
    sum_m = {i: 0.0 for i in indices}
    sumsq_m = {i: 0.0 for i in indices}
    sum_s = {i: 0.0 for i in indices}
    sumsq_s = {i: 0.0 for i in indices}
    sum_a = {i: 0.0 for i in indices}
    sum_a2 = {i: 0.0 for i in indices}
    sum_ab = {i: 0.0 for i in indices}

    S_mm = np.memmap(str(residual_after_path), dtype=np.float32, mode="r",
                     shape=(T, H, W))
    chunk = max(int(cfg.subtract_chunk_frames), 1)
    try:
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            cs = t1 - t0
            # Force a contiguous in-RAM copy. np.asarray / np.array with
            # copy=False would keep flat as a memmap view, so every
            # flat[:, mask_flat[i]] fancy-index below would re-read disk —
            # defeating the whole point of streaming.
            flat = np.array(S_mm[t0:t1], dtype=np.float32, copy=True).reshape(cs, H * W)
            for i in indices:
                if i in empty_mask:
                    continue
                mvals = flat[:, mask_flat[i]].astype(np.float64, copy=False)
                sum_m[i] += float(mvals.sum())
                sumsq_m[i] += float((mvals * mvals).sum())

                a_chunk = mvals.mean(axis=1)                                 # (cs,) float64
                b_chunk = traces[i, t0:t1].astype(np.float64, copy=False)    # (cs,)
                sum_a[i] += float(a_chunk.sum())
                sum_a2[i] += float((a_chunk * a_chunk).sum())
                sum_ab[i] += float((a_chunk * b_chunk).sum())

                sf = surround_flat[i]
                if sf.size:
                    svals = flat[:, sf].astype(np.float64, copy=False)
                    sum_s[i] += float(svals.sum())
                    sumsq_s[i] += float((svals * svals).sum())
    finally:
        del S_mm

    report: dict = {}
    T_f = float(T)
    for i in indices:
        roi = rois[i]
        if i in empty_mask:
            # Match pre-C.1 behavior: empty mask → zeros everywhere → fails
            # the std_ratio > 0.3 check → pass=False.
            report[int(roi.label_id)] = {
                "mean_ratio": 0.0,
                "std_ratio": 0.0,
                "anticorr_max": 0.0,
                "pass": False,
            }
            continue

        nm = n_m[i]
        mean_mask = sum_m[i] / (nm * T_f)
        var_mask = max(sumsq_m[i] / (nm * T_f) - mean_mask * mean_mask, 0.0)
        std_mask = var_mask ** 0.5

        if n_s[i] > 0:
            ns = n_s[i]
            mean_surr = sum_s[i] / (ns * T_f)
            var_surr = max(sumsq_s[i] / (ns * T_f) - mean_surr * mean_surr, 0.0)
            std_surr = var_surr ** 0.5
        else:
            # Empty annulus → fall back to mask stats (matches the original
            # else branch at lines 373-375).
            mean_surr = mean_mask
            std_surr = std_mask

        mean_ratio = abs(mean_mask) / (abs(mean_surr) + 1e-6)
        std_ratio = std_mask / (std_surr + 1e-6)

        # Pearson via second moments:
        #   r = (T·Σab − Σa·Σb) / sqrt((T·Σa² − (Σa)²)·(T·Σb² − (Σb)²))
        # Equivalent to the original `(a−ā)(b−b̄)/(σa σb)` formulation at
        # subtraction.py:383-386, with the same 1e-12 denom guard.
        num = T_f * sum_ab[i] - sum_a[i] * sum_b[i]
        var_a = max(T_f * sum_a2[i] - sum_a[i] * sum_a[i], 0.0)
        var_b = max(T_f * sum_b2[i] - sum_b[i] * sum_b[i], 0.0)
        denom_prod = var_a * var_b
        if denom_prod > 0.0:
            anticorr = num / (denom_prod ** 0.5 + 1e-12)
        else:
            anticorr = 0.0

        passed = (
            (mean_ratio < 3.0)
            and (0.3 < std_ratio < 3.0)
            and (anticorr > cfg.subtract_anticorr_threshold)
        )

        report[int(roi.label_id)] = {
            "mean_ratio": float(mean_ratio),
            "std_ratio": float(std_ratio),
            "anticorr_max": float(anticorr),
            "pass": bool(passed),
        }

    return report


def validate_subtraction(
    residual_before_path: Path,
    residual_after_path: Path,
    shape: tuple,
    rois: list[ROI],
    traces: np.ndarray,
    cfg: PipelineConfig,
) -> dict:
    """Per-ROI subtraction validation (spec §5.2).

    Three checks per ROI:
      1. mean_ratio: |mean(after[mask])| / |mean(after[surround])|
         Near 1.0 = no halo/depression; far from 1.0 = artifact.
      2. std_ratio:  std(after[mask]) / std(after[surround])
         Near 1.0 = matched noise; >> 1 = residual activity, << 1 = over-subtracted.
      3. anticorr_max: Pearson corr between trace[mask in AFTER] and removed trace.
         Should not be strongly negative (indicates over-subtraction into noise).

    Returns
    -------
    dict {roi_label_id: {mean_ratio, std_ratio, anticorr_max, pass}}
    """
    # `residual_before_path` is in the signature for API stability but neither
    # the current streaming implementation nor the prior per-ROI loop reads it.
    return _validate_streaming(residual_after_path, shape, rois, traces, cfg)


# ─────────────────────────────────────────────────────────────────────────
# NNLS fallback for flagged ROIs (bounded scope per Plan agent A3)
# ─────────────────────────────────────────────────────────────────────────

def _nnls_refine_flagged(
    residual_before_path: Path,
    residual_after_path: Path,
    shape: tuple,
    rois: list[ROI],
    profiles: list[np.ndarray],
    traces: np.ndarray,
    flagged_indices: list[int],
    cfg: PipelineConfig,
) -> np.ndarray:
    """Re-estimate traces for flagged ROIs only, over their LOCAL pixel support.

    For each flagged ROI i:
      - pull S_before at mask_i pixels: shape (T, P_i) where P_i ≈ 100-300
      - solve NNLS per frame: profile_i[mask_i] × c_i(t) ≈ S(mask_i, t)
        (single-variable NNLS reduces to max(0, lstsq(w, s)))
      - replace traces[i] with the refined trace
    This is cheap: ~ms per frame, ~seconds per ROI total.

    Returns updated traces (N, T).
    """
    from scipy.optimize import nnls

    T, H, W = shape
    S_before = np.memmap(str(residual_before_path), dtype=np.float32, mode="r",
                         shape=(T, H, W))

    traces_refined = traces.copy()
    for idx in flagged_indices:
        roi = rois[idx]
        prof = profiles[idx]
        ys, xs = np.where(roi.mask)
        w = prof[ys, xs]   # (P_i,)
        P_i = w.size
        if P_i == 0 or w.sum() <= 0:
            continue
        # Single-variable NNLS with design vector w has closed form:
        #   c(t) = max(0, (w.T @ s_t) / (w.T @ w))
        wtw = float(w @ w)
        S_local = S_before[:, ys, xs]  # (T, P_i)
        c_raw = (S_local @ w) / wtw
        traces_refined[idx] = np.maximum(c_raw, 0.0).astype(np.float32)

    del S_before
    return traces_refined


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────

def compute_std_map(
    residual_path: Path,
    shape: tuple,
    chunk: int = 500,
) -> np.ndarray:
    """Per-pixel std of a (T, H, W) float32 memmap, streamed in temporal chunks.

    Used as the profile_source for Stage 2+ subtractions. After earlier stages
    have subtracted their detected neurons, the surviving per-pixel variance
    highlights the neurons that THIS stage will discover — so we must
    recompute, not reuse std from the original residual (see D2 in the plan).
    """
    T, H, W = shape
    mm = np.memmap(str(residual_path), dtype=np.float32, mode="r", shape=(T, H, W))
    sum_ = np.zeros((H, W), dtype=np.float64)
    sumsq = np.zeros((H, W), dtype=np.float64)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        c = np.asarray(mm[t0:t1], dtype=np.float64)
        sum_ += c.sum(axis=0)
        sumsq += (c ** 2).sum(axis=0)
    del mm
    mean = sum_ / T
    var = np.maximum(sumsq / T - mean ** 2, 0.0)
    return np.sqrt(var).astype(np.float32)


def run_source_subtraction(
    residual_S_path: Path,
    shape: tuple,
    profile_source: np.ndarray,
    rois: list[ROI],
    output_dir: Path,
    cfg: PipelineConfig,
    output_name: str = "residual_S1",
    delete_input: bool = False,
) -> tuple[Path, dict, np.ndarray]:
    """Full subtraction pipeline: profiles → traces → subtract → validate → NNLS fallback.

    Parameters
    ----------
    residual_S_path : input residual (T, H, W) float32 memmap
    shape           : (T, H, W)
    profile_source  : (H, W) spatial activity map for profile estimation
                      (typically std_S; see estimate_spatial_profiles docstring)
    rois            : ROIs to subtract (typically accepted + flagged from Gate 1)
    output_dir      : where to write outputs
    cfg             : PipelineConfig
    output_name     : base name for output files. Default 'residual_S1' preserves
                      Phase 1B behavior; pass 'residual_S2' / 'residual_S3' for
                      subsequent stages to avoid filename collisions.
    delete_input    : if True, unlink residual_S_path after validation + NNLS
                      complete successfully. The caller is responsible for
                      ensuring nothing downstream still references the input.

    Returns
    -------
    residual_out_path   : Path to (T, H, W) float32 memmap of post-subtraction residual
    validation_report   : dict keyed by ROI label_id
    traces              : (N, T) float32 — estimated traces (for roi.trace population)
    """
    residual_out_path = output_dir / f"{output_name}.dat"
    meta_path = output_dir / f"{output_name}.meta.json"
    report_path = output_dir / f"subtraction_report_{output_name}.json"

    if not rois:
        # No ROIs to subtract; copy input → output unchanged
        shutil.copyfile(str(residual_S_path), str(residual_out_path))
        meta = {"shape": list(shape), "dtype": "float32"}
        meta_path.write_text(json.dumps(meta, indent=2))
        if delete_input:
            Path(residual_S_path).unlink(missing_ok=True)
        return residual_out_path, {}, np.zeros((0, shape[0]), dtype=np.float32)

    # Step 1: profiles
    t0 = time.time()
    profiles = estimate_spatial_profiles(profile_source, rois)
    print(f"  profiles: {len(profiles)} ROIs in {time.time()-t0:.2f}s", flush=True)

    # Step 2: simultaneous traces via GPU-chunked normal equations
    t0 = time.time()
    traces, _union_idx, _union_yx = estimate_traces_simultaneous(
        residual_S_path, shape, profiles, cfg,
    )
    print(f"  traces: {traces.shape} via GPU lstsq in {time.time()-t0:.2f}s", flush=True)

    # Step 3: rank-1 subtract (streaming, writes residual_out_path)
    t0 = time.time()
    subtract_sources(residual_S_path, residual_out_path, shape, profiles, traces,
                     chunk=cfg.reconstruct_chunk)
    meta = {"shape": list(shape), "dtype": "float32"}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  rank-1 subtract → {residual_out_path.name} in {time.time()-t0:.2f}s", flush=True)

    # Step 4: validate
    t0 = time.time()
    validation = validate_subtraction(
        residual_S_path, residual_out_path, shape, rois, traces, cfg,
    )
    n_fail = sum(1 for v in validation.values() if not v["pass"])
    n_anticorr = sum(1 for v in validation.values()
                      if v["anticorr_max"] < cfg.subtract_anticorr_threshold)
    print(f"  validate: {len(rois)-n_fail}/{len(rois)} passed "
          f"({n_anticorr} anticorr flags) in {time.time()-t0:.2f}s", flush=True)

    # Step 5: NNLS fallback on flagged ROIs if anticorrelation failure rate > threshold
    failure_frac = n_anticorr / max(len(rois), 1)
    if failure_frac > cfg.subtract_anticorr_failure_fraction:
        flagged_indices = [
            i for i, r in enumerate(rois)
            if validation.get(int(r.label_id), {}).get("anticorr_max", 0.0)
               < cfg.subtract_anticorr_threshold
        ][: cfg.subtract_nnls_fallback_max_rois]

        if flagged_indices:
            t0 = time.time()
            print(f"  NNLS fallback on {len(flagged_indices)} anticorr-flagged ROIs "
                  f"(failure_frac={failure_frac:.1%})", flush=True)
            traces = _nnls_refine_flagged(
                residual_S_path, residual_out_path, shape, rois, profiles,
                traces, flagged_indices, cfg,
            )
            # Re-subtract with refined traces
            subtract_sources(residual_S_path, residual_out_path, shape, profiles, traces,
                             chunk=cfg.reconstruct_chunk)
            # Only flagged ROIs' traces changed, so re-validate just those and
            # merge into the first-pass dict. Unflagged entries remain correct.
            flagged_update = _validate_streaming(
                residual_out_path, shape, rois, traces, cfg,
                subset_indices=flagged_indices,
            )
            validation.update(flagged_update)
            n_fail = sum(1 for v in validation.values() if not v["pass"])
            print(f"  post-NNLS: {len(rois)-n_fail}/{len(rois)} passed "
                  f"in {time.time()-t0:.2f}s", flush=True)

    # Persist validation report
    report_path.write_text(json.dumps(validation, indent=2))

    # Drop the input residual now that all passes that needed it (validation,
    # NNLS fallback, re-subtract) are complete. Keeps disk at ~1 residual.
    if delete_input:
        Path(residual_S_path).unlink(missing_ok=True)

    return residual_out_path, validation, traces
