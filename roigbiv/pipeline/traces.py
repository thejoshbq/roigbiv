"""
ROI G. Biv pipeline — Trace Extraction (spec §13.2).

Extracts raw fluorescence and neuropil traces from the ORIGINAL registered
movie (Suite2p data.bin, int16). Residuals are unsuitable for final trace
extraction because they have had earlier stages' ROIs subtracted out.

Pipeline:
  1. build_neuropil_masks — annular rings around each ROI with cross-ROI
     exclusion (so neighboring cells do not contaminate each other's neuropil).
  2. extract_mean_trace_chunked — temporal-chunked mean over mask pixels. One
     pass over data.bin services both raw and neuropil (stacked masks).
  3. correct_neuropil — F_corrected = F_raw - α × F_neuropil.

Reads from fov.data_bin_path (Suite2p-format int16 (T, Ly, Lx) memmap). Casts
to float32 per chunk so the full 18 GB movie never lives in RAM.

Median and mode (extract_median_mode_traces_chunked / extract_all_traces_full)
are a separate, opt-in extraction path — mean's per-frame spatial average is a
linear operator (a matmul against a dense mask matrix), but median and mode
are not, so they cannot reuse that trick and instead gather each mask's pixels
directly out of every chunk. They get the same neuropil-correction treatment
as mean (stat_corrected = stat(ROI px) - α·stat(neuropil px), using that
statistic's own aggregate), but NOT overlap correction
(roigbiv.pipeline.overlap_correction is a linear least-squares demixing
solver — meaningful only for the mean). This path is only invoked from the
Discovery page's on-demand extraction (roigbiv.pipeline.discovery_extract);
the automatic per-FOV pipeline run stays mean-only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

from roigbiv.pipeline.types import FOVData, ROI, PipelineConfig


def _union_of_masks(masks: list[np.ndarray], shape_hw: tuple) -> np.ndarray:
    H, W = shape_hw
    union = np.zeros((H, W), dtype=bool)
    for m in masks:
        union |= m
    return union


def build_neuropil_masks(
    roi_masks: list[np.ndarray],
    shape_hw: tuple,
    inner_buffer: int,
    outer_radius: int,
) -> list[np.ndarray]:
    """Build an annular neuropil mask for every ROI.

    For each ROI:
      outer_disk = dilate(mask, inner_buffer + outer_radius)
      inner_disk = dilate(mask, inner_buffer)
      annulus    = outer_disk & ~inner_disk & ~union_of_ALL_other_ROIs

    The exclusion union is built ONCE and reused — critical for correctness
    (a neuron in a cluster would otherwise pull neighbor signal into its
    neuropil estimate).

    If the excluded annulus is empty, fall back to widening outer_radius by
    +5 px once; if still empty, use the un-excluded annulus and emit a warning
    via the caller's aggregator (returned as None is NOT used — we always
    return a boolean mask so caller does not have to branch).
    """
    H, W = shape_hw
    n = len(roi_masks)
    if n == 0:
        return []

    # Use scipy's binary_dilation with iterations for consistency with
    # existing gate1 / subtraction code paths (same geometry).
    union_all = _union_of_masks(roi_masks, (H, W))

    annuli: list[np.ndarray] = []
    for i, mask in enumerate(roi_masks):
        others = union_all & ~mask
        outer = binary_dilation(mask, iterations=inner_buffer + outer_radius)
        inner = binary_dilation(mask, iterations=inner_buffer)
        annulus = outer & ~inner & ~others

        if not annulus.any():
            # Fallback: widen outer radius
            outer2 = binary_dilation(mask, iterations=inner_buffer + outer_radius + 5)
            annulus = outer2 & ~inner & ~others
            if not annulus.any():
                # Last resort: drop the cross-ROI exclusion (contamination risk
                # logged via zero mask area below, caller may set the trace to 0).
                annulus = outer & ~inner
        annuli.append(annulus)
    return annuli


def extract_mean_trace_chunked(
    memmap_path: Path,
    shape: tuple,
    dtype: np.dtype,
    masks: list[np.ndarray],
    chunk: int = 500,
) -> np.ndarray:
    """Stream a (T, H, W) memmap in temporal chunks, returning per-mask mean traces.

    Parameters
    ----------
    memmap_path : path to raw binary memmap
    shape       : (T, H, W)
    dtype       : memmap dtype (np.int16 for Suite2p data.bin, np.float32
                  for residuals)
    masks       : list of N (H, W) bool arrays
    chunk       : frames per iteration

    Returns
    -------
    traces : (N, T) float32
    """
    T, H, W = shape
    N = len(masks)
    if N == 0:
        return np.zeros((0, T), dtype=np.float32)

    # Dense (N, H*W) float32 mask matrix — same pattern as stage2.extract_traces_from_residual.
    M = np.zeros((N, H * W), dtype=np.float32)
    mask_sizes = np.zeros(N, dtype=np.float32)
    for i, m in enumerate(masks):
        flat = m.ravel().astype(np.float32)
        M[i] = flat
        mask_sizes[i] = flat.sum()
    # Guard division by zero for empty annuli
    mask_sizes_safe = np.where(mask_sizes > 0, mask_sizes, 1.0)

    traces = np.empty((N, T), dtype=np.float32)
    mm = np.memmap(str(memmap_path), dtype=dtype, mode="r", shape=(T, H, W))
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        cs = t1 - t0
        flat_chunk = np.asarray(mm[t0:t1], dtype=np.float32).reshape(cs, H * W)
        traces[:, t0:t1] = (M @ flat_chunk.T) / mask_sizes_safe[:, None]
    # Zero out traces for empty masks (avoid NaN from fake division)
    empty_idx = np.where(mask_sizes == 0)[0]
    if empty_idx.size:
        traces[empty_idx] = 0.0
    del mm
    return traces


def correct_neuropil(
    F_raw: np.ndarray,
    F_neu: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """F_corrected = F_raw - α × F_neuropil (spec §13.2)."""
    return (F_raw - alpha * F_neu).astype(np.float32)


# ── median / mode (opt-in, Discovery-triggered) ────────────────────────────

_MAX_MODE_BINS = 8192


def _estimate_value_range(
    memmap_path: Path,
    shape: tuple,
    dtype: np.dtype,
    *,
    sample_stride: int = 50,
    pad: int = 4,
) -> tuple[int, int]:
    """Cheap strided sample of data.bin's value range, for mode's bincount sizing.

    Reads every ``sample_stride``-th frame only — a full scan would double
    the cost of an already two-pass (median + mode) extraction. ``pad``
    guards against the sample missing the true extremes; the caller still
    clips to this band, so an under-estimate only costs accuracy on rare
    outlier pixels, never correctness.
    """
    T, H, W = shape
    mm = np.memmap(str(memmap_path), dtype=dtype, mode="r", shape=(T, H, W))
    sample = np.asarray(mm[::sample_stride])
    del mm
    if sample.size == 0:
        return 0, 1
    return int(sample.min()) - pad, int(sample.max()) + pad


def _mode_via_bincount(sub: np.ndarray, vmin: int, nbins: int) -> np.ndarray:
    """Per-row (per-frame) mode of ``sub`` (frames × pixels), vectorized.

    Pixel intensities are native int16 — an exact integer mode, no
    continuous-data binning ambiguity. Encodes (frame, clipped value) pairs
    into one flat key so a single ``np.bincount`` produces every frame's
    histogram at once; ``scipy.stats.mode`` would need one call per
    (mask, chunk) pair and is not fast enough at that call count.
    """
    cs, k = sub.shape
    if k == 0:
        return np.zeros(cs, dtype=np.float32)
    clipped = np.clip(sub, vmin, vmin + nbins - 1) - vmin
    keys = (np.arange(cs, dtype=np.int64)[:, None] * nbins + clipped).ravel()
    counts = np.bincount(keys, minlength=cs * nbins).reshape(cs, nbins)
    return (counts.argmax(axis=1) + vmin).astype(np.float32)


def extract_median_mode_traces_chunked(
    memmap_path: Path,
    shape: tuple,
    dtype: np.dtype,
    masks: list[np.ndarray],
    *,
    stats: tuple[str, ...] = ("median", "mode"),
    chunk: int = 500,
    value_range: Optional[tuple[int, int]] = None,
) -> dict[str, np.ndarray]:
    """Stream a (T, H, W) memmap in temporal chunks, returning per-mask
    median and/or mode traces (whichever ``stats`` names).

    Unlike ``extract_mean_trace_chunked``, mean is a linear operator; median
    and mode are not, so each mask's pixels are gathered directly out of
    every chunk instead of via a dense matmul. Each mask's flat pixel-index
    array is precomputed once, keeping this a single pass over the memmap.

    Only requested ``stats`` do work: the mode's value-range sample and
    bincount pass are skipped entirely when only ``"median"`` is asked for.

    Assumes integer-valued input (int16 Suite2p data.bin) when ``"mode"`` is
    requested — mode's bincount encoding does not support float movies.

    Returns
    -------
    dict with ``"median"`` and/or ``"mode"`` keys, each (N, T) float32.
    """
    T, H, W = shape
    N = len(masks)
    want_median = "median" in stats
    want_mode = "mode" in stats

    if N == 0:
        empty = np.zeros((0, T), dtype=np.float32)
        out: dict[str, np.ndarray] = {}
        if want_median:
            out["median"] = empty
        if want_mode:
            out["mode"] = empty.copy()
        return out

    idx_by_mask = [np.flatnonzero(m.ravel()) for m in masks]

    vmin = nbins = 0
    if want_mode:
        vmin, vmax = (value_range if value_range is not None
                       else _estimate_value_range(memmap_path, shape, dtype))
        nbins = max(1, min(int(vmax - vmin + 1), _MAX_MODE_BINS))

    median_traces = np.empty((N, T), dtype=np.float32) if want_median else None
    mode_traces = np.empty((N, T), dtype=np.float32) if want_mode else None

    mm = np.memmap(str(memmap_path), dtype=dtype, mode="r", shape=(T, H, W))
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        flat_chunk = np.asarray(mm[t0:t1], dtype=np.int32).reshape(t1 - t0, H * W)
        for i, idx in enumerate(idx_by_mask):
            if idx.size == 0:
                if want_median:
                    median_traces[i, t0:t1] = 0.0
                if want_mode:
                    mode_traces[i, t0:t1] = 0.0
                continue
            sub = flat_chunk[:, idx]
            if want_median:
                median_traces[i, t0:t1] = np.median(sub, axis=1)
            if want_mode:
                mode_traces[i, t0:t1] = _mode_via_bincount(sub, vmin, nbins)
    del mm

    out = {}
    if want_median:
        out["median"] = median_traces
    if want_mode:
        out["mode"] = mode_traces
    return out


def extract_all_traces_full(
    fov: FOVData,
    rois: list[ROI],
    cfg: PipelineConfig,
    *,
    stats: tuple[str, ...] = ("median", "mode"),
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract mean plus optional median/mode statistics, each independently
    neuropil-corrected.

    Mean is always included via the existing ``extract_all_traces`` (does not
    mutate ``rois`` any differently than that function already does).
    Requested extra ``stats`` get their own correction pass — see the module
    docstring for why overlap correction does not apply to them.

    ``stats`` may contain ``"median"``, ``"mode"``, or both; ``"mean"`` is
    implicit and does not need to be named.

    Returns
    -------
    dict keyed by statistic name → ``(F_raw, F_neu, F_corrected)``, each
    ``(N_rois, T)`` float32. Always contains ``"mean"``.
    """
    extra_stats = tuple(s for s in stats if s != "mean")
    for name in extra_stats:
        if name not in ("median", "mode"):
            raise ValueError(
                f"unknown trace statistic {name!r}; expected 'median' or 'mode'")

    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "mean": extract_all_traces(fov, rois, cfg),
    }
    if not extra_stats:
        return result

    T, H, W = fov.shape
    n = len(rois)
    if n == 0:
        empty = np.zeros((0, T), dtype=np.float32)
        for name in extra_stats:
            result[name] = (empty, empty.copy(), empty.copy())
        return result

    roi_masks = [r.mask for r in rois]
    neuropil_masks = build_neuropil_masks(
        roi_masks, (H, W), cfg.neuropil_inner_buffer, cfg.neuropil_outer_radius)
    all_masks = roi_masks + neuropil_masks

    traces_by_stat = extract_median_mode_traces_chunked(
        fov.data_bin_path, fov.shape, np.int16, all_masks, stats=extra_stats,
    )
    for name in extra_stats:
        stat_raw = traces_by_stat[name][:n]
        stat_neu = traces_by_stat[name][n:]
        stat_corrected = correct_neuropil(stat_raw, stat_neu, cfg.neuropil_coeff)
        result[name] = (stat_raw, stat_neu, stat_corrected)

    return result


def extract_all_traces(
    fov: FOVData,
    rois: list[ROI],
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orchestrator for trace extraction on the original registered movie.

    Reads fov.data_bin_path (Suite2p int16). Stores roi.trace = F_raw[i]
    and roi.trace_corrected = F_corrected[i] on each ROI. Also saves
    F_neu[i] on the ROI's features dict as roi.features['F_neuropil'] so
    downstream code can audit.

    Returns
    -------
    F_raw, F_neu, F_corrected : each (N_rois, T) float32
    """
    T, H, W = fov.shape
    n = len(rois)
    if n == 0:
        empty = np.zeros((0, T), dtype=np.float32)
        return empty, empty.copy(), empty.copy()

    roi_masks = [r.mask for r in rois]
    neuropil_masks = build_neuropil_masks(
        roi_masks,
        (H, W),
        cfg.neuropil_inner_buffer,
        cfg.neuropil_outer_radius,
    )

    # Extract ROI + neuropil traces in a single pass over data.bin by stacking
    # both mask lists into one call. Halves the memmap I/O cost vs two passes.
    all_masks = roi_masks + neuropil_masks
    all_traces = extract_mean_trace_chunked(
        fov.data_bin_path,
        fov.shape,
        dtype=np.int16,
        masks=all_masks,
        chunk=500,
    )
    F_raw = all_traces[:n]
    F_neu = all_traces[n:]

    F_corrected = correct_neuropil(F_raw, F_neu, cfg.neuropil_coeff)

    # Populate ROI objects
    for i, roi in enumerate(rois):
        roi.trace = F_raw[i].astype(np.float32, copy=True)
        roi.trace_corrected = F_corrected[i].astype(np.float32, copy=True)
        # Keep the raw neuropil trace on the ROI so QC can compute a
        # neuropil-relative baseline (Phase 5a). Stored as an ndarray feature;
        # _jsonable_features drops it from JSON like trace_bandpass.
        roi.features["F_neuropil"] = F_neu[i].astype(np.float32, copy=True)

    return F_raw, F_neu, F_corrected
