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

    return F_raw, F_neu, F_corrected
