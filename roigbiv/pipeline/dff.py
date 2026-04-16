"""
ROI G. Biv pipeline — dF/F Computation (spec §13.5).

Activity-type-aware baselines:
  phasic, sparse, ambiguous → 60 s sliding 10th percentile
  tonic                     → 120 s (wider — a tonic neuron's elevated
                              baseline would otherwise be tracked too
                              closely and compress the dF/F amplitude)
  silent                    → no dF/F (trace is flat noise; NaN row)

Formula: dF/F = (F_corrected - F0) / F0, guarded against near-zero F0.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter

from roigbiv.pipeline.types import ROI, PipelineConfig


def compute_sliding_baseline(
    trace: np.ndarray,
    window_frames: int,
    percentile: int,
) -> np.ndarray:
    """Running-percentile baseline via scipy.ndimage.percentile_filter."""
    trace = np.asarray(trace, dtype=np.float32)
    if trace.size == 0 or window_frames <= 1:
        return trace.copy()
    w = min(window_frames, trace.size)
    return percentile_filter(trace, percentile=percentile, size=w,
                             mode="nearest").astype(np.float32)


def compute_dff(
    trace_corrected: np.ndarray,
    activity_type: str,
    fs: float,
    cfg: PipelineConfig,
) -> np.ndarray:
    """Compute dF/F for one trace. Returns all-NaN array for silent cells."""
    trace = np.asarray(trace_corrected, dtype=np.float32)
    if activity_type == "silent":
        return np.full_like(trace, np.nan, dtype=np.float32)

    if activity_type == "tonic":
        window_s = cfg.tonic_baseline_window_s
    else:
        window_s = cfg.baseline_window_s

    window_frames = max(2, int(round(window_s * fs)))
    F0 = compute_sliding_baseline(trace, window_frames, cfg.baseline_percentile)

    eps = 1e-6
    dff = np.zeros_like(trace, dtype=np.float32)
    safe = np.abs(F0) >= eps
    dff[safe] = (trace[safe] - F0[safe]) / F0[safe]
    # Leave unsafe entries at 0 (division-by-zero would blow up)
    return dff.astype(np.float32)


def compute_all_dff(
    rois: list[ROI],
    fs: float,
    cfg: PipelineConfig,
) -> np.ndarray:
    """Compute dF/F for every ROI. Returns (N, T) float32 with NaN silent rows.

    Order of rows matches the ROI list (which the runner has already sorted
    by label_id).
    """
    if not rois:
        return np.zeros((0, 0), dtype=np.float32)

    # Infer T from first non-None corrected trace
    T = 0
    for r in rois:
        if r.trace_corrected is not None:
            T = len(r.trace_corrected)
            break
    if T == 0:
        return np.zeros((len(rois), 0), dtype=np.float32)

    dff = np.zeros((len(rois), T), dtype=np.float32)
    for i, roi in enumerate(rois):
        if roi.trace_corrected is None:
            dff[i] = np.nan
            continue
        dff[i] = compute_dff(roi.trace_corrected, roi.activity_type or "ambiguous",
                             fs, cfg)
    return dff
