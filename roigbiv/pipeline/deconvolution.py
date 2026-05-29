"""
ROI G. Biv pipeline — Spike Deconvolution (spec §13.6).

Applies Suite2p's OASIS deconvolution to phasic and sparse ROIs only.
Tonic, silent, and ambiguous ROIs receive NaN rows — the discrete-spike-
on-stable-baseline model does not apply to them; surfacing NaNs prevents
downstream consumers from treating the result as real.

Optional step: if Suite2p's dcnv module is unavailable, the pipeline logs
a warning and continues with an all-NaN spks array.
"""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline.types import ROI


def _try_import_dcnv():
    try:
        from suite2p.extraction import dcnv  # type: ignore
        return dcnv
    except Exception:
        return None


def _deconvolve_one(trace: np.ndarray, dcnv, tau: float, fs: float) -> np.ndarray:
    """Thin wrapper around dcnv.oasis handling shape and fallbacks."""
    x = np.asarray(trace, dtype=np.float32).reshape(1, -1)
    try:
        spks = dcnv.oasis(x, batch_size=1, tau=tau, fs=fs)
        return np.asarray(spks, dtype=np.float32).ravel()
    except Exception:
        return np.full(trace.shape, np.nan, dtype=np.float32)


def deconvolve_traces(
    dFF: np.ndarray,
    rois: list[ROI],
    tau: float,
    fs: float,
) -> np.ndarray:
    """Return (N, T) float32 spks array. NaN rows for non-phasic/sparse ROIs.

    Row ordering matches the input `rois` (already sorted by label_id
    at the caller).
    """
    N, T = dFF.shape
    spks = np.full((N, T), np.nan, dtype=np.float32)
    if N == 0 or T == 0:
        return spks

    dcnv = _try_import_dcnv()
    if dcnv is None:
        print("WARNING: suite2p.extraction.dcnv unavailable; spks = NaN",
              flush=True)
        return spks

    ok = 0
    for i, roi in enumerate(rois):
        if roi.activity_type not in ("phasic", "sparse"):
            continue
        dff_row = dFF[i]
        if not np.all(np.isfinite(dff_row)):
            continue
        try:
            spks[i] = _deconvolve_one(dff_row, dcnv, tau, fs)
            ok += 1
        except Exception:
            spks[i] = np.nan

    print(f"Deconvolution: {ok}/{N} ROIs (phasic/sparse only)", flush=True)
    return spks
