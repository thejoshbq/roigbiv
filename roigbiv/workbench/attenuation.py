"""Event-window attenuation: how much of a cell's event survives background removal and source subtraction.

For a footprint, ``M`` is its mean trace in the registered movie, ``L`` the same mean of the low-rank background
(``US_k @ V_k``, the pipeline's own factors), ``S = M - L`` the residual the summary images and Stage 2 read, and
``R = S - sub`` the residual after the earlier stages' sources are subtracted (what Stage 3 searches).

Events are detected on the neuropil-corrected movie trace ``M - c * M_annulus``, so the detector does not favour
any layer. Each event's amplitude is the mean over the peak frame and its two neighbours minus the mean of a
pre-event window (0.4 to 1.2 s before the peak), read in every layer. Ratios are per event, then the median:
``retained_s = a_S / a_M`` and ``retained_r = a_R / a_M``, which can fall below 0 or exceed 1 when the background
swings against the event; ``neuropil_share = a_annulus / a_M`` separates local events from field-wide ones. This
replaces the rank-30 "variance share", which can exceed 1 for any trace.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation, percentile_filter

MAD_TO_SD = 1.4826
PRE_WINDOW_S = (1.2, 0.4)  # pre-event baseline, seconds before the peak (start, end)
PEAK_HALF_FRAMES = 1


@dataclass(frozen=True)
class RegionLayers:
    """One region's mean traces: movie, low-rank background and subtracted sources, each (T,)."""

    m: np.ndarray
    l: np.ndarray  # noqa: E741  (the background layer's name in the pipeline's L + S split)
    sub: np.ndarray


@dataclass(frozen=True)
class Events:
    peaks: tuple[int, ...]
    z: tuple[float, ...]
    robust_sd: float


@dataclass(frozen=True)
class CellAttenuation:
    peaks: tuple[int, ...]
    n_events: int
    n_negative_events: int  # the same detector on the sign-flipped trace: the noise floor's event count
    max_z: float
    amp_m: float
    amp_l: float
    amp_s: float
    amp_r: float
    amp_neuropil: float
    retained_s: float
    retained_r: float
    neuropil_share: float
    neuropil_retained_s: float  # a_S / a_M in the annulus: how much of the surround's event S keeps


def averaging_matrix(footprints: Sequence[np.ndarray], n_pixels: int) -> sparse.csr_matrix:
    """Sparse (n_regions, n_pixels) matrix whose row r averages a flat image over footprint r."""
    rows, cols, vals = [], [], []
    for r, fp in enumerate(footprints):
        fp = np.unique(np.asarray(fp, dtype=np.int64))
        if fp.size == 0:
            raise ValueError(f"footprint {r} is empty")
        rows.append(np.full(fp.size, r))
        cols.append(fp)
        vals.append(np.full(fp.size, 1.0 / fp.size))
    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(footprints), n_pixels)
    )


def movie_region_traces(
    read_chunk: Callable[[int, int], np.ndarray], n_frames: int, a: sparse.csr_matrix, chunk: int = 250
) -> np.ndarray:
    """(n_regions, T) means of a movie read as (frames, n_pixels) chunks."""
    out = np.empty((a.shape[0], n_frames), dtype=np.float64)
    for t0 in range(0, n_frames, chunk):
        t1 = min(t0 + chunk, n_frames)
        frames = np.asarray(read_chunk(t0, t1), dtype=np.float64).reshape(t1 - t0, -1)
        out[:, t0:t1] = (a @ frames.T).reshape(a.shape[0], t1 - t0)
    return out


def background_region_traces(a: sparse.csr_matrix, us_k: np.ndarray, v_k_full: np.ndarray) -> np.ndarray:
    """(n_regions, T) means of the low-rank background ``V_k @ US_k.T``, without building it."""
    return np.asarray(a @ us_k.astype(np.float64)) @ v_k_full.astype(np.float64).T


def source_region_traces(a: sparse.csr_matrix, layer) -> np.ndarray:
    """(n_regions, T) means of one source layer's subtracted contribution ``traces.T @ W_design`` at its pixels."""
    a_src = a[:, layer.flat_idx]
    return np.asarray(a_src @ layer.W_design.astype(np.float64).T) @ layer.traces.astype(np.float64)


def annulus(footprint: np.ndarray, shape: tuple, occupied: np.ndarray, gap: int = 2, ring: int = 6) -> np.ndarray:
    """Flat indices of a ring ``gap`` px off the footprint and ``ring`` px wide, minus occupied pixels."""
    m = np.zeros(shape, bool)
    m.flat[footprint] = True
    disk = np.ones((3, 3), bool)
    inner = binary_dilation(m, disk, iterations=gap)
    outer = binary_dilation(inner, disk, iterations=ring)
    return np.flatnonzero(outer & ~inner & ~occupied)


def detect_events(
    trace: np.ndarray,
    fs: float,
    z: float = 4.0,
    min_separation_s: float = 2.0,
    baseline_window_s: float = 60.0,
    baseline_percentile: int = 10,
) -> Events:
    """Upward events: runs above ``z`` robust SDs over a sliding-percentile baseline, one peak per run.

    Peaks closer than ``min_separation_s`` keep the higher one.
    """
    x = np.asarray(trace, dtype=np.float64)
    w = max(2, min(x.size, int(round(baseline_window_s * fs))))
    d = x - percentile_filter(x, percentile=baseline_percentile, size=w, mode="nearest")
    d = d - np.median(d)
    sd = float(np.median(np.abs(d)) * MAD_TO_SD)
    if sd == 0.0:
        return Events((), (), 0.0)
    zs = d / sd
    above = np.flatnonzero(zs > z)
    if above.size == 0:
        return Events((), (), sd)
    runs = np.split(above, np.flatnonzero(np.diff(above) > 1) + 1)
    peaks = sorted((int(r[np.argmax(zs[r])]) for r in runs), key=lambda p: -zs[p])
    sep = max(1, int(round(min_separation_s * fs)))
    kept: list[int] = []
    for p in peaks:
        if all(abs(p - q) >= sep for q in kept):
            kept.append(p)
    kept.sort()
    return Events(tuple(kept), tuple(float(zs[p]) for p in kept), sd)


def event_amplitudes(trace: np.ndarray, peaks: Sequence[int], fs: float) -> np.ndarray:
    """Per event: mean over the peak +- PEAK_HALF_FRAMES minus the pre-event window's mean; NaN off the ends."""
    x = np.asarray(trace, dtype=np.float64)
    pre0, pre1 = (int(round(s * fs)) for s in PRE_WINDOW_S)
    out = np.full(len(peaks), np.nan)
    for i, p in enumerate(peaks):
        b0, b1 = p - pre0, p - pre1
        e0, e1 = p - PEAK_HALF_FRAMES, p + PEAK_HALF_FRAMES + 1
        if b0 < 0 or e1 > x.size or b1 <= b0:
            continue
        out[i] = x[e0:e1].mean() - x[b0:b1].mean()
    return out


def _median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        return _median(np.where(den > 0, num / den, np.nan))


def cell_attenuation(
    cell: RegionLayers, ann: RegionLayers, fs: float, neuropil_coeff: float = 0.7, z: float = 4.0
) -> CellAttenuation:
    """Detect events on ``cell.m - neuropil_coeff * ann.m`` and read each event's amplitude in every layer."""
    corrected = np.asarray(cell.m, dtype=np.float64) - neuropil_coeff * np.asarray(ann.m, dtype=np.float64)
    ev = detect_events(corrected, fs, z=z)
    neg = detect_events(-corrected, fs, z=z)
    a_m = event_amplitudes(cell.m, ev.peaks, fs)
    a_s = a_m - event_amplitudes(cell.l, ev.peaks, fs)
    a_r = a_s - event_amplitudes(cell.sub, ev.peaks, fs)
    n_m = event_amplitudes(ann.m, ev.peaks, fs)
    n_s = n_m - event_amplitudes(ann.l, ev.peaks, fs)
    return CellAttenuation(
        peaks=ev.peaks,
        n_events=len(ev.peaks),
        n_negative_events=len(neg.peaks),
        max_z=max(ev.z, default=float("nan")),
        amp_m=_median(a_m),
        amp_l=_median(a_m - a_s),
        amp_s=_median(a_s),
        amp_r=_median(a_r),
        amp_neuropil=_median(n_m),
        retained_s=_ratio(a_s, a_m),
        retained_r=_ratio(a_r, a_m),
        neuropil_share=_ratio(n_m, a_m),
        neuropil_retained_s=_ratio(n_s, n_m),
    )
