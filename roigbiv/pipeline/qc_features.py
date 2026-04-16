"""
ROI G. Biv pipeline — Unified QC Feature Extraction (spec §13.1).

Fills in every feature the downstream classifier, HITL reviewer, and
metadata consumer might need. Features already populated by earlier gates
(area, solidity, eccentricity, nuclear_shadow_score, soma_surround_contrast)
are reused; this module only computes what is missing.

Computes three groups:
  - Spatial: boundary_gradient, spatial_blur, fov_distance (new)
  - Temporal: std, skew, snr, n_transients, mean_fluorescence, bp_std,
              bp_power_ratio, autocorr_tau (from trace_corrected)
  - Provenance: n_stages_detected (cross-stage IoU count)

The bandpass-filtered trace (0.05–2.0 Hz) is stored on ROI.features
['trace_bandpass'] as an ndarray. It is consumed by:
  - the classifier (bp_std, tonic criterion)
  - the HITL package (PRIMARY evidence for Stage 4 candidates; Blindspot 13)
  - the Napari review-queue layer (optional trace display)
"""
from __future__ import annotations

import numpy as np
from scipy import signal, stats
from scipy.ndimage import binary_erosion

from roigbiv.pipeline.types import FOVData, ROI, PipelineConfig


# ─────────────────────────────────────────────────────────────────────────
# Spatial features
# ─────────────────────────────────────────────────────────────────────────

def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0.0, 0.0
    return float(ys.mean()), float(xs.mean())


def _boundary_gradient(mask: np.ndarray, mean_S: np.ndarray) -> float:
    """Mean |∇mean_S| at ROI boundary pixels.

    Boundary = mask XOR erode(mask, 1 px).
    """
    if not mask.any():
        return 0.0
    eroded = binary_erosion(mask, iterations=1)
    boundary = mask & ~eroded
    if not boundary.any():
        return 0.0
    gy, gx = np.gradient(mean_S.astype(np.float32))
    mag = np.sqrt(gy * gy + gx * gx)
    return float(mag[boundary].mean())


def _spatial_blur_fwhm(mask: np.ndarray, image: np.ndarray) -> float:
    """Radial FWHM of intensity distribution around the ROI centroid (Blindspot 7).

    Approach: compute radial intensity profile out to ~2× ROI radius, find
    the smallest radius at which intensity drops to half its peak. Ghost
    cells (out-of-focus) have broader FWHM than in-focus somas.
    """
    if not mask.any():
        return 0.0
    cy, cx = _centroid(mask)
    H, W = image.shape
    radius_px = max(4, int(np.sqrt(mask.sum() / np.pi) * 2))
    y0 = max(0, int(cy - radius_px)); y1 = min(H, int(cy + radius_px) + 1)
    x0 = max(0, int(cx - radius_px)); x1 = min(W, int(cx + radius_px) + 1)
    patch = image[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return 0.0
    yy, xx = np.indices(patch.shape)
    rr = np.sqrt((yy - (cy - y0)) ** 2 + (xx - (cx - x0)) ** 2)
    # Build radial profile by integer-bin averaging
    max_r = int(rr.max())
    if max_r <= 0:
        return 0.0
    profile = np.zeros(max_r + 1, dtype=np.float32)
    counts = np.zeros(max_r + 1, dtype=np.int32)
    r_int = rr.astype(np.int32)
    flat_r = r_int.ravel(); flat_v = patch.ravel()
    np.add.at(profile, flat_r, flat_v)
    np.add.at(counts, flat_r, 1)
    profile = np.where(counts > 0, profile / np.maximum(counts, 1), 0.0)
    if profile.size == 0 or profile[0] <= 0:
        return 0.0
    peak = float(profile[0])
    half = peak * 0.5
    below = np.where(profile < half)[0]
    if below.size == 0:
        return float(profile.size)  # never crosses half → very broad
    # 2 × first crossing radius ≈ FWHM
    return float(2 * below[0])


def compute_spatial_features(
    roi: ROI,
    mean_S: np.ndarray,
    dog_map: np.ndarray,
    all_masks_union: np.ndarray,
    fov_shape_hw: tuple,
) -> None:
    """Populate missing spatial features on roi.features.

    Does NOT overwrite existing features. Existing gate-time fields
    (area, solidity, eccentricity, nuclear_shadow_score,
    soma_surround_contrast) live on the ROI dataclass as top-level attributes
    and remain canonical.
    """
    feats = roi.features
    H, W = fov_shape_hw

    cy, cx = _centroid(roi.mask)
    feats.setdefault("centroid_y", cy)
    feats.setdefault("centroid_x", cx)

    if "boundary_gradient" not in feats:
        feats["boundary_gradient"] = _boundary_gradient(roi.mask, mean_S)

    if "spatial_blur" not in feats:
        feats["spatial_blur"] = _spatial_blur_fwhm(roi.mask, mean_S)

    if "fov_distance" not in feats:
        feats["fov_distance"] = float(
            np.hypot(cy - H / 2.0, cx - W / 2.0)
        )


# ─────────────────────────────────────────────────────────────────────────
# Temporal features
# ─────────────────────────────────────────────────────────────────────────

def _bandpass(trace: np.ndarray, fs: float,
              low: float = 0.05, high: float = 2.0) -> np.ndarray:
    """0.05–2.0 Hz zero-phase Butterworth bandpass (spec §13.1).

    Clamp `high` below Nyquist and guard short traces.
    """
    trace = np.asarray(trace, dtype=np.float32)
    if trace.size < 16:
        return trace.copy()
    nyq = fs / 2.0
    high = min(high, nyq * 0.99)
    low = min(low, high - 1e-4)
    try:
        sos = signal.butter(4, [low, high], btype="bandpass",
                            fs=fs, output="sos")
        return signal.sosfiltfilt(sos, trace).astype(np.float32)
    except ValueError:
        return trace.copy()


def _count_transients(
    trace: np.ndarray,
    template: np.ndarray,
    fs: float,
    tau: float,
    threshold_sigma: float = 3.0,
) -> int:
    """1-D matched filter peak count.

    Convolve the trace with `template`, threshold at threshold_sigma × MAD-based
    noise floor, count peaks separated by at least 2τ seconds.
    """
    trace = np.asarray(trace, dtype=np.float32)
    if trace.size < template.size + 4:
        return 0
    score = signal.fftconvolve(trace, template[::-1], mode="same").astype(np.float32)
    # Noise floor from MAD of score
    mad = float(np.median(np.abs(score - np.median(score))))
    sigma = mad / 0.6745 if mad > 0 else float(score.std() + 1e-12)
    if sigma <= 0:
        return 0
    min_distance = max(1, int(round(2.0 * tau * fs)))
    peaks, _ = signal.find_peaks(
        score, height=threshold_sigma * sigma, distance=min_distance,
    )
    return int(peaks.size)


def _autocorr_tau(trace_bp: np.ndarray, fs: float) -> float:
    """Approximate 1/e decay time of the bandpass autocorrelation, in seconds.

    Returns 0 on degenerate inputs.
    """
    x = np.asarray(trace_bp, dtype=np.float32)
    n = x.size
    if n < 8:
        return 0.0
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    # Use FFT-based autocorrelation for speed on long traces
    max_lag = min(n - 1, int(30 * fs))  # up to 30 s
    ac = np.correlate(x, x, mode="full")[n - 1: n - 1 + max_lag + 1] / denom
    # First lag where autocorr drops below 1/e
    below = np.where(ac < np.exp(-1.0))[0]
    if below.size == 0:
        return float(max_lag / fs)
    return float(below[0] / fs)


def compute_temporal_features(
    roi: ROI,
    fs: float,
    tau: float,
    template: np.ndarray,
) -> None:
    """Populate temporal features on roi.features.

    Operates on roi.trace_corrected (neuropil-corrected) for std/skew/snr/
    n_transients/bp_*, and on roi.trace (raw) for mean_fluorescence.
    Also stores roi.features['trace_bandpass'] = bandpass-filtered trace
    (needed by classifier and HITL package).
    """
    feats = roi.features
    tc = roi.trace_corrected
    tr = roi.trace

    if tc is None or tr is None:
        return

    tc = np.asarray(tc, dtype=np.float32)
    tr = np.asarray(tr, dtype=np.float32)

    # Raw-corrected temporal stats
    feats["std"] = float(tc.std())
    feats["skew"] = float(stats.skew(tc, bias=False)) if tc.size > 2 else 0.0
    feats["mean_fluorescence"] = float(tr.mean())

    # SNR = peak / MAD-based noise floor
    mad = float(np.median(np.abs(tc - np.median(tc))))
    noise_floor = mad / 0.6745 if mad > 0 else float(tc.std() + 1e-12)
    feats["noise_floor"] = float(noise_floor)
    if noise_floor > 0:
        feats["snr"] = float((tc.max() - tc.mean()) / noise_floor)
    else:
        feats["snr"] = 0.0

    # 1-D matched-filter transient count (not the Stage 3 spatial sweep)
    feats["n_transients"] = _count_transients(tc, template, fs, tau)

    # Bandpass features
    trace_bp = _bandpass(tc, fs)
    feats["trace_bandpass"] = trace_bp
    feats["bp_std"] = float(trace_bp.std())

    # Power ratio: power in [low, high] / total power
    if tc.size >= 16:
        try:
            nperseg = min(256, max(8, tc.size // 4))
            freqs, psd = signal.welch(tc, fs=fs, nperseg=nperseg,
                                      noverlap=nperseg // 2)
            band = (freqs >= 0.05) & (freqs <= min(2.0, fs / 2 * 0.99))
            total_p = float(psd.sum())
            feats["bp_power_ratio"] = (
                float(psd[band].sum() / total_p) if total_p > 0 else 0.0
            )
        except Exception:
            feats["bp_power_ratio"] = 0.0
    else:
        feats["bp_power_ratio"] = 0.0

    feats["autocorr_tau"] = _autocorr_tau(trace_bp, fs)


# ─────────────────────────────────────────────────────────────────────────
# Provenance features
# ─────────────────────────────────────────────────────────────────────────

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


def compute_provenance_features(roi: ROI, all_rois: list[ROI]) -> None:
    """Count how many stages independently discovered this ROI (via IoU > 0.3).

    Cross-stage matches were not tracked during detection, so compute now
    from the full ROI pool. Always at least 1 (self).
    """
    if "n_stages_detected" in roi.features:
        return
    stages = {int(roi.source_stage)}
    for other in all_rois:
        if other.label_id == roi.label_id:
            continue
        if int(other.source_stage) in stages:
            continue
        if _iou(roi.mask, other.mask) > 0.3:
            stages.add(int(other.source_stage))
    roi.features["n_stages_detected"] = int(len(stages))


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────

def compute_all_features(
    fov: FOVData,
    rois: list[ROI],
    cfg: PipelineConfig,
    template_bank: list,
) -> None:
    """Fill in every QC feature on every ROI. Modifies rois in place."""
    if not rois:
        return

    H, W = fov.shape[1], fov.shape[2]
    all_masks_union = np.zeros((H, W), dtype=bool)
    for r in rois:
        all_masks_union |= r.mask

    # Pick the first (single_*) template for 1-D matched-filter counting
    template = template_bank[0][1]

    for roi in rois:
        compute_spatial_features(roi, fov.mean_S, fov.dog_map,
                                 all_masks_union, (H, W))
        compute_temporal_features(roi, cfg.fs, cfg.tau, template)
        compute_provenance_features(roi, rois)
