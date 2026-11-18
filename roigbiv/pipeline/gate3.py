"""
ROI G. Biv pipeline — Gate 3: Waveform Validation (spec §10).

Verifies that Stage 3 candidates have transient waveforms consistent with
real calcium events — not noise crossings, residual subtraction artifacts,
or anti-correlation signatures from imperfect prior subtraction.

Checks per spec §10:
  - Waveform R² against best-matching template (≥ 0.6, or ≥ 0.5 for
    single-event candidates — min_r2 relaxation per plan D8)
  - Rise/decay asymmetry < 0.5 (fast rise, slow decay)
  - Anti-correlation against prior Stage 1+2 ROIs within 20 px (≤ -0.5
    indicates subtraction artifact — cascade defense, Blindspot 2)
  - Spatial compactness: solidity ≥ 0.5

Confidence grading by event count:
  1 event → "low" (retain but HITL priority)
  2-5 events → "moderate"
  6+ events → "high"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.measure import regionprops

from roigbiv.pipeline.types import GATE3_WAVEFORM_SOURCES, ROI, PipelineConfig


# ─────────────────────────────────────────────────────────────────────────
# Waveform utilities
# ─────────────────────────────────────────────────────────────────────────

def gate3_window_frames(cfg: PipelineConfig) -> int:
    """Event waveform window length in frames: ``tau_multiple * tau * fs``, at least 5."""
    return max(5, int(cfg.gate3_waveform_window_tau_multiple * cfg.tau * cfg.fs))


def event_window_bounds(event_frame: int, window_frames: int, n_frames: int) -> tuple[int, int]:
    """``[lo, hi)`` of the asymmetric window ``[t - window/4, t + 3*window/4]``, clamped to the trace."""
    pre = window_frames // 4
    lo = max(0, event_frame - pre)
    hi = min(n_frames, event_frame + window_frames - pre)
    return lo, hi


def baseline_subtract(segment: np.ndarray) -> np.ndarray:
    """Subtract the mean of the first 10 % of the window (at least one sample)."""
    w = np.asarray(segment, dtype=np.float32)
    if w.size == 0:
        return w
    return w - w[: max(1, int(0.1 * len(w)))].mean()


def _extract_event_waveform(
    trace: np.ndarray,
    event_frame: int,
    window_frames: int,
) -> np.ndarray:
    """Extract a baseline-subtracted waveform around an event.

    Returns a zero-length array if the window falls outside the trace.
    """
    lo, hi = event_window_bounds(event_frame, window_frames, len(trace))
    if hi - lo <= 0:
        return np.zeros(0, dtype=np.float32)
    return baseline_subtract(trace[lo:hi])


def _waveform_r2(waveform: np.ndarray, templates: list[np.ndarray]) -> tuple[float, int]:
    """R² of waveform vs the best-matching template, aligned at peak.

    For each template: shift so template peak aligns with waveform peak,
    truncate/zero-pad to common length, compute R² = 1 - SS_res / SS_tot.
    Returns (best_r2, best_template_idx).
    """
    if len(waveform) < 3:
        return 0.0, 0
    ss_tot = float(((waveform - waveform.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 0.0, 0

    best_r2 = -np.inf
    best_k = 0
    w_peak = int(np.argmax(waveform))
    w_amp = float(waveform[w_peak])
    if w_amp <= 0:
        return 0.0, 0

    for k, tmpl in enumerate(templates):
        if len(tmpl) == 0:
            continue
        t_peak = int(np.argmax(tmpl))
        # Align template peak to waveform peak
        start_w = max(0, w_peak - t_peak)
        start_t = max(0, t_peak - w_peak)
        length = min(len(waveform) - start_w, len(tmpl) - start_t)
        if length < 3:
            continue
        w_seg = waveform[start_w : start_w + length]
        t_seg = tmpl[start_t : start_t + length]
        # Least-squares amplitude: amp = (w @ t) / (t @ t)
        tt = float(t_seg @ t_seg)
        if tt <= 0:
            continue
        amp = float(w_seg @ t_seg) / tt
        if amp <= 0:
            continue
        t_scaled = t_seg * amp
        ss_res = float(((w_seg - t_scaled) ** 2).sum())
        ss_tot_seg = float(((w_seg - w_seg.mean()) ** 2).sum())
        if ss_tot_seg <= 0:
            continue
        r2 = 1.0 - ss_res / ss_tot_seg
        if r2 > best_r2:
            best_r2 = r2
            best_k = k
    if not np.isfinite(best_r2):
        return 0.0, 0
    return float(best_r2), int(best_k)


def _crossing(w: np.ndarray, i: int, level: float) -> float:
    """Fractional index where ``w`` crosses ``level`` between samples ``i`` and ``i + 1``."""
    a, b = float(w[i]), float(w[i + 1])
    return i + (level - a) / (b - a) if b != a else float(i)


def _rise_decay_ratio(waveform: np.ndarray) -> float:
    """Rise time (10→90% of peak) divided by decay time (peak→37% of peak).

    The rise starts at the event onset: the last sample below 10% of the peak
    before the peak, so baseline noise earlier in the window does not move
    it. Crossings are linearly interpolated between samples, so a rise inside
    one frame (a fast indicator at a low frame rate) is a small finite value.
    Returns +inf if there is no sub-10% sample before the peak or no fall to
    37% after it.
    """
    if len(waveform) < 5:
        return np.inf
    w = np.asarray(waveform, dtype=np.float64)
    peak_idx = int(np.argmax(w))
    peak_val = float(w[peak_idx])
    if peak_val <= 0 or peak_idx <= 0 or peak_idx >= len(w) - 1:
        return np.inf

    thr10 = 0.10 * peak_val
    thr90 = 0.90 * peak_val
    thr37 = 0.37 * peak_val

    below10 = np.where(w[:peak_idx] < thr10)[0]
    if below10.size == 0:
        return np.inf
    onset = int(below10[-1])
    t10 = _crossing(w, onset, thr10)
    i90 = onset + 1 + int(np.argmax(w[onset + 1 : peak_idx + 1] >= thr90))
    t90 = _crossing(w, i90 - 1, thr90) if w[i90 - 1] < thr90 else float(i90)
    rise = t90 - t10
    if rise <= 0:
        return np.inf

    below37 = np.where(w[peak_idx:] <= thr37)[0]
    if below37.size == 0:
        return np.inf
    i37 = peak_idx + int(below37[0])
    decay = _crossing(w, i37 - 1, thr37) - peak_idx
    if decay <= 0:
        return np.inf
    return rise / decay


# ─────────────────────────────────────────────────────────────────────────
# Prior-ROI anti-correlation (cascade defense)
# ─────────────────────────────────────────────────────────────────────────

def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return -1.0, -1.0
    return float(ys.mean()), float(xs.mean())


def _pearson_row(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a_c = a - a.mean()
    B_c = B - B.mean(axis=1, keepdims=True)
    num = B_c @ a_c
    a_norm = float(np.linalg.norm(a_c))
    B_norm = np.linalg.norm(B_c, axis=1)
    denom = a_norm * B_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, num / denom, 0.0)
    return r.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────

def evaluate_gate3(
    candidates: list[ROI],
    prior_rois: list[ROI],
    template_bank: list[tuple[str, np.ndarray]],
    cfg: PipelineConfig,
) -> list[ROI]:
    """Apply Gate 3 decision rules to Stage 3 candidates.

    Mutates candidate ROIs in place (gate_outcome, confidence, features,
    gate_reasons, solidity, eccentricity). Returns the same list.
    """
    if cfg.gate3_waveform_source not in GATE3_WAVEFORM_SOURCES:
        raise ValueError(f"gate3_waveform_source must be one of {GATE3_WAVEFORM_SOURCES}, "
                         f"got {cfg.gate3_waveform_source!r}")
    templates = [wf for _, wf in template_bank]
    window_frames = gate3_window_frames(cfg)

    # Pre-extract prior traces + centroids for anti-correlation check
    prior_centroids = []
    prior_traces_list = []
    for r in prior_rois:
        if r.trace is None:
            continue
        prior_centroids.append(_centroid(r.mask))
        prior_traces_list.append(r.trace)
    if prior_traces_list:
        shared_len = min(len(t) for t in prior_traces_list)
        prior_traces = np.stack([t[:shared_len] for t in prior_traces_list])
        prior_centroids_arr = np.array(prior_centroids, dtype=np.float32)
    else:
        prior_traces = None
        prior_centroids_arr = np.empty((0, 2), dtype=np.float32)

    for roi in candidates:
        failures: list[str] = []

        # ── Morphology: solidity + compute eccentricity ──────────────────
        lbl = roi.mask.astype(np.uint8)
        rp = regionprops(lbl)
        if rp:
            roi.solidity = float(rp[0].solidity or 0.0)
            roi.eccentricity = float(rp[0].eccentricity or 0.0)
        if roi.solidity < cfg.gate3_min_solidity:
            failures.append(f"solidity:{roi.solidity:.3f}")

        # ── Waveform R² per event, take max ──────────────────────────────
        event_list = roi.features.get("picked_events") or roi.features.get("events") or []
        event_count = int(roi.event_count or 0)
        per_event_r2: list[float] = []
        best_event_waveform: np.ndarray = np.zeros(0, dtype=np.float32)
        best_r2 = 0.0
        best_k = 0
        if cfg.gate3_waveform_source == "event_patch" and roi.event_waveforms is not None:
            waveforms = list(roi.event_waveforms)
            roi.features["gate3_waveform_source"] = "event_patch"
        elif roi.trace is not None:
            waveforms = [_extract_event_waveform(roi.trace, int(ev["frame"]), window_frames)
                         for ev in event_list]
            roi.features["gate3_waveform_source"] = "roi_trace"
        else:
            waveforms = []
        for wf in waveforms:
            if wf.size:
                r2, k = _waveform_r2(wf, templates)
                per_event_r2.append(r2)
                if r2 > best_r2:
                    best_r2 = r2
                    best_k = k
                    best_event_waveform = wf

        min_r2 = (cfg.gate3_min_waveform_r2_single_event
                  if event_count == 1 else cfg.gate3_min_waveform_r2)
        if not per_event_r2 or best_r2 < min_r2:
            failures.append(f"waveform_r2:{best_r2:.3f}<{min_r2}")

        # ── Rise/decay asymmetry on best event ───────────────────────────
        rise_decay = _rise_decay_ratio(best_event_waveform) if best_event_waveform.size else np.inf
        if rise_decay >= cfg.gate3_max_rise_decay_ratio:
            failures.append(f"rise_decay:{rise_decay:.3f}")

        # ── Anti-correlation against prior ROIs within gate3 radius ──────
        min_corr = 0.0
        if prior_traces is not None and roi.trace is not None:
            cy, cx = _centroid(roi.mask)
            dists = np.linalg.norm(prior_centroids_arr - np.array([cy, cx], np.float32),
                                   axis=1)
            nearby = dists <= cfg.gate2_spatial_radius  # same 20px radius as Gate 2
            if nearby.any():
                # Align trace length with the stored prior traces
                candtrace = roi.trace[: prior_traces.shape[1]]
                r = _pearson_row(candtrace, prior_traces[nearby])
                min_corr = float(r.min())
                if min_corr <= cfg.gate3_anticorr_threshold:
                    failures.append(f"anticorr:{min_corr:.3f}")

        # ── Confidence grading by event count ────────────────────────────
        if event_count >= 6:
            confidence = "high"
        elif event_count >= 2:
            confidence = "moderate"
        else:
            confidence = "low"

        # ── Decision ──────────────────────────────────────────────────────
        if failures:
            roi.gate_outcome = "reject"
            roi.confidence = "requires_review"
        elif best_r2 < (min_r2 + 0.1):
            # Marginal R² (just above threshold) → flag for review
            roi.gate_outcome = "flag"
            roi.confidence = confidence
        else:
            roi.gate_outcome = "accept"
            roi.confidence = confidence

        roi.gate_reasons = failures
        roi.features.update({
            "gate3_best_r2": best_r2,
            "gate3_best_template_idx": best_k,
            "gate3_rise_decay_ratio": float(rise_decay) if np.isfinite(rise_decay) else None,
            "gate3_min_corr": min_corr,
            "gate3_event_r2_values": per_event_r2,
        })

    return candidates
