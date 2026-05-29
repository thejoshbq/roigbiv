"""
ROI G. Biv pipeline — Gate 4: Correlation Contrast Validation (spec §12).

Final validation gate for Stage 4 tonic-neuron candidates. Six checks:

  1. Correlation contrast   >  gate4_min_corr_contrast    (default 0.10)
  2. Eccentricity           <= stage4_max_eccentricity    (default 0.85)
  3. Solidity               >= stage4_min_solidity        (default 0.60)
  4. Motion correlation     |r with ops['xoff','yoff']|
                              <  gate4_max_motion_corr     (default 0.30)
     — UNIQUE to Gate 4: post-subtraction residuals leave fluctuating ring
     artifacts at soma boundaries that track sub-pixel motion and mimic
     tonic signals (Blindspot 9).
  5. Anti-correlation cascade defense (Blindspot 2): r with any prior
     Stage 1-3 ROI within gate4_spatial_radius (20px)
                              >  gate4_anticorr_threshold (default -0.50)
  6. Mean projection intensity above FOV percentile floor:
     mean(mean_M[mask])      >= percentile(mean_M, gate4_min_mean_intensity_pct)
     (default 25th percentile)
     — the original task spec calls for mean_S here, but mean_S ≈ 0 under
     truncated-SVD L+S (foundation.py:500-502 documents this), making a
     percentile filter on it meaningless noise. mean_M preserves the raw
     morphological brightness and is the reference Cellpose/Gate 1 already
     use for soma-surround contrast.

Decision tier: Gate 4 has NO "accept" tier. Every candidate that passes all
six checks receives gate_outcome="flag" and confidence="requires_review";
failures receive gate_outcome="reject". This reflects the pipeline's
epistemic humility about tonic detection — the automated algorithm cannot
confirm these with the same confidence as Stages 1-3, so HITL review of
bandpass traces + correlation maps is mandatory.
"""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline.gate3 import _centroid, _pearson_row
from roigbiv.pipeline.types import ROI, PipelineConfig


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two 1-D arrays, returning 0.0 if either is constant."""
    if a.size != b.size or a.size < 2:
        return 0.0
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_c = a - a.mean()
    b_c = b - b.mean()
    an = float(np.linalg.norm(a_c))
    bn = float(np.linalg.norm(b_c))
    if an <= 0 or bn <= 0:
        return 0.0
    return float((a_c @ b_c) / (an * bn))


def evaluate_gate4(
    candidates: list[ROI],
    prior_rois: list[ROI],
    mean_M: np.ndarray,
    motion_x: np.ndarray,
    motion_y: np.ndarray,
    cfg: PipelineConfig,
) -> list[ROI]:
    """Apply Gate 4 decision rules. Mutates candidates in place; returns them.

    Parameters
    ----------
    candidates : list[ROI]
        Stage 4 candidates (source_stage=4, corr_contrast + trace populated).
    prior_rois : list[ROI]
        Stage 1-3 ROIs with gate_outcome in {"accept", "flag"} — used for
        the anti-correlation cascade-defense check.
    mean_M : (H, W) float32
        Raw morphological mean of the registered movie. Intensity reference
        for the 25th-percentile check (see module docstring for why mean_M
        rather than mean_S).
    motion_x, motion_y : (T,)
        Suite2p registration displacements (from ops['xoff']/['yoff']).
    cfg : PipelineConfig
    """
    if mean_M is None:
        raise ValueError("Gate 4 requires fov.mean_M for the intensity check")

    intensity_floor = float(np.percentile(mean_M.astype(np.float32),
                                          cfg.gate4_min_mean_intensity_pct))

    # Pre-build prior traces + centroids for the anti-correlation check
    prior_centroids: list[tuple[float, float]] = []
    prior_traces_list: list[np.ndarray] = []
    for r in prior_rois:
        if r.trace is None:
            continue
        prior_centroids.append(_centroid(r.mask))
        prior_traces_list.append(r.trace)
    if prior_traces_list:
        shared_len = min(len(t) for t in prior_traces_list)
        prior_traces = np.stack([t[:shared_len].astype(np.float32)
                                 for t in prior_traces_list])
        prior_centroids_arr = np.array(prior_centroids, dtype=np.float32)
    else:
        prior_traces = None
        prior_centroids_arr = np.empty((0, 2), dtype=np.float32)

    mx = np.asarray(motion_x, dtype=np.float32) if motion_x is not None else None
    my = np.asarray(motion_y, dtype=np.float32) if motion_y is not None else None

    for roi in candidates:
        failures: list[str] = []

        # 1. Correlation contrast
        cc = float(roi.corr_contrast) if roi.corr_contrast is not None else 0.0
        if cc <= cfg.gate4_min_corr_contrast:
            failures.append(f"corr_contrast:{cc:.3f}")

        # 2. Eccentricity
        if roi.eccentricity > cfg.stage4_max_eccentricity:
            failures.append(f"eccentricity:{roi.eccentricity:.3f}")

        # 3. Solidity
        if roi.solidity < cfg.stage4_min_solidity:
            failures.append(f"solidity:{roi.solidity:.3f}")

        # 4. Motion correlation (RAW trace, not bandpass-filtered —
        #    motion artifacts have power across frequencies)
        r_x = r_y = 0.0
        if roi.trace is not None and mx is not None and my is not None:
            trc = roi.trace.astype(np.float32)
            tlen = min(len(trc), len(mx), len(my))
            if tlen >= 2:
                r_x = _safe_pearson(trc[:tlen], mx[:tlen])
                r_y = _safe_pearson(trc[:tlen], my[:tlen])
                if max(abs(r_x), abs(r_y)) >= cfg.gate4_max_motion_corr:
                    failures.append(f"motion:max(|rx|,|ry|)={max(abs(r_x), abs(r_y)):.3f}")

        # 5. Anti-correlation cascade defense
        min_prior_corr: float | None = None
        if prior_traces is not None and roi.trace is not None:
            cy, cx = _centroid(roi.mask)
            dists = np.linalg.norm(
                prior_centroids_arr - np.array([cy, cx], dtype=np.float32),
                axis=1,
            )
            nearby = dists <= cfg.gate4_spatial_radius
            if nearby.any():
                cand_trace = roi.trace[:prior_traces.shape[1]].astype(np.float32)
                r = _pearson_row(cand_trace, prior_traces[nearby])
                min_prior_corr = float(r.min())
                if min_prior_corr <= cfg.gate4_anticorr_threshold:
                    failures.append(f"anticorr:{min_prior_corr:.3f}")

        # 6. Mean projection intensity floor (mean_M; see module docstring)
        mean_intensity = float(mean_M[roi.mask].mean()) if roi.mask.any() else 0.0
        if mean_intensity < intensity_floor:
            failures.append(f"intensity:{mean_intensity:.3g}<{intensity_floor:.3g}")

        # Decision — no "accept" tier for Stage 4
        if failures:
            roi.gate_outcome = "reject"
            roi.confidence = "requires_review"
        else:
            roi.gate_outcome = "flag"
            roi.confidence = "requires_review"
        roi.gate_reasons = failures

        roi.features.update({
            "motion_corr_x": float(r_x),
            "motion_corr_y": float(r_y),
            "gate4_anticorr_min": (None if min_prior_corr is None
                                   else float(min_prior_corr)),
            "gate4_mean_intensity": float(mean_intensity),
            "gate4_mean_intensity_floor": float(intensity_floor),
        })

    return candidates
