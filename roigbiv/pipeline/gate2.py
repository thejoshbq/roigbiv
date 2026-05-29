"""
ROI G. Biv pipeline — Gate 2: Temporal Cross-Validation (spec §8).

Verifies that Stage 2 candidates are genuinely independent sources — not
rediscoveries the IoU filter missed, spatial spillover from a nearby Stage 1
ROI, or subtraction artifacts from imperfect Stage 1 subtraction.

Decision rules (spec §8 + §18.5):
  REJECT if
    - |r| ≥ gate2_max_correlation with any nearby Stage 1 ROI (redundant/spillover), OR
    - r ≤ gate2_anticorr_threshold with any nearby Stage 1 ROI (subtraction artifact,
      Blindspot 2 — the cascade-defense check), OR
    - centroid within gate2_near_distance px AND |r| > gate2_near_corr_threshold
      with any Stage 1 ROI (near-duplicate just below IoU cutoff), OR
    - area ∉ [gate2_min_area, gate2_max_area] OR solidity < gate2_min_solidity
      (relaxed vs Gate 1 because Suite2p footprints are noisier)
  FLAG   if passes all but any nearby |r| > gate2_flag_corr_threshold
  ACCEPT otherwise

"Nearby" = centroid within gate2_spatial_radius px. Candidates with NO nearby
Stage 1 ROIs skip the correlation checks (nothing to compare against).

Stage 1 traces are read from ROI.trace, which was populated by the Stage 1
subtraction engine during Phase 1B.
"""
from __future__ import annotations

import numpy as np
from skimage.measure import regionprops

from roigbiv.pipeline.types import ROI, PipelineConfig


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    """Centroid (cy, cx) of a dense bool mask."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return -1.0, -1.0
    return float(ys.mean()), float(xs.mean())


def _pearson_row(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r between a (T,) trace and each row of B (N, T).

    Robust to zero-variance traces — returns 0 where denom is 0.
    """
    a_c = a - a.mean()
    B_c = B - B.mean(axis=1, keepdims=True)
    num = B_c @ a_c
    a_norm = float(np.linalg.norm(a_c))
    B_norm = np.linalg.norm(B_c, axis=1)
    denom = a_norm * B_norm
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, num / denom, 0.0)
    return r.astype(np.float32)


def evaluate_gate2(
    candidates: list[ROI],
    stage1_rois: list[ROI],
    cfg: PipelineConfig,
) -> list[ROI]:
    """Apply Gate 2 decision rules to Stage 2 candidates.

    Mutates candidates' gate_outcome, confidence, and gate_reasons in place;
    also fills solidity/eccentricity from regionprops (not populated by Stage 2).
    Returns the same list for convenience.
    """
    # Pre-compute Stage 1 centroids and trace matrix for batched correlations
    s1_centroids = np.array([_centroid(r.mask) for r in stage1_rois],
                            dtype=np.float32) if stage1_rois else np.empty((0, 2))
    s1_traces = None
    if stage1_rois:
        s1_traces_list = [r.trace for r in stage1_rois if r.trace is not None]
        if s1_traces_list and all(t.shape == s1_traces_list[0].shape for t in s1_traces_list):
            s1_traces = np.stack(s1_traces_list)

    for roi in candidates:
        failures: list[str] = []

        # ── Morphology (relaxed thresholds) ───────────────────────────────
        if roi.area < cfg.gate2_min_area or roi.area > cfg.gate2_max_area:
            failures.append(f"area:{roi.area}")

        lbl = roi.mask.astype(np.uint8)
        rp = regionprops(lbl)
        if rp:
            roi.solidity = float(rp[0].solidity or 0.0)
            roi.eccentricity = float(rp[0].eccentricity or 0.0)
        if roi.solidity < cfg.gate2_min_solidity:
            failures.append(f"solidity:{roi.solidity:.3f}")

        # ── Temporal independence / anti-correlation ──────────────────────
        max_abs_corr = 0.0
        min_corr = 0.0
        near_duplicate_corr = 0.0
        if s1_traces is not None and roi.trace is not None and s1_traces.shape[0] > 0:
            cy, cx = _centroid(roi.mask)
            dists = np.linalg.norm(s1_centroids - np.array([cy, cx], dtype=np.float32),
                                   axis=1)
            nearby_mask = dists <= cfg.gate2_spatial_radius
            if nearby_mask.any():
                nearby_traces = s1_traces[nearby_mask]
                r = _pearson_row(roi.trace, nearby_traces)
                abs_r = np.abs(r)
                max_abs_corr = float(abs_r.max())
                min_corr = float(r.min())
                # Near-duplicate check: Stage 1 ROIs within near_distance
                near_mask = dists <= cfg.gate2_near_distance
                if near_mask.any():
                    near_r = _pearson_row(roi.trace, s1_traces[near_mask])
                    near_duplicate_corr = float(np.abs(near_r).max())

                if max_abs_corr >= cfg.gate2_max_correlation:
                    failures.append(f"corr:{max_abs_corr:.3f}")
                if min_corr <= cfg.gate2_anticorr_threshold:
                    failures.append(f"anticorr:{min_corr:.3f}")
                if (near_mask.any() and
                        near_duplicate_corr > cfg.gate2_near_corr_threshold):
                    failures.append(f"near_duplicate:{near_duplicate_corr:.3f}")

        # ── Decision ──────────────────────────────────────────────────────
        if failures:
            roi.gate_outcome = "reject"
            roi.confidence = "requires_review"
        elif max_abs_corr > cfg.gate2_flag_corr_threshold:
            roi.gate_outcome = "flag"
            roi.confidence = "moderate"
        else:
            roi.gate_outcome = "accept"
            roi.confidence = "high"

        roi.gate_reasons = failures
        roi.features.update({
            "gate2_max_abs_corr": max_abs_corr,
            "gate2_min_corr": min_corr,
            "gate2_near_duplicate_corr": near_duplicate_corr,
        })

    return candidates
