"""
ROI G. Biv pipeline — Activity-Type Classification (spec §13.3).

Rule-based decision tree (not a trained model). Every ROI receives one of
five labels: phasic, sparse, tonic, silent, ambiguous. The tree is evaluated
top-to-bottom — the first matching label sticks.

Silent-cell policy (Blindspot 8): a cell with no visible activity is
RETAINED if and only if its spatial morphology is convincing
(nuclear_shadow_score > 0 OR solidity > 0.7). This keeps cells that may
fire in a different session while rejecting flat traces at fragmented,
low-contrast locations.
"""
from __future__ import annotations

import numpy as np

from roigbiv.pipeline.types import ROI, PipelineConfig


def estimate_noise_floor(trace: np.ndarray) -> float:
    """MAD-based robust Gaussian std estimate."""
    if trace is None or len(trace) == 0:
        return 0.0
    trace = np.asarray(trace, dtype=np.float32)
    mad = float(np.median(np.abs(trace - np.median(trace))))
    if mad > 0:
        return mad / 0.6745
    return float(trace.std() + 1e-12)


def classify_activity_type(
    roi: ROI,
    median_F: float,
    median_std: float,
    cfg: PipelineConfig,
) -> str:
    """Apply the spec §13.3 decision tree.

    Population statistics (median_F, median_std) drive the tonic criterion's
    "high mean, low variance" fallback for non-Stage-4 ROIs.
    """
    f = roi.features
    n_trans = int(f.get("n_transients", 0))
    skew = float(f.get("skew", 0.0))
    bp_std = float(f.get("bp_std", 0.0))
    mean_F = float(f.get("mean_fluorescence", 0.0))
    std_F = float(f.get("std", 0.0))
    noise_floor = float(f.get("noise_floor", 0.0))

    # 1. PHASIC
    if n_trans >= cfg.phasic_min_transients and skew > cfg.phasic_min_skew:
        return "phasic"

    # 2. SPARSE
    if (
        n_trans >= cfg.sparse_min_transients
        and n_trans < cfg.phasic_min_transients
        and skew > cfg.sparse_min_skew
    ):
        return "sparse"

    # 3. TONIC
    tonic_bp_ok = bp_std > cfg.tonic_bp_std_factor * max(noise_floor, 1e-12)
    tonic_skew_ok = skew <= cfg.phasic_min_skew
    tonic_population_ok = (mean_F > median_F) and (std_F < median_std)
    if tonic_bp_ok and tonic_skew_ok and (int(roi.source_stage) == 4 or tonic_population_ok):
        return "tonic"

    # 4. SILENT — retained only if spatial morphology is convincing
    spatial_ok = (roi.nuclear_shadow_score > 0) or (roi.solidity > 0.7)
    if n_trans == 0 and bp_std < noise_floor and spatial_ok:
        return "silent"

    # 5. Fallback
    return "ambiguous"


def classify_all_rois(rois: list[ROI], cfg: PipelineConfig) -> None:
    """Assign activity_type to every ROI. Modifies in place; prints a summary."""
    if not rois:
        return

    mean_vals = np.array([float(r.features.get("mean_fluorescence", 0.0)) for r in rois])
    std_vals = np.array([float(r.features.get("std", 0.0)) for r in rois])
    median_F = float(np.median(mean_vals)) if mean_vals.size else 0.0
    median_std = float(np.median(std_vals)) if std_vals.size else 0.0

    for roi in rois:
        roi.activity_type = classify_activity_type(roi, median_F, median_std, cfg)

    counts = {}
    for r in rois:
        counts[r.activity_type] = counts.get(r.activity_type, 0) + 1
    ordered = ["phasic", "sparse", "tonic", "silent", "ambiguous"]
    summary = ", ".join(f"{counts.get(k, 0)} {k}" for k in ordered)
    print(f"Activity types: {summary}", flush=True)
