"""Isolation test: robust vs ridge solver on a synthetic two-source + ghost scene.

Creates two overlapping disc sources (~30% overlap area) with a planted ghost
contaminant (broad ring around source 1, 50% amplitude). Compares the ghost-induced
bias on source 1's trace between the ridge and robust solvers.

Expected: robust solver recovers source 1 trace with lower ghost bias (lower MSE
against ground truth) than the ridge solver.

Usage:
    conda run -n roigbiv python experiments/harness/test_robust_isolation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from roigbiv.pipeline.types import PipelineConfig
from roigbiv.pipeline.subtraction import solve_traces_from_chunks, solve_traces_robust_irls
from roigbiv.pipeline.subtraction import _build_union_design


def _disc_mask(H: int, W: int, cy: int, cx: int, r: int) -> np.ndarray:
    y, x = np.ogrid[:H, :W]
    return ((y - cy) ** 2 + (x - cx) ** 2 <= r ** 2).astype(np.float32)


def _ring_mask(H: int, W: int, cy: int, cx: int, r_inner: int, r_outer: int) -> np.ndarray:
    y, x = np.ogrid[:H, :W]
    d2 = (y - cy) ** 2 + (x - cx) ** 2
    return ((d2 > r_inner ** 2) & (d2 <= r_outer ** 2)).astype(np.float32)


def run():
    rng = np.random.default_rng(0)
    H, W, T = 64, 64, 500

    # Ground-truth traces: sparse GCaMP-like events
    c1_gt = np.zeros(T, dtype=np.float32)
    c2_gt = np.zeros(T, dtype=np.float32)
    for i in rng.integers(0, T, size=20):
        c1_gt[i: min(i + 12, T)] += np.exp(-np.arange(min(12, T - i)) * 0.3).astype(np.float32)
    for i in rng.integers(0, T, size=18):
        c2_gt[i: min(i + 12, T)] += np.exp(-np.arange(min(12, T - i)) * 0.3).astype(np.float32)
    c_ghost = 0.5 * c1_gt  # ghost = 50% amplitude of source 1

    # Spatial profiles
    profile1 = _disc_mask(H, W, 28, 28, 9)   # source 1
    profile2 = _disc_mask(H, W, 36, 36, 9)   # source 2 (overlaps ~30% with 1)
    ghost_profile = _ring_mask(H, W, 28, 28, 10, 20)  # broad ring around source 1

    # Synthetic residual movie: ground-truth + ghost
    # Shape: (T, H, W)
    movie = (
        profile1[None] * c1_gt[:, None, None]
        + profile2[None] * c2_gt[:, None, None]
        + ghost_profile[None] * c_ghost[:, None, None]
        + rng.standard_normal((T, H, W)).astype(np.float32) * 0.05
    )

    # Normalise profiles to [0, 1] as the pipeline does
    p1_norm = profile1 / max(profile1.max(), 1e-6)
    p2_norm = profile2 / max(profile2.max(), 1e-6)
    profiles = [p1_norm, p2_norm]  # ghost is UNMODELLED

    design, union_flat_idx, _ = _build_union_design(profiles)

    def _iter():
        chunk = 100
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            cs = t1 - t0
            chunk_ram = movie[t0:t1].reshape(cs, H * W)[:, union_flat_idx]
            yield t0, t1, chunk_ram

    cfg_ridge = PipelineConfig()
    cfg_robust = PipelineConfig(
        subtract_solver="robust",
        subtract_robust_kappa=0.5,
        subtract_robust_max_iter=8,
    )

    traces_ridge = solve_traces_from_chunks(design, T, _iter(), cfg_ridge)

    def _iter2():
        chunk = 100
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            cs = t1 - t0
            chunk_ram = movie[t0:t1].reshape(cs, H * W)[:, union_flat_idx]
            yield t0, t1, chunk_ram

    traces_robust = solve_traces_robust_irls(design, T, _iter2(), cfg_robust)

    # Evaluate bias on source 1
    mse_ridge = float(np.mean((traces_ridge[0] - c1_gt) ** 2))
    mse_robust = float(np.mean((traces_robust[0] - c1_gt) ** 2))
    corr_ridge = float(np.corrcoef(traces_ridge[0], c1_gt)[0, 1])
    corr_robust = float(np.corrcoef(traces_robust[0], c1_gt)[0, 1])

    print("=== Isolation test: two-source + ghost ===")
    print(f"  Ghost amplitude: 50% of source 1")
    print(f"  Ridge  — source-1 MSE: {mse_ridge:.6f}  corr: {corr_ridge:.4f}")
    print(f"  Robust — source-1 MSE: {mse_robust:.6f}  corr: {corr_robust:.4f}")
    print(f"  MSE ratio (robust/ridge): {mse_robust / (mse_ridge + 1e-12):.3f}")

    if mse_robust < mse_ridge:
        print("  RESULT: robust solver reduces ghost-induced bias on source 1 ✓")
    else:
        print("  RESULT: robust solver does NOT reduce ghost-induced bias "
              "(may indicate kappa needs tuning or ghost amplitude is too low).")
        print("  This is informational — the test does not fail the build.")

    # Source 2 sanity check
    mse2_ridge = float(np.mean((traces_ridge[1] - c2_gt) ** 2))
    mse2_robust = float(np.mean((traces_robust[1] - c2_gt) ** 2))
    print(f"  Ridge  — source-2 MSE: {mse2_ridge:.6f}")
    print(f"  Robust — source-2 MSE: {mse2_robust:.6f}")
    print("=== done ===")


if __name__ == "__main__":
    run()
