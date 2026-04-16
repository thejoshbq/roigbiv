"""
Algebra smoke tests for the sequential pipeline (Phase 0 + 1A).

Run via:
    conda run -n roigbiv python -m roigbiv.pipeline.tests.test_algebra

Tests catch the bug classes identified in Plan agent review §F:
  - DoG sign convention (nuclear shadow score positive at nucleus)
  - torch.svd_lowrank U/V transpose convention
  - Subtraction round-trip recovery of known traces
  - Streaming summary images accuracy
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Test 1: DoG sign at a synthetic nuclear-shadow ring
# ─────────────────────────────────────────────────────────────────────────

def test_dog_nuclear_shadow_sign():
    """Verify DoG = G(σ=6) - G(σ=2) is POSITIVE at nucleus of a realistically-sized neuron.

    Critical: neuron dimensions must be realistic for the chosen sigmas (σ_inner=2,
    σ_outer=6). For GCaMP neurons at 2P magnification, nucleus radius ~4-5 px and
    soma outer radius ~7-8 px give:
      - G(σ=2) at nucleus center ≈ dark nucleus value (σ << nucleus radius)
      - G(σ=6) at nucleus center ≈ soma-weighted average (spans whole cell)
      - DoG positive at nucleus → "nuclear shadow score"
    """
    from roigbiv.pipeline.foundation import compute_nuclear_shadow_map

    H, W = 128, 128
    img = np.full((H, W), 50.0, dtype=np.float32)  # neuropil baseline

    # Cells: nucleus r=4, cytoplasm r=4..8 (thickness 4, matches σ=2 scale)
    cell_centers = [(32, 32), (32, 96), (64, 64), (96, 32), (96, 96)]
    nuc_r2 = 16        # radius 4
    soma_r2 = 64       # radius 8
    for cy, cx in cell_centers:
        ys, xs = np.ogrid[:H, :W]
        r2 = (ys - cy) ** 2 + (xs - cx) ** 2
        cytoplasm = (r2 <= soma_r2) & (r2 > nuc_r2)
        nucleus = (r2 <= nuc_r2)
        img[cytoplasm] = 200.0      # bright cytoplasm
        img[nucleus] = 20.0         # dark nucleus

    dog = compute_nuclear_shadow_map(img)

    for cy, cx in cell_centers:
        nucleus_score = dog[cy, cx]
        # Cytoplasm pixel: radius 6 from center
        ring_y, ring_x = cy + 6, cx
        ring_score = dog[ring_y, ring_x]
        assert nucleus_score > 0, (
            f"Nuclear centroid DoG should be positive (nuclear shadow); "
            f"got {nucleus_score:.3f} at ({cy},{cx})"
        )
        assert ring_score < nucleus_score, (
            f"Cytoplasmic ring DoG should be less than nuclear centroid; "
            f"ring={ring_score:.3f} vs centroid={nucleus_score:.3f}"
        )
    # Also verify far neuropil has ~0 DoG
    neuropil_dog = dog[0, 0]
    assert abs(neuropil_dog) < 1.0, f"Far-neuropil DoG should be ~0; got {neuropil_dog:.3f}"
    print(f"  ✓ DoG nuclear-shadow sign: nucleus>0, ring<nucleus, neuropil~0 "
          f"(nucleus={dog[32,32]:.2f}, ring={dog[38,32]:.2f}, neuropil={neuropil_dog:.3f})")


# ─────────────────────────────────────────────────────────────────────────
# Test 2: torch.svd_lowrank reconstruction via our foundation._binned_svd_gpu
# ─────────────────────────────────────────────────────────────────────────

def test_svd_roundtrip():
    from roigbiv.pipeline.foundation import _binned_svd_gpu

    # Build a matrix of the form we'd see: N_pix × T_bin, rank 30 + noise
    N_pix, T_bin, k_true = 1000, 500, 30
    rng = np.random.RandomState(42)
    U_true = rng.randn(N_pix, k_true).astype(np.float32)
    V_true = rng.randn(T_bin, k_true).astype(np.float32)
    M = U_true @ V_true.T + 0.01 * rng.randn(N_pix, T_bin).astype(np.float32)

    # _binned_svd_gpu expects (T_bin, N_pix); it internally transposes
    M_bin = M.T  # (T_bin, N_pix)
    U, S, V = _binned_svd_gpu(M_bin, n_svd=k_true)
    # Our convention: M_bin (T_bin, N_pix) ≈ V @ diag(S) @ U.T
    # Equivalently: M (N_pix, T_bin) ≈ U @ diag(S) @ V.T
    rec = U @ np.diag(S) @ V.T
    err = np.linalg.norm(M - rec) / np.linalg.norm(M)
    assert err < 0.05, f"SVD reconstruction error too high: {err:.4f}"
    print(f"  ✓ SVD round-trip reconstruction error = {err:.4f} (k={k_true})")


# ─────────────────────────────────────────────────────────────────────────
# Test 3: Summary images on synthetic S memmap
# ─────────────────────────────────────────────────────────────────────────

def test_summary_images():
    from roigbiv.pipeline.foundation import generate_summary_images

    T, H, W = 200, 64, 64
    rng = np.random.RandomState(0)
    # Background noise + 3 correlated blobs
    S = rng.randn(T, H, W).astype(np.float32) * 0.5
    blob_centers = [(16, 16), (32, 48), (48, 32)]
    traces = [np.sin(2 * np.pi * np.arange(T) / 50) * (2 + i)
              for i in range(len(blob_centers))]
    for (cy, cx), trace in zip(blob_centers, traces):
        ys, xs = np.ogrid[:H, :W]
        blob = np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / 10.0).astype(np.float32)
        # Add rank-1 outer product
        S += trace[:, None, None].astype(np.float32) * blob[None, :, :]

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "S.dat"
        S_mm = np.memmap(str(path), dtype=np.float32, mode="w+", shape=(T, H, W))
        S_mm[:] = S
        S_mm.flush()
        del S_mm

        summaries = generate_summary_images(path, (T, H, W), chunk=50)

    mean_expected = S.mean(axis=0)
    std_expected = S.std(axis=0)
    max_expected = S.max(axis=0)

    mean_err = np.abs(summaries["mean"] - mean_expected).max()
    std_err = np.abs(summaries["std"] - std_expected).max()
    max_err = np.abs(summaries["max"] - max_expected).max()

    assert mean_err < 1e-3, f"Mean mismatch {mean_err:.2e}"
    assert std_err < 1e-3, f"Std mismatch {std_err:.2e}"
    assert max_err < 1e-5, f"Max mismatch {max_err:.2e}"

    # Vcorr should peak at blob centers
    vcorr = summaries["vcorr"]
    for cy, cx in blob_centers:
        assert vcorr[cy, cx] > 0.5, (
            f"Vcorr at blob center ({cy},{cx}) = {vcorr[cy,cx]:.3f}; expected >0.5"
        )
    print(f"  ✓ Summary images: mean_err={mean_err:.2e}, "
          f"std_err={std_err:.2e}, max_err={max_err:.2e}, "
          f"vcorr peaks at blob centers > 0.5")


# ─────────────────────────────────────────────────────────────────────────
# Test 4: Source subtraction recovers known traces
# ─────────────────────────────────────────────────────────────────────────

def test_subtraction_recovery():
    from roigbiv.pipeline.subtraction import (
        estimate_spatial_profiles,
        estimate_traces_simultaneous,
        subtract_sources,
        validate_subtraction,
    )
    from roigbiv.pipeline.types import ROI, PipelineConfig

    T, H, W = 500, 64, 64
    rng = np.random.RandomState(1)

    # 3 known ROIs with known traces
    blob_centers = [(16, 16), (32, 48), (48, 32)]
    true_traces = [
        np.sin(2 * np.pi * np.arange(T) / 50.0) + 1.5,
        np.cos(2 * np.pi * np.arange(T) / 75.0) + 1.5,
        np.abs(np.sin(2 * np.pi * np.arange(T) / 100.0)) * 2.0,
    ]
    masks = []
    profiles_true = []
    for cy, cx in blob_centers:
        ys, xs = np.ogrid[:H, :W]
        r2 = (ys - cy) ** 2 + (xs - cx) ** 2
        m = r2 <= 25  # radius ~5
        masks.append(m)
        prof = np.exp(-r2 / 10.0).astype(np.float32)
        prof[~m] = 0
        profiles_true.append(prof)

    # Build residual S: noise + Σ_i w_i * c_i
    S = 0.3 * rng.randn(T, H, W).astype(np.float32)
    for prof, trace in zip(profiles_true, true_traces):
        S += trace[:, None, None].astype(np.float32) * prof[None, :, :]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        path_in = td / "S.dat"
        mm = np.memmap(str(path_in), dtype=np.float32, mode="w+", shape=(T, H, W))
        mm[:] = S
        mm.flush()
        del mm

        rois = [
            ROI(mask=m, label_id=i + 1, source_stage=1, confidence="high",
                gate_outcome="accept")
            for i, m in enumerate(masks)
        ]
        mean_S = S.mean(axis=0)

        profiles = estimate_spatial_profiles(mean_S, rois)

        cfg = PipelineConfig(fs=30, subtract_chunk_frames=200)
        traces_est, _, _ = estimate_traces_simultaneous(path_in, (T, H, W), profiles, cfg)

        # Check trace recovery (correlation with true)
        for i, (est, true) in enumerate(zip(traces_est, true_traces)):
            corr = np.corrcoef(est, true)[0, 1]
            assert corr > 0.95, f"ROI {i} trace recovery corr = {corr:.3f}"

        # Subtract and check residual RMS matches noise level
        path_out = td / "S1.dat"
        subtract_sources(path_in, path_out, (T, H, W), profiles, traces_est, chunk=100)

        S1 = np.memmap(str(path_out), dtype=np.float32, mode="r", shape=(T, H, W))
        for i, m in enumerate(masks):
            rms_mask = S1[:, m].std()
            # Residual at mask should be within 5x the injected noise level
            assert rms_mask < 1.5, f"ROI {i} residual RMS at mask = {rms_mask:.3f} (expected near noise ~0.3)"

        # Validation
        validation = validate_subtraction(path_in, path_out, (T, H, W), rois, traces_est, cfg)
        n_pass = sum(1 for v in validation.values() if v["pass"])
        assert n_pass == len(rois), f"Only {n_pass}/{len(rois)} ROIs passed validation"
        del S1

    print(f"  ✓ Subtraction recovers {len(rois)} known traces (all corr > 0.95, "
          f"validation pass {n_pass}/{len(rois)})")


# ─────────────────────────────────────────────────────────────────────────
# Test 5: Gate 1 decision logic on hand-crafted ROIs
# ─────────────────────────────────────────────────────────────────────────

def test_gate1_decisions():
    from roigbiv.pipeline.gate1 import evaluate_gate1
    from roigbiv.pipeline.types import PipelineConfig

    H, W = 128, 128
    mean_S = np.full((H, W), 10.0, dtype=np.float32)
    vcorr_S = np.zeros((H, W), dtype=np.float32)
    dog_map = np.ones((H, W), dtype=np.float32) * 0.5  # uniform positive

    # Craft 4 candidates:
    # A. Good (accept): circular, area~150, bright
    # B. Too small (reject): area=40
    # C. Too elongated (reject): long strip
    # D. Marginal area (flag): area=70 (within 20 of min=80)

    def make_disk(cy, cx, r, value):
        ys, xs = np.ogrid[:H, :W]
        m = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
        mean_S[m] = value
        return m

    def make_elongated(cy, cx, halflen, halfwid):
        ys, xs = np.ogrid[:H, :W]
        m = (np.abs(ys - cy) <= halfwid) & (np.abs(xs - cx) <= halflen)
        mean_S[m] = 30.0
        return m

    mA = make_disk(30, 30, 7, 30.0)     # area ~ πr² = ~154
    mB = make_disk(30, 80, 3, 30.0)     # area ~ ~28
    mC = make_elongated(80, 30, 12, 1)  # very elongated, area ~ 50, eccentricity ~ 0.99
    mD = make_disk(80, 80, 5, 30.0)     # area ~ π*25 = ~78 — marginal

    cfg = PipelineConfig(fs=30)
    candidates = [mA, mB, mC, mD]
    probs = [0.9, 0.8, 0.7, 0.6]
    rois = evaluate_gate1(candidates, probs, mean_S, vcorr_S, dog_map, cfg)

    outcomes = {i: r.gate_outcome for i, r in enumerate(rois)}
    print(f"  gate1 outcomes: A={outcomes[0]}, B={outcomes[1]}, C={outcomes[2]}, D={outcomes[3]}")
    assert rois[0].gate_outcome == "accept", f"Good disk should accept; got {rois[0].gate_outcome}"
    assert rois[1].gate_outcome == "reject", f"Too-small disk should reject; got {rois[1].gate_outcome}"
    # C has area ~50 AND eccentricity high — at least 2 failures
    assert rois[2].gate_outcome == "reject", f"Elongated should reject; got {rois[2].gate_outcome}"
    # D: area=78 is within margin 20 of min=80 (delta=2); should flag
    assert rois[3].gate_outcome in ("flag", "accept"), f"Marginal should flag/accept; got {rois[3].gate_outcome}"
    print("  ✓ Gate 1 decisions: accept/reject/reject/flag-or-accept (expected)")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

def main():
    tests = [
        ("DoG nuclear-shadow sign", test_dog_nuclear_shadow_sign),
        ("SVD round-trip via _binned_svd_gpu", test_svd_roundtrip),
        ("Summary images (mean/max/std/Vcorr)", test_summary_images),
        ("Source subtraction recovers known traces", test_subtraction_recovery),
        ("Gate 1 accept/flag/reject decisions", test_gate1_decisions),
    ]
    for name, fn in tests:
        print(f"\n▶ {name}")
        fn()
    print("\nAll algebra smoke tests PASSED")


if __name__ == "__main__":
    main()
