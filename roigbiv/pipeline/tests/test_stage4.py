"""
Algebra / smoke tests for Phase 1E (Stage 4 + Gate 4).

Run via:
    conda run -n roigbiv python -m roigbiv.pipeline.tests.test_stage4

Tests catch:
  - Bandpass filter recovers the intended frequency band
  - Correlation contrast discriminates soma-like local correlation vs
    broad neuropil correlation (catches kernel orientation / normalization
    bugs in the spatial-convolution implementation)
  - Cross-window IoU merge keeps the highest-score duplicate
  - Gate 4 motion rejection (unique to Gate 4)
  - Gate 4 anti-correlation cascade defense
  - Gate 4 mean_M intensity-floor filter
  - Gate 4 survivors always get gate_outcome="flag" + confidence="requires_review"
    (never "accept")
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Stage 4 algorithm components
# ─────────────────────────────────────────────────────────────────────────

def test_detrend_removes_linear_drift():
    """Per-pixel linear detrend: a pure ramp becomes near-zero."""
    from roigbiv.pipeline.stage4 import detrend_to_memmap

    T, H, W = 200, 16, 16
    # Each pixel has a linear drift plus pixel-specific offset
    t = np.arange(T, dtype=np.float32)
    drift = np.outer(t, np.ones(H * W)).reshape(T, H, W)
    offsets = np.arange(H * W, dtype=np.float32).reshape(H, W)
    data = drift + offsets[None, :, :]

    with tempfile.TemporaryDirectory() as td:
        in_p = Path(td) / "in.dat"
        out_p = Path(td) / "out.dat"
        mm = np.memmap(str(in_p), dtype=np.float32, mode="w+", shape=(T, H, W))
        mm[:] = data.astype(np.float32)
        mm.flush(); del mm

        detrend_to_memmap(in_p, out_p, (T, H, W), chunk_rows=4)
        out = np.memmap(str(out_p), dtype=np.float32, mode="r", shape=(T, H, W))
        resid = np.asarray(out)
    max_abs = float(np.max(np.abs(resid)))
    assert max_abs < 1e-3, f"detrended residual magnitude {max_abs}, expected ~0"
    print(f"  [PASS] test_detrend_removes_linear_drift (max |resid| = {max_abs:.2e})")


def test_bandpass_isolation():
    """bandpass_to_memmap isolates in-band frequencies and rejects out-of-band."""
    from roigbiv.pipeline.stage4 import bandpass_to_memmap

    fs = 30.0
    T = 1800           # 60s of data
    H, W = 4, 4
    t = np.arange(T, dtype=np.float32) / fs

    f_tonic = 0.2      # in medium band [0.1, 1.0]
    f_phasic = 5.0     # out of band — should be removed
    trace_full = np.sin(2 * np.pi * f_tonic * t) + 0.8 * np.sin(2 * np.pi * f_phasic * t)
    data = np.broadcast_to(trace_full[:, None, None], (T, H, W)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        in_p = Path(td) / "in.dat"
        out_p = Path(td) / "out.dat"
        mm = np.memmap(str(in_p), dtype=np.float32, mode="w+", shape=(T, H, W))
        mm[:] = data
        mm.flush(); del mm

        bandpass_to_memmap(in_p, out_p, (T, H, W), fs=fs, low=0.1, high=1.0,
                           order=4, chunk_rows=2)
        out = np.memmap(str(out_p), dtype=np.float32, mode="r", shape=(T, H, W))
        filtered = np.asarray(out)

    filt_trace = filtered[:, 0, 0]
    tonic_ref = np.sin(2 * np.pi * f_tonic * t).astype(np.float32)
    # Trim transients at edges (first/last 5%)
    s = int(0.05 * T); e = int(0.95 * T)
    r_tonic = float(np.corrcoef(filt_trace[s:e], tonic_ref[s:e])[0, 1])
    assert r_tonic > 0.9, f"expected r(filtered, tonic) > 0.9, got {r_tonic:.3f}"

    # Phasic component should be heavily attenuated
    phasic_ref = np.sin(2 * np.pi * f_phasic * t).astype(np.float32)
    r_phasic = abs(float(np.corrcoef(filt_trace[s:e], phasic_ref[s:e])[0, 1]))
    assert r_phasic < 0.1, f"expected |r(filtered, phasic)| < 0.1, got {r_phasic:.3f}"
    print(f"  [PASS] test_bandpass_isolation "
          f"(r_tonic={r_tonic:.3f}, r_phasic={r_phasic:.3f})")


def test_uniform_disk_kernel_counts():
    """Disk kernel sums to 1 and has the correct pixel count."""
    from roigbiv.pipeline.stage4 import uniform_disk_kernel

    k, n = uniform_disk_kernel(6, exclude_center=False)
    assert abs(k.sum() - 1.0) < 1e-5, f"kernel sum = {k.sum()}"
    # Count agrees with kernel nnz
    assert n == int((k > 0).sum())
    # radius=1 including center: 5 pixels (+, center, -, left, right) — NESW + center
    k1, n1 = uniform_disk_kernel(1, exclude_center=False)
    assert n1 == 5, f"radius=1 disk should have 5 pixels; got {n1}"
    # Excluding center: 4
    k1nc, n1nc = uniform_disk_kernel(1, exclude_center=True)
    assert n1nc == 4, f"radius=1 ring should have 4 pixels; got {n1nc}"
    print(f"  [PASS] test_uniform_disk_kernel_counts (r=6 n={n}, r=1 n={n1}, r=1(nc)={n1nc})")


def test_correlation_contrast_soma_vs_neuropil():
    """Somata show HIGH contrast; neuropil shows LOW contrast.

    Constructs a 64×64 compressed matrix with:
      - Broad "neuropil" signal shared by every pixel
      - Three circular "soma" patches whose pixels share a distinct local signal
      - Per-pixel independent noise

    Soma pixels should correlate strongly with inner neighbors (within the
    patch) and weakly with outer neighbors (outside the patch) → high contrast.
    Neuropil pixels correlate with everything via the broad signal → low
    contrast.
    """
    from roigbiv.pipeline.stage4 import compute_correlation_contrast

    rng = np.random.default_rng(0)
    H, W = 64, 64
    D = 80
    N = H * W

    # Neuropil: broad shared signal at moderate amplitude + per-pixel noise
    neuropil = rng.normal(size=D).astype(np.float32) * 1.0
    noise = rng.normal(size=(N, D)).astype(np.float32) * 0.5
    compressed = noise + neuropil[None, :]

    # Three soma patches, each with a stronger local signal
    centers = [(15, 15), (32, 40), (50, 20)]
    soma_radius = 4  # inside the inner_r=6 kernel radius used in the check below
    for cy, cx in centers:
        soma_signal = rng.normal(size=D).astype(np.float32) * 3.0
        for y in range(H):
            for x in range(W):
                if (y - cy) ** 2 + (x - cx) ** 2 <= soma_radius ** 2:
                    compressed[y * W + x] += soma_signal

    contrast, inner_corr = compute_correlation_contrast(compressed, (H, W),
                                                       inner_radius=6,
                                                       outer_radius=15)
    # Soma-center pixels should have markedly higher contrast than
    # neuropil pixels far from any soma.
    soma_vals = [float(contrast[cy, cx]) for cy, cx in centers]
    neuro_pts = [(5, 55), (55, 55), (5, 5)]
    neuro_vals = [float(contrast[y, x]) for y, x in neuro_pts]

    for cy, cx in centers:
        v = float(contrast[cy, cx])
        assert v > 0.10, f"soma at ({cy},{cx}) contrast {v:.3f} < 0.10"
    max_neuro = max(abs(v) for v in neuro_vals)
    assert max_neuro < 0.05, f"neuropil contrast {max_neuro:.3f} >= 0.05"
    print(f"  [PASS] test_correlation_contrast_soma_vs_neuropil "
          f"(soma={[round(v, 3) for v in soma_vals]}, "
          f"neuro={[round(v, 3) for v in neuro_vals]})")


def test_cluster_contrast_map():
    """cluster_contrast_map finds the embedded circular hotspots and rejects too-small/-large ones."""
    from roigbiv.pipeline.stage4 import cluster_contrast_map
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(fs=30.0)
    H, W = 64, 64
    contrast = np.zeros((H, W), dtype=np.float32)
    inner = np.zeros((H, W), dtype=np.float32)

    # Cluster A: proper soma-sized (radius 6 → area ≈ 113)
    cy, cx = 20, 20
    for y in range(H):
        for x in range(W):
            if (y - cy) ** 2 + (x - cx) ** 2 <= 36:
                contrast[y, x] = 0.2
                inner[y, x] = 0.8
    # Cluster B: too small (radius 2 → area ~ 13)
    cy, cx = 50, 50
    for y in range(H):
        for x in range(W):
            if (y - cy) ** 2 + (x - cx) ** 2 <= 4:
                contrast[y, x] = 0.2
                inner[y, x] = 0.8

    cands = cluster_contrast_map(contrast, inner, "test", cfg)
    assert len(cands) == 1, f"expected 1 cluster (size filter removes B), got {len(cands)}"
    c = cands[0]
    assert c["bandpass_window"] == "test"
    assert 80 <= c["area"] <= 350
    # mean correlation contrast should be the threshold value (0.2)
    assert abs(c["corr_contrast"] - 0.2) < 1e-3
    print(f"  [PASS] test_cluster_contrast_map (kept 1/2; area={c['area']}, "
          f"corr_contrast={c['corr_contrast']:.3f})")


def test_merge_across_windows_prefers_highest_score():
    """When two candidates overlap strongly across windows, the highest-scoring
    one wins and records both windows detected it."""
    from roigbiv.pipeline.stage4 import merge_across_windows

    H, W = 32, 32
    mask_a = np.zeros((H, W), dtype=bool); mask_a[10:15, 10:15] = True
    # mask_b has ~80% overlap with mask_a (IoU > 0.3)
    mask_b = np.zeros((H, W), dtype=bool); mask_b[10:15, 11:16] = True
    # mask_c is disjoint
    mask_c = np.zeros((H, W), dtype=bool); mask_c[20:25, 20:25] = True

    cand_fast = {"mask": mask_a, "corr_contrast": 0.15, "bandpass_window": "fast",
                 "area": int(mask_a.sum()), "solidity": 1.0, "eccentricity": 0.0,
                 "centroid_y": 12.0, "centroid_x": 12.0, "mean_intra_corr": 0.5}
    cand_medium_dup = {"mask": mask_b, "corr_contrast": 0.25, "bandpass_window": "medium",
                       "area": int(mask_b.sum()), "solidity": 1.0, "eccentricity": 0.0,
                       "centroid_y": 12.0, "centroid_x": 13.0, "mean_intra_corr": 0.6}
    cand_slow_novel = {"mask": mask_c, "corr_contrast": 0.12, "bandpass_window": "slow",
                       "area": int(mask_c.sum()), "solidity": 1.0, "eccentricity": 0.0,
                       "centroid_y": 22.0, "centroid_x": 22.0, "mean_intra_corr": 0.4}

    merged = merge_across_windows(
        [[cand_fast], [cand_medium_dup], [cand_slow_novel]],
        iou_threshold=0.3,
    )
    assert len(merged) == 2, f"expected 2 merged candidates, got {len(merged)}"

    # Winning duplicate should be medium (higher corr_contrast)
    top = max(merged, key=lambda c: c["corr_contrast"])
    assert top["bandpass_window"] == "medium"
    assert set(top["bandpass_windows_detected"]) == {"fast", "medium"}
    assert top["n_windows_detected"] == 2

    other = [c for c in merged if c is not top][0]
    assert other["bandpass_window"] == "slow"
    assert other["n_windows_detected"] == 1
    print(f"  [PASS] test_merge_across_windows_prefers_highest_score "
          f"(kept {len(merged)}; winner windows={top['bandpass_windows_detected']})")


# ─────────────────────────────────────────────────────────────────────────
# Gate 4
# ─────────────────────────────────────────────────────────────────────────

def _make_cand(label, H, W, cy, cx, radius, trace, corr_contrast=0.2,
               solidity=1.0, eccentricity=0.0):
    from roigbiv.pipeline.types import ROI
    mask = np.zeros((H, W), dtype=bool)
    area = 0
    for y in range(H):
        for x in range(W):
            if (y - cy) ** 2 + (x - cx) ** 2 <= radius * radius:
                mask[y, x] = True
                area += 1
    return ROI(
        mask=mask,
        label_id=label,
        source_stage=4,
        confidence="requires_review",
        gate_outcome="flag",
        area=area,
        solidity=solidity,
        eccentricity=eccentricity,
        nuclear_shadow_score=0.0,
        soma_surround_contrast=0.0,
        corr_contrast=corr_contrast,
        trace=trace.astype(np.float32),
        features={"centroid_y": float(cy), "centroid_x": float(cx)},
    )


def _bright_mean_M(H, W):
    """Uniformly bright mean_M so the intensity floor is easily passed."""
    return np.ones((H, W), dtype=np.float32) * 100.0


def test_gate4_motion_rejection():
    """Candidate whose raw trace tracks motion → rejected with 'motion' reason."""
    from roigbiv.pipeline.gate4 import evaluate_gate4
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(fs=30.0)
    H, W = 64, 64
    T = 500
    rng = np.random.default_rng(1)
    motion_x = rng.normal(size=T).astype(np.float32)
    motion_y = rng.normal(size=T).astype(np.float32)

    # Trace proportional to motion_x → should be rejected
    cand = _make_cand(1, H, W, cy=32, cx=32, radius=10,
                      trace=motion_x.copy(), corr_contrast=0.3)
    evaluate_gate4([cand], prior_rois=[],
                   mean_M=_bright_mean_M(H, W),
                   motion_x=motion_x, motion_y=motion_y, cfg=cfg)
    assert cand.gate_outcome == "reject", (
        f"expected reject for motion-correlated candidate, got {cand.gate_outcome}")
    assert any("motion" in reason for reason in cand.gate_reasons), (
        f"expected 'motion' in gate_reasons {cand.gate_reasons}")
    print(f"  [PASS] test_gate4_motion_rejection (reasons={cand.gate_reasons})")


def test_gate4_anticorr_cascade():
    """Candidate anti-correlated with a nearby prior ROI → rejected."""
    from roigbiv.pipeline.gate4 import evaluate_gate4
    from roigbiv.pipeline.types import PipelineConfig, ROI

    cfg = PipelineConfig(fs=30.0)
    H, W = 64, 64
    T = 500
    rng = np.random.default_rng(2)
    prior_trace = rng.normal(size=T).astype(np.float32)

    prior_mask = np.zeros((H, W), dtype=bool); prior_mask[30:35, 30:35] = True
    prior = ROI(
        mask=prior_mask, label_id=999, source_stage=2, confidence="high",
        gate_outcome="accept", area=25, solidity=1.0, eccentricity=0.0,
        nuclear_shadow_score=0.0, soma_surround_contrast=0.0,
        trace=prior_trace,
    )
    # Candidate within 20px with anti-correlated trace
    cand = _make_cand(1, H, W, cy=34, cx=34, radius=10,
                      trace=-prior_trace.copy(), corr_contrast=0.3)
    motion = np.zeros(T, dtype=np.float32)
    evaluate_gate4([cand], prior_rois=[prior],
                   mean_M=_bright_mean_M(H, W),
                   motion_x=motion, motion_y=motion, cfg=cfg)
    assert cand.gate_outcome == "reject", (
        f"expected reject for anticorr cascade, got {cand.gate_outcome}")
    assert any("anticorr" in reason for reason in cand.gate_reasons), (
        f"expected 'anticorr' reason, got {cand.gate_reasons}")
    print(f"  [PASS] test_gate4_anticorr_cascade (reasons={cand.gate_reasons})")


def test_gate4_intensity_floor():
    """Candidate sitting in a sub-percentile-brightness region → rejected."""
    from roigbiv.pipeline.gate4 import evaluate_gate4
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(fs=30.0)
    H, W = 64, 64
    T = 500
    rng = np.random.default_rng(3)
    motion = np.zeros(T, dtype=np.float32)

    # Mostly-bright image with a small dim stripe (<25% of pixels) so the
    # 25th-percentile falls in the bright region (≈100). A candidate placed
    # entirely inside the dim stripe has mean ≈ 0.1 < 100 → fails.
    mean_M = np.full((H, W), 100.0, dtype=np.float32)
    mean_M[:, 48:]  = 0.1          # right 16 columns dim → 25% dim, 75% bright
    # Stripe is exactly 25% of pixels; reduce to 20% so p25 is clearly bright
    mean_M[:, 48:51] = 100.0       # bump a few cols back to bright → ~20% dim

    cand_dark = _make_cand(1, H, W, cy=32, cx=57, radius=5,
                           trace=rng.normal(size=T).astype(np.float32),
                           corr_contrast=0.3)
    evaluate_gate4([cand_dark], prior_rois=[],
                   mean_M=mean_M, motion_x=motion, motion_y=motion, cfg=cfg)
    assert cand_dark.gate_outcome == "reject", (
        f"expected reject; got {cand_dark.gate_outcome}, reasons={cand_dark.gate_reasons}")
    assert any("intensity" in reason for reason in cand_dark.gate_reasons), (
        f"expected 'intensity' reason; got {cand_dark.gate_reasons}")
    print(f"  [PASS] test_gate4_intensity_floor (reasons={cand_dark.gate_reasons})")


def test_gate4_all_pass_is_flag_not_accept():
    """A candidate passing every Gate 4 check gets gate_outcome='flag',
    confidence='requires_review'. Never 'accept'."""
    from roigbiv.pipeline.gate4 import evaluate_gate4
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(fs=30.0)
    H, W = 64, 64
    T = 500
    rng = np.random.default_rng(4)
    trace = rng.normal(size=T).astype(np.float32)
    motion = rng.normal(size=T).astype(np.float32)    # independent of trace

    # High corr_contrast, good shape, bright region, no prior ROIs.
    cand = _make_cand(1, H, W, cy=32, cx=32, radius=10,
                      trace=trace, corr_contrast=0.3,
                      solidity=0.95, eccentricity=0.2)
    evaluate_gate4([cand], prior_rois=[],
                   mean_M=_bright_mean_M(H, W),
                   motion_x=motion, motion_y=motion, cfg=cfg)
    assert cand.gate_outcome == "flag", (
        f"expected 'flag' for all-pass candidate, got '{cand.gate_outcome}' "
        f"reasons={cand.gate_reasons}")
    assert cand.confidence == "requires_review", (
        f"expected confidence='requires_review', got '{cand.confidence}'")
    assert cand.gate_reasons == []
    print("  [PASS] test_gate4_all_pass_is_flag_not_accept")


def test_gate4_low_contrast_rejection():
    """A candidate below the corr_contrast threshold is rejected."""
    from roigbiv.pipeline.gate4 import evaluate_gate4
    from roigbiv.pipeline.types import PipelineConfig

    cfg = PipelineConfig(fs=30.0)
    H, W = 64, 64
    T = 500
    rng = np.random.default_rng(5)
    cand = _make_cand(1, H, W, cy=32, cx=32, radius=10,
                      trace=rng.normal(size=T).astype(np.float32),
                      corr_contrast=0.05,     # below threshold of 0.10
                      solidity=0.95, eccentricity=0.2)
    motion = np.zeros(T, dtype=np.float32)
    evaluate_gate4([cand], prior_rois=[],
                   mean_M=_bright_mean_M(H, W),
                   motion_x=motion, motion_y=motion, cfg=cfg)
    assert cand.gate_outcome == "reject"
    assert any("corr_contrast" in reason for reason in cand.gate_reasons), (
        f"expected 'corr_contrast' reason; got {cand.gate_reasons}")
    print(f"  [PASS] test_gate4_low_contrast_rejection (reasons={cand.gate_reasons})")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        test_detrend_removes_linear_drift,
        test_bandpass_isolation,
        test_uniform_disk_kernel_counts,
        test_correlation_contrast_soma_vs_neuropil,
        test_cluster_contrast_map,
        test_merge_across_windows_prefers_highest_score,
        test_gate4_motion_rejection,
        test_gate4_anticorr_cascade,
        test_gate4_intensity_floor,
        test_gate4_all_pass_is_flag_not_accept,
        test_gate4_low_contrast_rejection,
    ]
    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            traceback.print_exc()
            failed.append(test.__name__)
    print()
    if failed:
        print(f"FAILED: {failed}")
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed.")
