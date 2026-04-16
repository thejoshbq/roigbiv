"""
Algebra / smoke tests for Phase 1C (Stage 2 + Gate 2) and Phase 1D (Stage 3 + Gate 3).

Run via:
    conda run -n roigbiv python -m roigbiv.pipeline.tests.test_stage2_stage3

Tests catch:
  - Suite2p stat → mask conversion bounds handling
  - IoU filter correctness against Stage 1
  - Gate 2 correlation / anti-correlation / morphological branches
  - Template bank kinetics + L2 normalization
  - Stage 3 FFT matched filter event detection on synthetic transients
  - Stage 3 event clustering
  - Gate 3 waveform R² fit, rise/decay asymmetry
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Stage 2 / Gate 2
# ─────────────────────────────────────────────────────────────────────────

def test_stat_to_mask_roundtrip():
    """Suite2p-like stat entries → dense mask, via roigbiv.merge.stat_to_mask."""
    from roigbiv.merge import stat_to_mask

    Ly, Lx = 64, 64
    stat = np.array([
        {"ypix": np.array([10, 11, 12]), "xpix": np.array([20, 20, 20])},
        {"ypix": np.array([30, 31]),      "xpix": np.array([40, 41])},
        {"ypix": np.array([63, 64, 65]),  "xpix": np.array([10, 10, 10])},  # last two out-of-bounds
    ], dtype=object)
    # No iscell filter → keep all
    labels = stat_to_mask(stat, Ly, Lx)
    assert labels.shape == (Ly, Lx)
    # Labels 1-indexed
    assert labels[10, 20] == 1 and labels[11, 20] == 1 and labels[12, 20] == 1
    assert labels[30, 40] == 2 and labels[31, 41] == 2
    # Third entry: only ypix=63 should survive bounds check
    assert labels[63, 10] == 3
    # Out-of-bounds pixels not written
    assert labels[0, 0] == 0
    print("  [PASS] test_stat_to_mask_roundtrip")


def test_iou_filter_against_stage1():
    """Stage 2 IoU filter: discard rediscoveries, keep novel detections."""
    from roigbiv.pipeline.stage2 import _filter_against_stage1

    H, W = 32, 32
    s1_masks = []
    m = np.zeros((H, W), dtype=bool); m[5:10, 5:10] = True
    s1_masks.append(m)

    # Candidate A: full overlap with S1[0] → IoU = 1.0 → discard
    a = np.zeros((H, W), dtype=bool); a[5:10, 5:10] = True
    # Candidate B: disjoint → IoU = 0 → keep
    b = np.zeros((H, W), dtype=bool); b[20:25, 20:25] = True
    # Candidate C: partial overlap, IoU ~= 0.2 → keep (below 0.3 threshold)
    c = np.zeros((H, W), dtype=bool); c[9:14, 9:14] = True
    # Candidate D: large overlap, IoU > 0.3 → discard
    d = np.zeros((H, W), dtype=bool); d[5:10, 6:11] = True

    candidates = [(i, m, 0.9) for i, m in enumerate([a, b, c, d])]
    kept = _filter_against_stage1(candidates, s1_masks, iou_threshold=0.3)
    kept_indices = {idx for idx, _, _ in kept}
    assert 0 not in kept_indices  # A discarded
    assert 1 in kept_indices      # B kept
    assert 2 in kept_indices      # C kept (low IoU)
    assert 3 not in kept_indices  # D discarded
    print(f"  [PASS] test_iou_filter_against_stage1 (kept: {sorted(kept_indices)})")


def test_gate2_correlation_branches():
    """Gate 2: redundant (|r|>0.7) rejects, anticorrelation (r<=-0.5) rejects,
    orthogonal accepts."""
    from roigbiv.pipeline.gate2 import evaluate_gate2
    from roigbiv.pipeline.types import ROI, PipelineConfig

    cfg = PipelineConfig(fs=30.0)
    H, W = 64, 64
    T = 500

    # Stage 1 ROI with trace
    s1_mask = np.zeros((H, W), dtype=bool); s1_mask[20:25, 20:25] = True
    rng = np.random.default_rng(0)
    s1_trace = rng.normal(size=T).astype(np.float32)
    s1_roi = ROI(
        mask=s1_mask, label_id=1, source_stage=1, confidence="high",
        gate_outcome="accept", area=25, solidity=1.0, eccentricity=0.0,
        nuclear_shadow_score=0.0, soma_surround_contrast=0.0,
        trace=s1_trace,
    )

    def make_cand(label, trace, cy, cx):
        m = np.zeros((H, W), dtype=bool)
        # Compact disk at (cy, cx) with radius 6 → area ~= 110 (above gate2_min_area=60)
        for dy in range(-7, 8):
            for dx in range(-7, 8):
                y, x = cy + dy, cx + dx
                if 0 <= y < H and 0 <= x < W and (dy * dy + dx * dx) <= 36:
                    m[y, x] = True
        return ROI(
            mask=m, label_id=label, source_stage=2, confidence="moderate",
            gate_outcome="accept", area=int(m.sum()), solidity=1.0,
            eccentricity=0.2, nuclear_shadow_score=0.0,
            soma_surround_contrast=0.0, trace=trace.astype(np.float32),
        )

    # (a) correlated (r=1) within 20 px → REJECT
    cand_a = make_cand(100, s1_trace.copy(), cy=23, cx=23)
    # (b) anti-correlated within 20 px → REJECT
    cand_b = make_cand(101, -s1_trace.copy() * 0.9, cy=30, cx=30)
    # (c) orthogonal, far (>20 px) → ACCEPT
    cand_c = make_cand(102, rng.normal(size=T).astype(np.float32), cy=55, cx=55)

    evaluate_gate2([cand_a, cand_b, cand_c], [s1_roi], cfg)
    assert cand_a.gate_outcome == "reject", f"expected reject, got {cand_a.gate_outcome}"
    assert cand_b.gate_outcome == "reject", f"expected reject, got {cand_b.gate_outcome}"
    assert cand_c.gate_outcome == "accept", f"expected accept, got {cand_c.gate_outcome}"
    print("  [PASS] test_gate2_correlation_branches")


# ─────────────────────────────────────────────────────────────────────────
# Stage 3 / Gate 3
# ─────────────────────────────────────────────────────────────────────────

def test_template_bank_sanity():
    """Template bank: returns 3 L2-normalized waveforms for each indicator family."""
    from roigbiv.pipeline.stage3_templates import build_template_bank

    for tau in (1.0, 0.5):
        bank = build_template_bank(fs=30.0, tau=tau)
        assert len(bank) == 3, f"expected 3 templates, got {len(bank)}"
        for name, wf in bank:
            norm = np.linalg.norm(wf)
            assert abs(norm - 1.0) < 1e-5, f"{name}: L2 norm {norm}"
            # Argmax should be in the first 30% of the waveform (fast rise)
            assert np.argmax(wf) < 0.3 * len(wf), f"{name}: peak not in first 30%"
            # Length at least 5*tau_decay*fs = 150 (for tau=1.0) or 75 (for tau=0.5)
            expected_min_len = int(5 * 0.5 * 30)  # loosest: tau_decay=0.5
            assert len(wf) >= expected_min_len, f"{name}: too short ({len(wf)})"
    print("  [PASS] test_template_bank_sanity")


def test_stage3_synthetic_event_detection():
    """Inject known events at known (y, x, t) into a residual memmap; confirm
    Stage 3 recovers most of them via FFT template matching."""
    from roigbiv.pipeline.stage3 import run_stage3
    from roigbiv.pipeline.stage3_templates import build_template_bank
    from roigbiv.pipeline.types import FOVData, PipelineConfig

    T, H, W = 400, 64, 64
    fs = 30.0
    tau = 1.0

    # Build template, make a residual with 3 localized events
    bank = build_template_bank(fs=fs, tau=tau)
    tmpl = bank[0][1]  # 'single_gcamp6s'

    rng = np.random.default_rng(42)
    residual = rng.normal(scale=0.5, size=(T, H, W)).astype(np.float32)

    # Inject 3 events: well-separated in both space and time
    truth_events = [(10, 10, 50), (40, 40, 150), (55, 8, 300)]
    for cy, cx, t0 in truth_events:
        # Spread amplitude over a ~3 px radius disk (compact, clustering-friendly)
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if dy * dy + dx * dx > 16:
                    continue
                y, x = cy + dy, cx + dx
                if 0 <= y < H and 0 <= x < W:
                    end = min(t0 + len(tmpl), T)
                    # Strong signal with rapid spatial falloff
                    amp = 15.0 * np.exp(-(dy * dy + dx * dx) / 8.0)
                    residual[t0:end, y, x] += amp * tmpl[: end - t0]

    with tempfile.TemporaryDirectory() as td:
        rpath = Path(td) / "test_residual.dat"
        mm = np.memmap(str(rpath), dtype=np.float32, mode="w+", shape=(T, H, W))
        mm[:] = residual
        mm.flush()
        del mm

        cfg = PipelineConfig(
            fs=fs, tau=tau, reconstruct_chunk=200,
            template_threshold=4.0,
            spatial_pool_radius=8,
            cluster_distance=12,
            min_event_separation=2.0,
            stage3_pixel_chunk_rows=16,
        )
        # Minimal FOVData for Stage 3
        fov = FOVData(
            raw_path=Path("dummy.tif"),
            output_dir=Path(td),
            data_bin_path=Path(td) / "data.bin",
            shape=(T, H, W),
            residual_S_path=rpath,
            mean_M=np.zeros((H, W), dtype=np.float32),
            mean_S=np.zeros((H, W), dtype=np.float32),
            max_S=np.zeros((H, W), dtype=np.float32),
            std_S=np.ones((H, W), dtype=np.float32),
            vcorr_S=np.zeros((H, W), dtype=np.float32),
            dog_map=np.zeros((H, W), dtype=np.float32),
            mean_L=np.zeros((H, W), dtype=np.float32),
            k_background=30,
        )
        rois = run_stage3(rpath, fov, bank, cfg, starting_label_id=1)

    assert len(rois) >= 2, f"expected ≥ 2 candidates, got {len(rois)}"
    # Match each truth event to the nearest recovered centroid
    recovered = [(r.features["centroid_y"], r.features["centroid_x"]) for r in rois]
    n_matched = 0
    for cy, cx, _ in truth_events:
        for ry, rx in recovered:
            if abs(ry - cy) <= 4 and abs(rx - cx) <= 4:
                n_matched += 1
                break
    assert n_matched >= 2, f"only {n_matched}/3 truth events matched"
    print(f"  [PASS] test_stage3_synthetic_event_detection "
          f"({n_matched}/3 truth events matched, {len(rois)} total candidates)")


def test_stage3_clustering():
    """Hierarchical clustering groups nearby events; distant events stay separate."""
    from roigbiv.pipeline.stage3 import _cluster_events_spatial

    # 20 events across 3 spatial clusters within 5 px of each other
    rng = np.random.default_rng(1)
    centers = [(10.0, 10.0), (40.0, 40.0), (5.0, 50.0)]
    ys, xs = [], []
    for cy, cx in centers:
        for _ in range(5):
            ys.append(cy + rng.uniform(-3, 3))
            xs.append(cx + rng.uniform(-3, 3))
    # Add 5 isolated events (far from any cluster)
    for cy in [60, 62, 63, 30, 25]:
        ys.append(cy + 0.1)
        xs.append(cy + 0.1)
    cluster_ids = _cluster_events_spatial(np.array(ys), np.array(xs),
                                          distance_threshold=8.0)
    # Expect at least the 3 dense clusters to be identified as clusters with multiple members
    counts = np.bincount(cluster_ids)
    big_clusters = np.sum(counts >= 3)
    assert big_clusters >= 3, f"expected ≥ 3 big clusters, got {big_clusters}"
    print(f"  [PASS] test_stage3_clustering (found {big_clusters} big clusters)")


def test_gate3_waveform_r2():
    """Gate 3: accepts waveforms matching the template, rejects symmetric waveforms."""
    from roigbiv.pipeline.gate3 import _waveform_r2, _rise_decay_ratio
    from roigbiv.pipeline.stage3_templates import build_template_bank

    fs, tau = 30.0, 1.0
    bank = build_template_bank(fs, tau)
    templates = [wf for _, wf in bank]
    tmpl = templates[0]

    # Waveform matching template → high R²
    wf_match = tmpl.copy() * 5.0
    r2, k = _waveform_r2(wf_match, templates)
    assert r2 > 0.9, f"expected r² > 0.9 on template itself, got {r2}"

    # Noisy version still should be >0.7
    rng = np.random.default_rng(0)
    wf_noisy = wf_match + rng.normal(scale=0.2, size=wf_match.shape).astype(np.float32)
    r2, _ = _waveform_r2(wf_noisy, templates)
    assert r2 > 0.7, f"expected r² > 0.7 on noisy template, got {r2}"

    # Symmetric Gaussian should have low R² AND symmetric rise/decay
    t = np.arange(len(tmpl), dtype=np.float32)
    gauss = np.exp(-((t - len(tmpl) / 2) ** 2) / (2 * 10 ** 2))
    rd = _rise_decay_ratio(gauss)
    assert rd >= 0.5 or not np.isfinite(rd), f"symmetric Gaussian rise/decay {rd}; expected ≥ 0.5"

    # Real transient should have low rise/decay ratio
    rd_match = _rise_decay_ratio(wf_match)
    assert rd_match < 0.5, f"expected rise/decay < 0.5 for real transient, got {rd_match}"
    print(f"  [PASS] test_gate3_waveform_r2 (r² match={r2:.3f}, rd_match={rd_match:.3f}, rd_gauss={rd:.3f})")


# ─────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        test_stat_to_mask_roundtrip,
        test_iou_filter_against_stage1,
        test_gate2_correlation_branches,
        test_template_bank_sanity,
        test_stage3_clustering,
        test_gate3_waveform_r2,
        test_stage3_synthetic_event_detection,   # slower (uses GPU)
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
