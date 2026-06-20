"""Gate 1 — peak-count merge check (spec §6).

A high ``max_area`` recovers large somata but also admits 2-soma merges. The
peak-count check must distinguish a single large soma (1 intensity peak →
accept) from a merge (≥2 peaks → flag, never silently accept), and stay inert
on masks below ``gate1_merge_peak_min_area``.
"""
import numpy as np

from roigbiv.pipeline.gate1 import evaluate_gate1, count_mask_peaks
from roigbiv.pipeline.types import PipelineConfig

H = W = 256


def _disk(cy, cx, r):
    yy, xx = np.ogrid[:H, :W]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _gauss(cy, cx, sigma):
    yy, xx = np.ogrid[:H, :W]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))


def _prism_cfg(**over):
    base = dict(
        min_area=900, max_area=9000, min_solidity=0.40, max_eccentricity=0.97,
        min_contrast=0.10, gate1_merge_peak_min_area=4000,
        gate1_merge_peak_min_separation=28,
    )
    base.update(over)
    return PipelineConfig(**base)


def _run(mask, intensity, cfg):
    """Evaluate a single candidate; return its ROI."""
    rois = evaluate_gate1(
        [mask], [0.9], intensity.astype(np.float32),
        np.zeros((H, W), np.float32), np.zeros((H, W), np.float32), cfg,
    )
    assert len(rois) == 1
    return rois[0]


def test_count_mask_peaks_single_vs_merge():
    mask = _disk(128, 128, 48)
    single = _gauss(128, 128, 12)
    merge = _gauss(128, 108, 10) + _gauss(128, 148, 10)   # centers ±20 (sep 40 > 28)
    assert count_mask_peaks(mask, single, 28) == 1
    assert count_mask_peaks(mask, merge, 28) == 2


def test_single_large_soma_accepts():
    # Disk r=40 → area ~5027 px (> merge threshold), one intensity peak.
    mask = _disk(128, 128, 40)
    roi = _run(mask, _gauss(128, 128, 12) * 100.0, _prism_cfg())
    assert roi.area > 4000
    assert roi.gate_outcome == "accept"
    assert roi.features["mask_peak_count"] == 1
    assert not any("merge_peaks" in r for r in roi.gate_reasons)


def test_two_soma_merge_demoted_to_flag():
    # Clean morphology (filled disk: high solidity, low ecc) so the ROI would
    # ACCEPT on morphology alone — isolating the merge demotion. Two intensity
    # peaks inside the one mask.
    mask = _disk(128, 128, 48)                            # area ~7238 px
    intensity = (_gauss(128, 108, 10) + _gauss(128, 148, 10)) * 100.0
    roi = _run(mask, intensity, _prism_cfg())
    assert roi.area > 4000
    assert roi.features["mask_peak_count"] == 2
    assert roi.gate_outcome == "flag"
    assert any(r == "merge_peaks:2" for r in roi.gate_reasons)


def test_below_threshold_mask_skips_peak_check():
    # Disk r=25 → area ~1963 px (< gate1_merge_peak_min_area) — check is inert.
    mask = _disk(128, 128, 25)
    intensity = (_gauss(128, 120, 6) + _gauss(128, 136, 6)) * 100.0  # 2 peaks present
    roi = _run(mask, intensity, _prism_cfg())
    assert 900 < roi.area < 4000
    assert roi.features["mask_peak_count"] == 0      # not computed
    assert roi.gate_outcome == "accept"
