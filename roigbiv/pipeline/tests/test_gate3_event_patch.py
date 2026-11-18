"""Gate 3 judges an event's waveform on a 3x3 patch at the event's own pixel, not on the candidate's pooled disk.

The disk (radius ``spatial_pool_radius``) also averages in neighbouring tissue; a slow signal there drowns the
sharp transient the pixel detector found, so the disk-mean waveform peaks at the end of the window and never
decays. The disk stays the ROI footprint; only the waveform check moves to the event pixel.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from roigbiv.pipeline.gate3 import evaluate_gate3
from roigbiv.pipeline.residual import ResidualView
from roigbiv.pipeline.stage3 import run_stage3
from roigbiv.pipeline.stage3_templates import build_template_bank
from roigbiv.pipeline.types import FOVData, PipelineConfig

T, H, W = 400, 48, 48
FS, TAU = 7.5, 1.0
EVENT_YX, EVENT_T = (24, 24), 150


def _residual() -> np.ndarray:
    rng = np.random.default_rng(7)
    res = rng.normal(scale=0.3, size=(T, H, W)).astype(np.float32)
    tmpl = build_template_bank(FS, TAU)[0][1]
    tmpl = tmpl / tmpl.max()
    y, x = EVENT_YX
    res[EVENT_T:EVENT_T + len(tmpl), y - 1:y + 2, x - 1:x + 2] += 12.0 * tmpl[:, None, None]
    # Slow neighbour inside the pooling disk: ramps up for 6 s from the event onset and stays up.
    yy, xx = np.mgrid[0:H, 0:W]
    slow = ((yy - 24) ** 2 + (xx - 30) ** 2) <= 36
    ramp = np.clip((np.arange(T) - EVENT_T) / 45.0, 0, 1).astype(np.float32) * 4.0
    res[:, slow] += ramp[:, None]
    return res


def _gate(cfg: PipelineConfig):
    res = _residual()
    with tempfile.TemporaryDirectory() as td:
        view = ResidualView.from_dense(res)
        fov = FOVData(
            raw_path=Path("dummy.tif"), output_dir=Path(td), data_bin_path=Path(td) / "data.bin",
            shape=(T, H, W), residual_view=view,
            mean_M=np.zeros((H, W), np.float32), mean_S=np.zeros((H, W), np.float32),
            max_S=np.zeros((H, W), np.float32), std_S=np.ones((H, W), np.float32),
            vcorr_S=np.zeros((H, W), np.float32), dog_map=np.zeros((H, W), np.float32),
            mean_L=np.zeros((H, W), np.float32), k_background=30,
        )
        bank = build_template_bank(FS, TAU)
        rois = run_stage3(view, fov, bank, cfg, starting_label_id=1)
        return evaluate_gate3(rois, [], bank, cfg)


def _cfg(**kw) -> PipelineConfig:
    return PipelineConfig(fs=FS, tau=TAU, reconstruct_chunk=100, template_threshold=6.0, spatial_pool_radius=8,
                          cluster_distance=12, min_event_separation=2.0, stage3_pixel_chunk_rows=16, force_cpu=True,
                          **kw)


def _nearest(rois):
    y, x = EVENT_YX
    d = [np.hypot(r.features["centroid_y"] - y, r.features["centroid_x"] - x) for r in rois]
    assert rois and min(d) <= 6, f"no Stage 3 candidate near the event: {d}"
    return rois[int(np.argmin(d))]


def test_sharp_event_next_to_slow_neighbour_passes_gate3():
    r = _nearest(_gate(_cfg()))
    assert r.gate_outcome != "reject", r.gate_reasons


def test_roi_trace_source_restores_disk_pooled_judgement():
    r = _nearest(_gate(_cfg(gate3_waveform_source="roi_trace")))
    assert r.gate_outcome == "reject"
    assert any(g.startswith(("rise_decay", "waveform_r2")) for g in r.gate_reasons), r.gate_reasons


def test_event_waveforms_are_not_serialized():
    r = _nearest(_gate(_cfg()))
    import json

    json.dumps(r.to_serializable())
