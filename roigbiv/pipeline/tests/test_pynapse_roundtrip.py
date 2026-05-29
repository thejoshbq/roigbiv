"""Round-trip test for the ROIGBIV → pynapse handoff (AC7).

The strict part (``test_pynapse_sample_roundtrip``) is skipped when pynapse
is not installed in the environment. The format-contract test always runs
— it's the part that would actually break the handoff.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from roigbiv.pipeline.traces_io import write_traces_bundle
from roigbiv.pipeline.types import PipelineConfig, ROI


def _rect(H: int, W: int, y0, x0, y1, x1) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _roi(label_id: int, mask: np.ndarray) -> ROI:
    return ROI(
        mask=mask,
        label_id=label_id,
        source_stage=1,
        confidence="high",
        gate_outcome="accept",
        area=int(mask.sum()),
    )


def _write_bundle(tmp_path: Path, n_rois: int = 3, n_frames: int = 200):
    rng = np.random.default_rng(42)
    F_corr = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
    rois = [_roi(i + 1, _rect(16, 16, i * 4, 0, i * 4 + 3, 3))
            for i in range(n_rois)]
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)
    bundle = write_traces_bundle(
        rois,
        F_raw=F_corr + 1.0,
        F_neu=np.ones_like(F_corr),
        F_corrected=F_corr,
        output_dir=tmp_path,
        cfg=cfg,
        source="pipeline",
    )
    return bundle, cfg, n_rois, n_frames, F_corr


def test_traces_npy_format_contract(tmp_path: Path):
    """Checks the exact format pynapse requires without needing pynapse.

    Shape must be ``(n_rois, n_frames)`` 2-D, dtype float32, and ``np.load``
    with ``squeeze()`` (what pynapse applies) must preserve shape.
    """
    bundle, cfg, n, T, F_corr = _write_bundle(tmp_path)
    arr = np.load(bundle / "traces.npy")
    assert arr.shape == (n, T)
    assert arr.dtype == np.float32
    # Pynapse .squeeze()'s on load — must still be 2-D.
    assert arr.squeeze().shape == (n, T)
    # Sidecar advertises the same shape.
    meta = json.loads((bundle / "traces_meta.json").read_text())
    assert meta["shape"] == [n, T]
    assert meta["n_rois"] == n
    assert meta["n_frames"] == T
    # fs/frame_averaging/effective_fps consistency (AC2).
    assert meta["fs"] == cfg.fs
    assert meta["frame_averaging"] == cfg.frame_averaging
    assert meta["effective_fps"] == cfg.fs


def test_pynapse_sample_roundtrip(tmp_path: Path):
    """Actually hand the bundle to pynapse. Skipped if pynapse isn't installed."""
    SignalRecording = pytest.importorskip(
        "pynapse.core.io.microscopy",
        reason="pynapse not installed in this env",
    ).SignalRecording

    bundle, cfg, n_rois, n_frames, _ = _write_bundle(tmp_path)
    meta = json.loads((bundle / "traces_meta.json").read_text())

    sig = SignalRecording(source=str(bundle / "traces.npy"))
    assert sig.num_neurons == n_rois
    assert sig.num_frames == n_frames

    # Sample instantiation requires a companion event log which lives outside
    # roigbiv's test surface. Verify only the signal-side frame-rate math.
    raw_fps = meta["fs"] * meta["frame_averaging"]
    effective = raw_fps / meta["frame_averaging"]
    assert effective == pytest.approx(meta["effective_fps"])
    assert effective == pytest.approx(meta["fs"])
