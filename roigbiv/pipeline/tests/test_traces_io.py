"""Tests for :mod:`roigbiv.pipeline.traces_io` — sidecar schema, byte
determinism, row ordering, identifier plumbing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from roigbiv.pipeline.traces_io import (
    SCHEMA_VERSION,
    build_sidecar,
    compute_corrections_rev,
    finalize_fov_bundle,
    write_traces_bundle,
)
from roigbiv.pipeline.types import PipelineConfig, ROI


SHAPE = (8, 8)


def _rect_mask(y0: int, x0: int, y1: int, x1: int) -> np.ndarray:
    m = np.zeros(SHAPE, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _roi(label_id: int, mask: np.ndarray, *,
         stage: int = 1, gate: str = "accept",
         confidence: str = "high") -> ROI:
    return ROI(
        mask=mask,
        label_id=label_id,
        source_stage=stage,
        confidence=confidence,
        gate_outcome=gate,
        area=int(mask.sum()),
    )


def _fake_bundle(n_rois: int, n_frames: int):
    rng = np.random.default_rng(0)
    F_corrected = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
    F_raw = (F_corrected + 1.0).astype(np.float32)
    F_neu = rng.standard_normal((n_rois, n_frames)).astype(np.float32)
    return F_raw, F_neu, F_corrected


def test_sidecar_schema_shape_and_rows():
    rois = [
        _roi(1, _rect_mask(0, 0, 3, 3)),
        _roi(2, _rect_mask(4, 4, 7, 7), stage=2, gate="flag", confidence="moderate"),
    ]
    _, _, F_corr = _fake_bundle(2, 16)
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)

    sidecar = build_sidecar(rois, F_corr, cfg, source="pipeline")

    assert sidecar["schema_version"] == SCHEMA_VERSION
    assert sidecar["fs"] == 7.5
    assert sidecar["frame_averaging"] == 4
    assert sidecar["effective_fps"] == 7.5
    assert sidecar["n_rois"] == 2
    assert sidecar["n_frames"] == 16
    assert sidecar["shape"] == [2, 16]
    assert sidecar["dtype"] == "float32"
    assert sidecar["source"] == "pipeline"
    # No registry plumbing yet → null top-level IDs
    assert sidecar["session_id"] is None
    assert sidecar["fov_id"] is None
    assert sidecar["registry_decision"] is None
    # Row order follows input (label 1 before 2)
    assert [r["row_index"] for r in sidecar["rois"]] == [0, 1]
    assert [r["local_label_id"] for r in sidecar["rois"]] == [1, 2]
    # Omission-not-null: no global_cell_id when unregistered
    for r in sidecar["rois"]:
        assert "global_cell_id" not in r
    # Standard per-row fields
    assert sidecar["rois"][0]["gate_outcome"] == "accept"
    assert sidecar["rois"][1]["gate_outcome"] == "flag"
    assert sidecar["rois"][1]["confidence"] == "moderate"
    # Files block points at the three npy names
    assert sidecar["files"] == {
        "primary": "traces.npy",
        "raw": "traces_raw.npy",
        "neuropil": "traces_neuropil.npy",
    }


def test_sidecar_byte_deterministic_across_builds(tmp_path: Path):
    rois = [
        _roi(1, _rect_mask(0, 0, 3, 3)),
        _roi(2, _rect_mask(4, 4, 7, 7), stage=2),
    ]
    F_raw, F_neu, F_corr = _fake_bundle(2, 16)
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)

    bundle1 = write_traces_bundle(
        rois, F_raw, F_neu, F_corr, tmp_path / "a", cfg,
        source="pipeline",
    )
    bundle2 = write_traces_bundle(
        rois, F_raw, F_neu, F_corr, tmp_path / "b", cfg,
        source="pipeline",
    )
    meta1 = (bundle1 / "traces_meta.json").read_bytes()
    meta2 = (bundle2 / "traces_meta.json").read_bytes()
    assert hashlib.sha256(meta1).digest() == hashlib.sha256(meta2).digest()


def test_bundle_arrays_roundtrip_float32(tmp_path: Path):
    rois = [_roi(1, _rect_mask(0, 0, 3, 3)),
            _roi(2, _rect_mask(4, 4, 7, 7))]
    F_raw, F_neu, F_corr = _fake_bundle(2, 20)
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)

    bundle = write_traces_bundle(
        rois, F_raw, F_neu, F_corr, tmp_path, cfg, source="pipeline",
    )
    arr = np.load(bundle / "traces.npy")
    raw = np.load(bundle / "traces_raw.npy")
    neu = np.load(bundle / "traces_neuropil.npy")
    assert arr.dtype == np.float32
    assert raw.dtype == np.float32
    assert neu.dtype == np.float32
    assert arr.shape == (2, 20)
    np.testing.assert_array_equal(arr, F_corr)
    np.testing.assert_array_equal(raw, F_raw)
    np.testing.assert_array_equal(neu, F_neu)


def test_registry_report_merges_into_sidecar_rows():
    rois = [_roi(1, _rect_mask(0, 0, 3, 3)),
            _roi(2, _rect_mask(4, 4, 7, 7)),
            _roi(3, _rect_mask(0, 4, 3, 7), stage=3)]
    _, _, F_corr = _fake_bundle(3, 5)
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)
    report = {
        "decision": "auto_match",
        "session_id": "sess-uuid",
        "fov_id": "fov-uuid",
        "cell_assignments": [
            {"local_label_id": 1, "global_cell_id": "gid-A",
             "match_kind": "matched"},
            {"local_label_id": 3, "global_cell_id": "gid-C",
             "match_kind": "new"},
            # label 2 intentionally missing → omitted from sidecar row
        ],
    }

    sidecar = build_sidecar(
        rois, F_corr, cfg, source="pipeline", registry_report=report,
    )
    assert sidecar["session_id"] == "sess-uuid"
    assert sidecar["fov_id"] == "fov-uuid"
    assert sidecar["registry_decision"] == "auto_match"
    row_by_label = {r["local_label_id"]: r for r in sidecar["rois"]}
    assert row_by_label[1]["global_cell_id"] == "gid-A"
    assert row_by_label[3]["global_cell_id"] == "gid-C"
    assert "global_cell_id" not in row_by_label[2]


def test_review_decision_keeps_ids_null():
    rois = [_roi(1, _rect_mask(0, 0, 3, 3))]
    _, _, F_corr = _fake_bundle(1, 4)
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)
    report = {
        "decision": "review",
        "session_id": None,
        "fov_id": None,
        "cell_assignments": [],
    }
    sidecar = build_sidecar(
        rois, F_corr, cfg, source="pipeline", registry_report=report,
    )
    assert sidecar["registry_decision"] == "review"
    assert sidecar["session_id"] is None
    assert sidecar["fov_id"] is None
    assert "global_cell_id" not in sidecar["rois"][0]


def test_corrections_rev_stable_across_ordering():
    r1 = _roi(1, _rect_mask(0, 0, 3, 3))
    r2 = _roi(2, _rect_mask(4, 4, 7, 7))
    r3 = _roi(3, _rect_mask(0, 4, 3, 7))
    h1 = compute_corrections_rev([r1, r2, r3])
    h2 = compute_corrections_rev([r3, r1, r2])
    assert h1 == h2
    assert len(h1) == 12
    # Hash changes when mask content changes
    r2_moved = _roi(2, _rect_mask(4, 0, 7, 3))
    h3 = compute_corrections_rev([r1, r2_moved, r3])
    assert h3 != h1


def test_neuropil_presence_flag(tmp_path: Path):
    rois = [_roi(1, _rect_mask(0, 0, 3, 3))]
    F_raw = np.ones((1, 8), dtype=np.float32)
    F_corr = F_raw.copy()
    F_neu_zero = np.zeros((1, 8), dtype=np.float32)
    F_neu_real = np.ones((1, 8), dtype=np.float32)
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)

    bundle_empty = write_traces_bundle(
        rois, F_raw, F_neu_zero, F_corr, tmp_path / "empty", cfg,
        source="pipeline",
    )
    bundle_real = write_traces_bundle(
        rois, F_raw, F_neu_real, F_corr, tmp_path / "real", cfg,
        source="pipeline",
    )
    m_empty = json.loads((bundle_empty / "traces_meta.json").read_text())
    m_real = json.loads((bundle_real / "traces_meta.json").read_text())
    assert m_empty["neuropil"]["present"] is False
    assert m_real["neuropil"]["present"] is True
    # File exists either way (AC4: always write the array)
    assert (bundle_empty / "traces_neuropil.npy").exists()


def test_finalize_fov_bundle_produces_subdir(tmp_path: Path):
    rois = [_roi(1, _rect_mask(0, 0, 3, 3))]
    F_raw, F_neu, F_corr = _fake_bundle(1, 5)
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)
    out = finalize_fov_bundle(
        rois, F_raw, F_neu, F_corr, tmp_path, cfg,
    )
    assert out == tmp_path / "traces"
    assert (out / "traces.npy").exists()
    assert (out / "traces_raw.npy").exists()
    assert (out / "traces_neuropil.npy").exists()
    assert (out / "traces_meta.json").exists()
