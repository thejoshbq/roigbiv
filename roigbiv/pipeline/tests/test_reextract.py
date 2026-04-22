"""End-to-end test for :mod:`roigbiv.pipeline.reextract`.

Builds a tiny synthetic FOV: fake ``data.bin`` + primary ``traces/``
bundle + ``corrections/`` artifacts, then verifies:

  * re-extract produces a sibling ``traces/corrections-{rev}/`` dir
  * the primary ``traces.npy`` is never mutated
  * a second invocation is a no-op (idempotent)
  * fresh HITL labels get no ``global_cell_id``, inherited labels keep theirs
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline.corrections import (
    CorrectionOp,
    apply_corrections,
    corrections_dir,
    materialize,
    write_corrections,
)
from roigbiv.pipeline.reextract import reextract_from_corrections
from roigbiv.pipeline.traces_io import write_traces_bundle
from roigbiv.pipeline.types import PipelineConfig, ROI


H, W, T = 16, 16, 120


def _rect(y0, x0, y1, x1) -> np.ndarray:
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


def _make_data_bin(path: Path) -> None:
    """Write a deterministic (T, H, W) int16 memmap."""
    rng = np.random.default_rng(0)
    mm = np.memmap(str(path), dtype=np.int16, mode="w+", shape=(T, H, W))
    mm[:] = rng.integers(-32000, 32000, size=(T, H, W), dtype=np.int16)
    mm.flush()
    del mm


def _write_primary_bundle(
    fov_out: Path, data_bin: Path, rois: list[ROI], cfg: PipelineConfig,
) -> None:
    """Write a hand-rolled primary traces/ bundle with fake identifiers."""
    n = len(rois)
    F_raw = np.ones((n, T), dtype=np.float32)
    F_neu = np.zeros((n, T), dtype=np.float32)
    F_corr = F_raw.copy()
    report = {
        "decision": "new_fov",
        "session_id": "sess-abc",
        "fov_id": "fov-xyz",
        "cell_assignments": [
            {"local_label_id": int(r.label_id),
             "global_cell_id": f"gid-{int(r.label_id)}",
             "match_kind": "new"}
            for r in rois
        ],
    }
    write_traces_bundle(
        rois, F_raw, F_neu, F_corr, fov_out, cfg,
        source="pipeline",
        registry_report=report,
        data_bin_path=data_bin,
        fov_shape=(T, H, W),
    )


def test_reextract_writes_sibling_and_preserves_primary(tmp_path: Path):
    fov_out = tmp_path / "fov1"
    fov_out.mkdir()
    data_bin = fov_out / "data.bin"
    _make_data_bin(data_bin)

    r1 = _roi(1, _rect(0, 0, 4, 4))
    r2 = _roi(2, _rect(6, 6, 10, 10))
    rois = [r1, r2]
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)

    _write_primary_bundle(fov_out, data_bin, rois, cfg)

    # Capture primary state for mutation check.
    primary_traces = fov_out / "traces" / "traces.npy"
    primary_bytes = primary_traces.read_bytes()
    primary_mtime = primary_traces.stat().st_mtime_ns

    # Apply a correction: delete label 2, add a fresh ROI.
    ops = [
        CorrectionOp.delete(2),
        CorrectionOp.add(polygon=[[12.0, 12.0], [12.0, 15.0],
                                   [15.0, 15.0], [15.0, 12.0]]),
    ]
    write_corrections(fov_out, ops)
    corrected = apply_corrections(rois, ops, (H, W))
    materialize(corrected, fov_out, (H, W))

    target = reextract_from_corrections(
        fov_out, cfg=cfg, skip_overlap_correction=True,
    )
    assert target.exists()
    assert target.parent == fov_out / "traces"
    assert target.name.startswith("corrections-")

    # All four files present.
    for fname in ("traces.npy", "traces_raw.npy",
                  "traces_neuropil.npy", "traces_meta.json"):
        assert (target / fname).exists(), f"missing {fname}"

    # Primary was NOT touched.
    assert primary_traces.read_bytes() == primary_bytes
    assert primary_traces.stat().st_mtime_ns == primary_mtime

    # Shape: corrected set has 2 ROIs (1 kept, 1 new from add; 2 deleted).
    arr = np.load(target / "traces.npy")
    assert arr.shape[1] == T
    assert arr.shape[0] == len(corrected)

    # Sidecar: inherited identifiers for surviving label 1; fresh label has none.
    meta = json.loads((target / "traces_meta.json").read_text())
    assert meta["source"] == "corrections"
    assert meta["session_id"] == "sess-abc"
    assert meta["fov_id"] == "fov-xyz"
    assert meta["registry_decision"] == "new_fov"
    assert meta["corrections_rev"]
    by_label = {r["local_label_id"]: r for r in meta["rois"]}
    assert 1 in by_label
    assert by_label[1]["global_cell_id"] == "gid-1"
    # The freshly added ROI has a new label_id and no global_cell_id.
    fresh = [r for r in meta["rois"] if r["local_label_id"] != 1]
    assert fresh
    for r in fresh:
        assert "global_cell_id" not in r


def test_reextract_idempotent(tmp_path: Path):
    fov_out = tmp_path / "fov2"
    fov_out.mkdir()
    data_bin = fov_out / "data.bin"
    _make_data_bin(data_bin)

    rois = [_roi(1, _rect(0, 0, 4, 4))]
    cfg = PipelineConfig(fs=7.5, frame_averaging=4)
    _write_primary_bundle(fov_out, data_bin, rois, cfg)

    ops = [CorrectionOp.delete(1)]
    write_corrections(fov_out, ops)
    corrected = apply_corrections(rois, ops, (H, W))
    # Add a user ROI so there is something to extract from.
    ops2 = ops + [CorrectionOp.add(
        polygon=[[2.0, 2.0], [2.0, 5.0], [5.0, 5.0], [5.0, 2.0]])]
    write_corrections(fov_out, ops2)
    corrected = apply_corrections(rois, ops2, (H, W))
    materialize(corrected, fov_out, (H, W))

    first = reextract_from_corrections(
        fov_out, cfg=cfg, skip_overlap_correction=True,
    )
    first_mtime = (first / "traces.npy").stat().st_mtime_ns

    second = reextract_from_corrections(
        fov_out, cfg=cfg, skip_overlap_correction=True,
    )
    assert second == first
    # Idempotent: the short-circuit should leave mtime unchanged.
    assert (second / "traces.npy").stat().st_mtime_ns == first_mtime
