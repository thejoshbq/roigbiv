"""Tests for :mod:`roigbiv.ui.services.loaders` helpers that don't hit disk."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from roigbiv.pipeline.types import ROI
from roigbiv.ui.services.loaders import (
    _gcid_by_label_from_registry,
    list_motion_corrected_fovs,
    mc_input_mean,
    render_roi,
)


def _mask(shape, y0, x0, y1, x1) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _write_stack(path: Path, frames) -> Path:
    """Write a flat page-per-frame uint16 stack from a list of 2D arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(str(path)) as tw:
        for i, frame in enumerate(frames):
            tw.write(np.asarray(frame, dtype=np.uint16),
                     contiguous=(i > 0))
    return path


def _fake_ws(output_root: Path, tifs=()):
    return SimpleNamespace(output_root=output_root, tifs=tuple(tifs))


# ── list_motion_corrected_fovs: summary + pre-corrected-input merge ─────────


def test_list_mc_fovs_includes_precorrected_inputs(tmp_path):
    # A workspace with a pre-corrected *_mc.tif and no output/ should surface
    # the input FOV with an input:-prefixed value and an "(input)" label.
    mc = _write_stack(tmp_path / "session01_mc.tif", [np.zeros((4, 4))] * 3)
    ws = _fake_ws(tmp_path / "output", tifs=[mc])
    fovs = list_motion_corrected_fovs(ws)
    assert fovs == [("session01 (input)", f"input:{mc}")]


def test_list_mc_fovs_summary_wins_over_input(tmp_path):
    # Same output stem present BOTH as a finished summary and as a *_mc.tif
    # input must collapse to a single summary: entry (de-dup by output stem).
    out_root = tmp_path / "output"
    mean_path = out_root / "session01" / "summary" / "mean_M.tif"
    _write_stack(mean_path, [np.zeros((4, 4))])
    mc = _write_stack(tmp_path / "session01_mc.tif", [np.zeros((4, 4))] * 3)
    ws = _fake_ws(out_root, tifs=[mc])
    fovs = list_motion_corrected_fovs(ws)
    out_dir = mean_path.parent.parent
    assert fovs == [("session01", f"summary:{out_dir}")]


def test_list_mc_fovs_skips_uncorrected_inputs(tmp_path):
    # A plain raw stack (no _mc suffix, no content tag) is not previewable yet.
    raw = _write_stack(tmp_path / "session01.tif", [np.zeros((4, 4))] * 3)
    ws = _fake_ws(tmp_path / "output", tifs=[raw])
    assert list_motion_corrected_fovs(ws) == []


def test_list_mc_fovs_handles_missing_workspace():
    assert list_motion_corrected_fovs(None) == []


# ── mc_input_mean: sampled temporal mean + cache ────────────────────────────


def test_mc_input_mean_samples_and_means(tmp_path):
    # Frame i is a constant field of value i; the temporal mean over all frames
    # (n <= sample cap) is a constant field of mean(0..n-1).
    n, h, w = 5, 8, 8
    frames = [np.full((h, w), i, dtype=np.uint16) for i in range(n)]
    p = _write_stack(tmp_path / "foo_mc.tif", frames)
    mean = mc_input_mean(p)
    assert mean is not None
    assert mean.shape == (h, w)
    assert mean.dtype == np.float32
    np.testing.assert_allclose(mean, np.full((h, w), np.mean(range(n))))


def test_mc_input_mean_is_cached(tmp_path):
    p = _write_stack(tmp_path / "bar_mc.tif", [np.zeros((4, 4))] * 3)
    first = mc_input_mean(p)
    # Same file → same cache key → identical object returned (no recompute).
    assert mc_input_mean(p) is first


def test_mc_input_mean_missing_file_returns_none(tmp_path):
    assert mc_input_mean(tmp_path / "does_not_exist_mc.tif") is None


def test_render_roi_builds_contour_and_centroid() -> None:
    mask = _mask((32, 32), 10, 12, 20, 22)
    roi = ROI(
        mask=mask, label_id=7, source_stage=2,
        confidence="moderate", gate_outcome="flag",
        area=int(mask.sum()), activity_type="sparse",
    )
    rendered = render_roi(roi, gcid="abc-123")
    assert rendered.label_id == 7
    assert rendered.source_stage == 2
    assert rendered.gate_outcome == "flag"
    assert rendered.activity_type == "sparse"
    assert rendered.global_cell_id == "abc-123"
    assert len(rendered.contours) >= 1
    ys, xs = rendered.contours[0]
    assert len(ys) == len(xs) and len(ys) > 3
    # centroid should land inside the rectangle
    cy, cx = rendered.centroid_yx
    assert 10 <= cy <= 20
    assert 12 <= cx <= 22


def test_gcid_by_label_map_handles_missing_registry() -> None:
    assert _gcid_by_label_from_registry(None) == {}
    assert _gcid_by_label_from_registry({}) == {}


def test_gcid_by_label_map_extracts_entries() -> None:
    registry = {
        "cell_assignments": [
            {"local_label_id": 1, "global_cell_id": "g1",
             "match_kind": "matched"},
            {"local_label_id": 2, "global_cell_id": None,
             "match_kind": "new"},
            {"local_label_id": "bad", "global_cell_id": "g3"},
            {"local_label_id": 3, "global_cell_id": "g3"},
        ],
    }
    out = _gcid_by_label_from_registry(registry)
    assert out == {1: "g1", 3: "g3"}
