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
    load_centroids,
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


# ── load_centroids: centroids.json → ROIRender round-trip ───────────────────


def test_load_centroids_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_centroids(tmp_path, (32, 32)) == []


def test_load_centroids_round_trip(tmp_path: Path) -> None:
    import json

    (tmp_path / "centroids.json").write_text(json.dumps({
        "stem": "fovA", "source": "suite2p",
        "centroids": [
            {"label_id": 0, "y": 10.0, "x": 12.0, "npix": 30, "cellpose_prob": 0.87},
            {"label_id": 1, "y": 20.0, "x": 5.0, "npix": 18, "cellpose_prob": 0.10},
        ],
    }))

    rois = load_centroids(tmp_path, (32, 32), radius=4)
    assert len(rois) == 2

    r0 = rois[0]
    assert r0.label_id == 0
    assert r0.source_stage == 2       # palette slot only, not a detector claim
    assert r0.gate_outcome == "accept"
    assert r0.centroid_yx == (10.0, 12.0)
    assert r0.area == 30
    assert r0.features["cellpose_prob"] == 0.87
    assert r0.contours, "expected a non-empty disk contour around the centroid"

    r1 = rois[1]
    assert r1.features["cellpose_prob"] == 0.10, (
        "low-probability candidates are still returned unfiltered — the UI "
        "toggle controls the whole layer, not per-candidate filtering")


def test_load_centroids_corrupt_json_returns_empty(tmp_path: Path) -> None:
    (tmp_path / "centroids.json").write_text("not valid json {")
    assert load_centroids(tmp_path, (32, 32)) == []


# ── load_cross_session_bundle: timeline ordering ───────────────────────────


def _session_row(session_id, output_dir, session_date, created_at):
    return SimpleNamespace(
        session_id=session_id, output_dir=str(output_dir),
        session_date=session_date, created_at=created_at, fov_posterior=None,
    )


class _OrderedStore:
    """A store returning sessions in timeline order, as the real one does."""

    def __init__(self, rows):
        self._rows = rows

    def ensure_schema(self):
        pass

    def get_fov(self, fov_id):
        return SimpleNamespace(animal_id="DS-Prism-3", region="DS-Prism")

    def list_sessions(self, fov_id):
        return list(self._rows)

    def list_observations_for_session(self, session_id):
        return []


def test_cross_session_bundle_keeps_the_human_confirmed_order(tmp_path,
                                                              monkeypatch):
    """``list_sessions`` already orders by sequence_index; re-sorting by date
    undoes it.

    The dates here are chosen to disagree with the human order in both ways
    that actually occur: one stem's date is unreadable (``date_source`` is
    ``unparsed``), and the remaining two run backwards relative to the order
    the researcher set.
    """
    from datetime import date as _date

    from roigbiv.ui.services import loaders as L

    stems = ["pre-005", "beh-006", "post-007"]
    dates = [None, _date(2026, 5, 21), _date(2026, 5, 20)]
    rows = []
    for i, (stem, session_date) in enumerate(zip(stems, dates)):
        out = tmp_path / stem
        out.mkdir()
        rows.append(_session_row(f"s{i}", out, session_date, None))

    store = _OrderedStore(rows)
    monkeypatch.setattr("roigbiv.registry.build_store", lambda cfg=None: store)
    monkeypatch.setattr(L, "load_fov_bundle", lambda out_dir: SimpleNamespace(
        output_dir=out_dir, rois=[]))

    bundle = L.load_cross_session_bundle("fov-1")

    assert [s.session_id for s in bundle.sessions] == ["s0", "s1", "s2"]
    assert [s.output_dir.name for s in bundle.sessions] == stems


def test_cross_session_bundle_dedupes_without_reordering(tmp_path, monkeypatch):
    """A re-registered output_dir keeps its original timeline position."""
    from datetime import date as _date, datetime as _dt

    from roigbiv.ui.services import loaders as L

    stems = ["pre-005", "beh-006"]
    for stem in stems:
        (tmp_path / stem).mkdir()
    rows = [
        # pre-005 is first in the human order despite carrying the later date.
        _session_row("s0", tmp_path / "pre-005", _date(2026, 5, 21),
                     _dt(2026, 5, 22, 9, 0)),
        _session_row("s1", tmp_path / "beh-006", _date(2026, 5, 20),
                     _dt(2026, 5, 22, 9, 1)),
        # pre-005 registered a second time, later — newest row wins, but the
        # position must not move to the end of the timeline.
        _session_row("s2", tmp_path / "pre-005", _date(2026, 5, 21),
                     _dt(2026, 5, 22, 10, 0)),
    ]

    store = _OrderedStore(rows)
    monkeypatch.setattr("roigbiv.registry.build_store", lambda cfg=None: store)
    monkeypatch.setattr(L, "load_fov_bundle", lambda out_dir: SimpleNamespace(
        output_dir=out_dir, rois=[]))

    bundle = L.load_cross_session_bundle("fov-1")

    assert [s.session_id for s in bundle.sessions] == ["s2", "s1"]
