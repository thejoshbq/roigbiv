"""Seeded boundaries as a FOV artifact — edit replay, guards, and degradation.

Covers what ``boundaries.py`` owns on top of ``seeded_masks.py``'s partition
logic: reading the cached flow field, seeding off the *replayed* centroid edit
log rather than raw detector output, refusing to clobber a full cascade, and
degrading to nothing (rather than failing) when there is no flow field.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.boundaries import (
    compute_boundaries,
    load_boundary_labels,
    write_boundaries,
)
from roigbiv.pipeline.boundary_edits import BoundaryOp, append_boundary_op
from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op
from roigbiv.pipeline.seeded_masks import (
    ORIGIN_DISK_FALLBACK,
    ORIGIN_FLOW,
    ORIGIN_MANUAL,
)

H = W = 96
_PARAMS = {"detector": "cellpose", "diameter_px": 20.0,
           "cellprob_threshold": -2.0, "cellpose_model": "cyto3",
           "tissue_mask": False}


class _Cfg:
    roi_stamp_radius = 6
    cellprob_threshold = -2.0
    boundary_capture_px = 18.0
    boundary_min_area = 0
    boundary_max_area = None


def _gauss(cy, cx, sigma):
    yy, xx = np.ogrid[:H, :W]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma ** 2))


def _fixture(tmp: Path, centroids, *, blobs=None, flows=True) -> Path:
    """A FOV with centroids.json, a summary image, and a cached flow field.

    The flow field is built so every cell pixel converges *onto its own blob
    centre* — a field with one attractor per blob, which is what Cellpose
    produces when it does not merge two cells.
    """
    out = tmp
    (out / "summary").mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out / "summary" / "mean_M.tif",
                     np.zeros((H, W), np.float32))
    out.joinpath("centroids.json").write_text(json.dumps({
        "stem": "fovA", "schema": 5, "source": "cellpose", "params": _PARAMS,
        "centroids": [{"label_id": i, "y": y, "x": x, "npix": 100,
                       "equiv_diameter_px": 11.3, "cellpose_prob": 0.9}
                      for i, (y, x) in enumerate(centroids)],
    }))

    if not flows:
        return out

    blobs = centroids if blobs is None else blobs
    cellprob = np.zeros((H, W), np.float32)
    for cy, cx in blobs:
        cellprob = np.maximum(cellprob, _gauss(cy, cx, 7.0).astype(np.float32))
    cellprob = cellprob * 4.0 - 2.0     # straddle the -2.0 threshold

    # A flow field pointing at the nearest blob centre, scaled so that after
    # cellpose's /5 and 200 iterations every pixel lands on it.
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    dy = np.zeros((H, W), np.float32)
    dx = np.zeros((H, W), np.float32)
    nearest = np.full((H, W), -1, np.int32)
    best = np.full((H, W), np.inf, np.float32)
    for i, (cy, cx) in enumerate(blobs):
        d = np.hypot(yy - cy, xx - cx)
        take = d < best
        best, nearest = np.where(take, d, best), np.where(take, i, nearest)
    for i, (cy, cx) in enumerate(blobs):
        sel = nearest == i
        dy[sel], dx[sel] = (cy - yy)[sel], (cx - xx)[sel]

    flow_dir = out / "flows"
    flow_dir.mkdir(exist_ok=True)
    np.save(flow_dir / "dP.npy", np.stack([dy, dx]).astype(np.float32) * 5.0)
    np.save(flow_dir / "cellprob.npy", cellprob)
    (flow_dir / "meta.json").write_text(json.dumps({
        "schema": 5, "params": _PARAMS, "niter": 200,
        "dp_scale": 5.0, "diameter": 20.0, "shape": [H, W],
    }))
    return out


def test_boundaries_are_drawn_from_the_flow_field():
    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0), (30.0, 66.0)])

        result = write_boundaries(out, _Cfg())

        assert result is not None and result.written
        assert result.present_labels == (1, 2)
        assert result.origins == {1: ORIGIN_FLOW, 2: ORIGIN_FLOW}
        labels = load_boundary_labels(out)
        assert labels.shape == (H, W)
        # A real boundary, not the 6-px fallback disk.
        assert result.areas[1] > np.pi * _Cfg.roi_stamp_radius ** 2

        payload = json.loads((out / "boundaries.json").read_text())
        assert payload["n_seeds"] == 2
        assert {e["label"] for e in payload["labels"]} == {1, 2}


def test_centroid_edits_redraw_without_re_detection():
    """The edit log is the seed source — that is the whole point of the cache.

    A moved centroid must move its boundary with no Cellpose call, because the
    /cells page redraws on every edit.
    """
    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0), (30.0, 66.0)])
        before = write_boundaries(out, _Cfg())
        assert before.present_labels == (1, 2)

        append_centroid_op(out, CentroidOp.delete(label=2))
        after = write_boundaries(out, _Cfg())

        assert after.present_labels == (1,), "a deleted centroid keeps no boundary"
        # centroids.json is detector output and is never rewritten.
        assert len(json.loads(
            (out / "centroids.json").read_text())["centroids"]) == 2


def test_added_centroid_gets_a_boundary():
    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)], blobs=[(30.0, 30.0), (66.0, 66.0)])

        append_centroid_op(out, CentroidOp.add(label=2, y=66.0, x=66.0))
        result = write_boundaries(out, _Cfg())

        assert result.present_labels == (1, 2)
        assert result.origins[2] == ORIGIN_FLOW


def test_moved_centroid_onto_empty_background_falls_back_to_a_disk():
    """A confirmed cell never disappears, even when the model saw nothing."""
    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0), (30.0, 66.0)])

        append_centroid_op(out, CentroidOp.move(label=2, y=85.0, x=12.0))
        result = write_boundaries(out, _Cfg())

        assert result.origins[2] == ORIGIN_DISK_FALLBACK
        assert result.areas[2] > 0
        assert 2 in result.present_labels


def test_labels_match_merged_masks_so_the_registry_still_resolves():
    """Both geometry tracks address a cell by the same label id.

    ``CellObservation.local_label_id`` comes from ``merged_masks.tif``; the
    /cells page looks that label up in whichever image it renders.
    """
    from roigbiv.pipeline.centroid_masks import write_merged_masks

    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0), (30.0, 66.0)])
        append_centroid_op(out, CentroidOp.delete(label=1))

        stamped = write_merged_masks(out, _Cfg(), shape=(H, W))
        drawn = write_boundaries(out, _Cfg())

        assert stamped.present_labels == drawn.present_labels == (2,)


def test_missing_flow_cache_returns_none_rather_than_raising():
    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)], flows=False)

        assert compute_boundaries(out, _Cfg()) is None
        assert write_boundaries(out, _Cfg()) is None
        assert not (out / "boundaries.tif").exists()


def test_stale_flow_cache_reads_as_absent():
    """A field from other detection params must not seed confident boundaries."""
    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)])
        meta_path = out / "flows" / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["params"] = {**_PARAMS, "diameter_px": 99.0}
        meta_path.write_text(json.dumps(meta))

        assert write_boundaries(out, _Cfg()) is None


def test_no_centroids_json_returns_none():
    with tempfile.TemporaryDirectory() as td:
        assert write_boundaries(Path(td), _Cfg()) is None


def test_full_cascade_output_is_never_clobbered():
    """Four stages of real per-ROI geometry outrank centroid-seeded boundaries."""
    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)])
        write_boundaries(out, _Cfg())
        sentinel = np.full((H, W), 7, np.uint16)
        tifffile.imwrite(out / "boundaries.tif", sentinel)
        (out / "pipeline_log.json").write_text("{}")

        result = write_boundaries(out, _Cfg())

        assert result is not None and result.written is False
        assert np.array_equal(load_boundary_labels(out), sentinel)


def test_capture_px_defaults_to_the_calibrated_soma_radius():
    """Uncapped, unlike the stamp radius: ROICaT's 36x36 crop is irrelevant here."""
    from roigbiv.pipeline.boundaries import resolve_capture_px
    from roigbiv.pipeline.calibration import write_calibration

    class _NoExplicit(_Cfg):
        boundary_capture_px = None

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        write_calibration(out, 90.0)
        assert resolve_capture_px(out, _NoExplicit()) == pytest.approx(45.0)


def test_load_boundary_labels_absent_is_none():
    with tempfile.TemporaryDirectory() as td:
        assert load_boundary_labels(Path(td)) is None


# ── per-FOV settings, pinned only by an explicit save ──────────────────────
#
# The boundary page's sliders reach the pipeline through these overrides, and
# what a Save pins has to survive every later automatic redraw — an edit
# gesture, a fresh run_tracking — without any of *those* pinning anything of
# their own.


class _Resolved(_Cfg):
    boundary_capture_px = None
    boundary_min_area = None


def test_an_explicit_capture_px_is_reused_by_the_next_redraw():
    from roigbiv.pipeline.boundaries import resolve_capture_px
    from roigbiv.pipeline.calibration import write_calibration

    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)])
        write_calibration(out, 90.0)          # would otherwise resolve to 45.0

        write_boundaries(out, _Resolved(), capture_px=11.0, min_area=4)

        assert resolve_capture_px(out, _Resolved()) == pytest.approx(11.0)
        settings = json.loads((out / "boundaries.json").read_text())["settings"]
        assert settings == {"capture_px": 11.0, "min_area": 4}


def test_an_automatic_redraw_pins_nothing():
    """Otherwise the first redraw freezes the calibration it happened to see.

    A later re-calibration would then stop reaching the boundaries, silently.
    """
    from roigbiv.pipeline.boundaries import resolve_capture_px
    from roigbiv.pipeline.calibration import write_calibration

    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)])
        write_calibration(out, 90.0)
        write_boundaries(out, _Resolved())     # drawn at the resolved 45.0

        write_calibration(out, 30.0)           # the human re-measures
        assert resolve_capture_px(out, _Resolved()) == pytest.approx(15.0)


def test_a_pin_survives_a_later_automatic_redraw():
    from roigbiv.pipeline.boundaries import resolve_capture_px

    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)])
        write_boundaries(out, _Resolved(), capture_px=11.0)

        write_boundaries(out, _Resolved())     # an edit-triggered redraw

        assert resolve_capture_px(out, _Resolved()) == pytest.approx(11.0)
        assert json.loads(
            (out / "boundaries.json").read_text())["settings"]["capture_px"] == 11.0


def test_cfg_still_outranks_a_pinned_setting():
    """A CLI flag or config file is a deliberate instruction for this run."""
    from roigbiv.pipeline.boundaries import resolve_capture_px

    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)])
        write_boundaries(out, _Resolved(), capture_px=11.0)
        assert resolve_capture_px(out, _Cfg()) == pytest.approx(18.0)


def test_an_unreadable_boundaries_json_falls_back_rather_than_raising():
    from roigbiv.pipeline.boundaries import resolve_capture_px, resolve_min_area
    from roigbiv.pipeline.calibration import write_calibration

    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        write_calibration(out, 90.0)
        (out / "boundaries.json").write_text("{ not json")

        assert resolve_capture_px(out, _Resolved()) == pytest.approx(45.0)
        assert resolve_min_area(out, _Resolved()) == 0


# ── manual boundary overrides (roigbiv/pipeline/boundary_edits.py) ──────────
#
# compute_boundaries/write_boundaries layer corrections/boundaries.jsonl over
# seeded_labels' output as their last step — see boundaries.py's module
# docstring. These tests exercise that wiring end to end, through the real
# flow-field path.


def _manual_ring() -> list:
    """A small ring; real assertions compare against a from-scratch auto
    computation rather than assuming which pixels it does or doesn't claim —
    the fixture's synthetic flow field partitions almost the whole frame by
    nearest centroid, so "far from everything" is not a real place here."""
    return [[2.0, 2.0], [2.0, 12.0], [12.0, 12.0], [12.0, 2.0]]


def test_a_drawn_boundary_overrides_the_auto_shape_for_that_label_only():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        centroids = [(30.0, 30.0), (30.0, 66.0)]
        baseline = write_boundaries(_fixture(td / "auto", centroids), _Cfg())

        out = _fixture(td / "manual", centroids)
        append_boundary_op(out, BoundaryOp.draw(1, _manual_ring()))
        result = write_boundaries(out, _Cfg())

        assert result.origins[1] == ORIGIN_MANUAL
        # Label 2 is untouched by an op that only names label 1.
        assert result.origins[2] == baseline.origins[2]
        assert result.areas[2] == baseline.areas[2]
        # Label 1's manual footprint is the small hand-drawn ring, not the
        # auto computation's (much larger, in this fixture) basin.
        assert result.areas[1] != baseline.areas[1]
        assert result.areas[1] < 200   # the ring is a 10x10 square
        # boundaries.json reports the manual origin too.
        payload = json.loads((out / "boundaries.json").read_text())
        entry = next(e for e in payload["labels"] if e["label"] == 1)
        assert entry["origin"] == ORIGIN_MANUAL


def test_deleting_a_manual_boundary_reverts_to_the_current_auto_shape():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        centroids = [(30.0, 30.0), (30.0, 66.0)]
        baseline = write_boundaries(_fixture(td / "auto", centroids), _Cfg())

        out = _fixture(td / "manual", centroids)
        append_boundary_op(out, BoundaryOp.draw(1, _manual_ring()))
        drawn = write_boundaries(out, _Cfg())
        assert drawn.origins[1] == ORIGIN_MANUAL
        assert drawn.areas[1] != baseline.areas[1]

        append_boundary_op(out, BoundaryOp.delete(1))
        reverted = write_boundaries(out, _Cfg())

        assert reverted.origins[1] == baseline.origins[1]
        assert reverted.areas[1] == baseline.areas[1]
        assert np.array_equal(load_boundary_labels(out) == 1,
                              load_boundary_labels(td / "auto") == 1)


def test_undo_by_truncating_the_boundary_log():
    from roigbiv.pipeline.boundary_edits import load_boundary_ops, write_boundary_ops

    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)])
        auto = write_boundaries(out, _Cfg())
        auto_area = auto.areas[1]

        append_boundary_op(out, BoundaryOp.draw(1, _manual_ring()))
        drawn = write_boundaries(out, _Cfg())
        assert drawn.origins[1] == ORIGIN_MANUAL
        assert drawn.areas[1] != auto_area

        ops = load_boundary_ops(out)
        write_boundary_ops(out, ops[:-1])   # undo the draw
        reverted = write_boundaries(out, _Cfg())

        assert reverted.origins[1] == ORIGIN_FLOW
        assert reverted.areas[1] == auto_area


def test_a_manual_boundary_survives_a_min_area_change():
    """The precedence rule end to end: retuning ``min_area`` must not touch
    label 1's manual shape, while label 2 (no override) tracks the retune
    normally — here, dropping below the bound and falling back to a disk."""
    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0), (30.0, 66.0)])
        append_boundary_op(out, BoundaryOp.draw(1, _manual_ring()))

        loose = write_boundaries(out, _Cfg(), min_area=0)
        assert loose.origins[2] == ORIGIN_FLOW

        strict = write_boundaries(out, _Cfg(), min_area=loose.areas[2] + 500)

        assert loose.origins[1] == strict.origins[1] == ORIGIN_MANUAL
        assert loose.areas[1] == strict.areas[1], "the manual footprint must not move"
        # Label 2 (no override) is free to change with the stricter bound.
        assert strict.origins[2] == ORIGIN_DISK_FALLBACK
        assert strict.areas[2] != loose.areas[2]


def test_a_gesture_that_targets_a_nonexistent_label_still_replays_with_a_warning():
    """boundary_edits.apply_boundary_ops is permissive; UI-level validation
    (discovery_edit_ops.py) is what actually refuses this before it is ever
    written — this just documents that a stray op on disk degrades safely."""
    with tempfile.TemporaryDirectory() as td:
        out = _fixture(Path(td), [(30.0, 30.0)])
        append_boundary_op(out, BoundaryOp.draw(99, _manual_ring()))

        result = write_boundaries(out, _Cfg())

        assert result is not None
        assert 99 in result.origins
        assert result.origins[99] == ORIGIN_MANUAL
