"""Hand-drawn boundary edits — op-log round-trip and replay semantics.

Pure ``apply_boundary_ops`` tests operate on synthetic label images (no flow
field, no Cellpose) — the precedence rule this module exists to enforce is a
numpy question, not a detection question. Integration with
``compute_boundaries``/``write_boundaries`` (the flow-field path) lives in
``test_boundaries.py``.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from roigbiv.pipeline.boundary_edits import (
    BoundaryOp,
    active_manual_labels,
    append_boundary_op,
    apply_boundary_ops,
    load_boundary_ops,
    write_boundary_ops,
)
from roigbiv.pipeline.seeded_masks import ORIGIN_DISK_FALLBACK, ORIGIN_FLOW, ORIGIN_MANUAL

H = W = 40


def _auto_labels() -> np.ndarray:
    """Two disjoint squares: label 1 at rows/cols 5-14, label 2 at 25-34."""
    labels = np.zeros((H, W), dtype=np.uint16)
    labels[5:15, 5:15] = 1
    labels[25:35, 25:35] = 2
    return labels


def _origins() -> dict:
    return {1: ORIGIN_FLOW, 2: ORIGIN_DISK_FALLBACK}


def _square_ring(y0, x0, size) -> list:
    return [[y0, x0], [y0, x0 + size], [y0 + size, x0 + size], [y0 + size, x0]]


# ── op-log round-trip ───────────────────────────────────────────────────────


def test_append_load_round_trips():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        op1 = BoundaryOp.draw(1, _square_ring(5, 5, 10))
        op2 = BoundaryOp.delete(2)
        append_boundary_op(out, op1)
        append_boundary_op(out, op2)

        loaded = load_boundary_ops(out)
        assert [o.op for o in loaded] == ["draw", "delete"]
        assert loaded[0].label == 1
        assert loaded[0].ring == op1.ring
        assert loaded[1].label == 2

        assert (out / "corrections" / "boundaries.jsonl").exists()


def test_load_with_no_log_is_empty_list():
    with tempfile.TemporaryDirectory() as td:
        assert load_boundary_ops(Path(td)) == []


def test_write_boundary_ops_replaces_the_log_for_undo():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        append_boundary_op(out, BoundaryOp.draw(1, _square_ring(5, 5, 10)))
        append_boundary_op(out, BoundaryOp.delete(1))

        ops = load_boundary_ops(out)
        write_boundary_ops(out, ops[:-1])   # undo the delete

        after = load_boundary_ops(out)
        assert len(after) == 1
        assert after[0].op == "draw"


def test_write_boundary_ops_with_empty_list_deletes_the_file():
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        append_boundary_op(out, BoundaryOp.draw(1, _square_ring(5, 5, 10)))
        write_boundary_ops(out, [])
        assert not (out / "corrections" / "boundaries.jsonl").exists()
        assert load_boundary_ops(out) == []


# ── replay: draw overrides, delete reverts ──────────────────────────────────


def test_no_ops_returns_the_auto_labels_unchanged():
    labels = _auto_labels()
    origins = _origins()
    out_labels, out_origins, warnings = apply_boundary_ops(labels, origins, [])
    assert out_labels is labels
    assert out_origins is origins
    assert warnings == []


def test_draw_replaces_only_the_targeted_labels_auto_shape():
    labels = _auto_labels()
    origins = _origins()
    # A hand-drawn square for label 1, well clear of both its own old
    # footprint (5:15, 5:15) and label 2's (25:35, 25:35).
    ring = _square_ring(0, 20, 5)
    ops = [BoundaryOp.draw(1, ring)]

    out_labels, out_origins, warnings = apply_boundary_ops(labels, origins, ops)

    assert not warnings
    assert out_origins[1] == ORIGIN_MANUAL
    assert out_origins[2] == ORIGIN_DISK_FALLBACK   # untouched
    # The old auto footprint for label 1 is gone.
    assert not np.any(out_labels[5:15, 5:15] == 1)
    # The new hand-drawn footprint is in.
    assert np.any(out_labels[0:5, 20:25] == 1)
    # Label 2's auto pixels are exactly as before.
    assert np.array_equal(out_labels[25:35, 25:35] == 2, labels[25:35, 25:35] == 2)


def test_a_second_draw_for_the_same_label_replaces_the_first():
    labels = _auto_labels()
    origins = _origins()
    first_ring = _square_ring(0, 0, 5)
    second_ring = _square_ring(20, 0, 5)
    ops = [BoundaryOp.draw(1, first_ring), BoundaryOp.draw(1, second_ring)]

    out_labels, out_origins, warnings = apply_boundary_ops(labels, origins, ops)

    assert not warnings
    assert not np.any(out_labels[0:5, 0:5] == 1), "the first ring must be gone"
    assert np.any(out_labels[20:25, 0:5] == 1)


def test_delete_reverts_a_label_to_whatever_auto_currently_says():
    labels = _auto_labels()
    origins = _origins()
    ops = [BoundaryOp.draw(1, _square_ring(0, 0, 5)), BoundaryOp.delete(1)]

    out_labels, out_origins, warnings = apply_boundary_ops(labels, origins, ops)

    assert not warnings
    assert out_origins[1] == ORIGIN_FLOW   # back to the auto origin
    assert np.array_equal(out_labels == 1, labels == 1)   # back to the auto shape


def test_deleting_a_label_with_no_active_manual_boundary_warns_and_is_a_no_op():
    labels = _auto_labels()
    origins = _origins()
    ops = [BoundaryOp.delete(1)]

    out_labels, out_origins, warnings = apply_boundary_ops(labels, origins, ops)

    assert len(warnings) == 1
    assert "no active manual boundary" in warnings[0]
    assert np.array_equal(out_labels, labels)
    assert out_origins == origins


def test_redraw_after_delete_draws_again():
    labels = _auto_labels()
    origins = _origins()
    ops = [
        BoundaryOp.draw(1, _square_ring(0, 0, 5)),
        BoundaryOp.delete(1),
        BoundaryOp.draw(1, _square_ring(20, 0, 5)),
    ]

    out_labels, out_origins, _warnings = apply_boundary_ops(labels, origins, ops)

    assert out_origins[1] == ORIGIN_MANUAL
    assert np.any(out_labels[20:25, 0:5] == 1)


def test_undo_by_truncation_matches_dropping_the_last_op():
    labels = _auto_labels()
    origins = _origins()
    ops = [BoundaryOp.draw(1, _square_ring(0, 0, 5)),
           BoundaryOp.draw(1, _square_ring(20, 0, 5))]

    full_labels, _o, _w = apply_boundary_ops(labels, origins, ops)
    truncated_labels, _o2, _w2 = apply_boundary_ops(labels, origins, ops[:-1])

    assert np.any(full_labels[20:25, 0:5] == 1)
    assert not np.any(truncated_labels[20:25, 0:5] == 1)
    assert np.any(truncated_labels[0:5, 0:5] == 1)


# ── malformed / conflicting ops ──────────────────────────────────────────────


def test_a_draw_with_too_few_points_is_skipped_with_a_warning():
    labels = _auto_labels()
    origins = _origins()
    ops = [BoundaryOp.draw(1, [[0, 0], [0, 5]])]   # only 2 points

    out_labels, _out_origins, warnings = apply_boundary_ops(labels, origins, ops)

    assert any("at least 3" in w for w in warnings)
    assert np.array_equal(out_labels, labels)


def test_an_unknown_op_is_skipped_with_a_warning():
    labels = _auto_labels()
    origins = _origins()
    bogus = BoundaryOp(op="frobnicate", label=1)

    _out_labels, _out_origins, warnings = apply_boundary_ops(labels, origins, [bogus])

    assert any("unknown" in w for w in warnings)


# ── overlap: last-drawn-wins ─────────────────────────────────────────────────


def test_overlapping_manual_polygons_resolve_last_drawn_wins():
    labels = _auto_labels()
    origins = _origins()
    # Label 1's ring and label 2's ring share the block [10:15, 10:15].
    ring1 = _square_ring(0, 0, 15)
    ring2 = _square_ring(10, 10, 15)
    ops = [BoundaryOp.draw(1, ring1), BoundaryOp.draw(2, ring2)]

    out_labels, _origins2, warnings = apply_boundary_ops(labels, origins, ops)

    assert not warnings
    # Label 2 was drawn after label 1, so it owns the overlap.
    assert out_labels[12, 12] == 2


def test_active_manual_labels_tracks_draw_and_delete():
    ops = [BoundaryOp.draw(1, _square_ring(0, 0, 5)),
           BoundaryOp.draw(2, _square_ring(20, 0, 5)),
           BoundaryOp.delete(1)]
    assert active_manual_labels(ops) == {2}


# ── the precedence rule under retuned auto parameters ───────────────────────


def test_a_manual_boundary_survives_retuning_while_an_unrelated_label_updates():
    """The core precedence rule the whole feature exists for.

    Simulates ``capture_px``/``min_area`` moving by recomputing a *different*
    auto label image (label 2 grows) and replaying the same ops against it.
    Label 1's manual shape must be identical both times; label 2 must track
    the new auto computation exactly.
    """
    ops = [BoundaryOp.draw(1, _square_ring(0, 0, 5))]

    before_auto = _auto_labels()
    after_auto = _auto_labels()
    after_auto[20:38, 20:38] = 2   # label 2's auto footprint grew, e.g. a wider capture_px

    before_labels, before_origins, _w1 = apply_boundary_ops(
        before_auto, _origins(), ops)
    after_labels, after_origins, _w2 = apply_boundary_ops(
        after_auto, _origins(), ops)

    # Label 1: identical manual shape, unaffected by the retune.
    assert np.array_equal(before_labels == 1, after_labels == 1)
    assert before_origins[1] == after_origins[1] == ORIGIN_MANUAL

    # Label 2: tracks the new auto computation, not held back by anything.
    assert not np.array_equal(before_labels == 2, after_labels == 2)
    assert np.array_equal(after_labels == 2, after_auto == 2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
