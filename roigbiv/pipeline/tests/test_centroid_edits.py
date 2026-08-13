"""Tests for :mod:`roigbiv.pipeline.centroid_edits`."""
from __future__ import annotations

import json
from pathlib import Path

from roigbiv.pipeline.centroid_edits import (
    CentroidOp,
    append_centroid_op,
    apply_centroid_ops,
    centroid_log_path,
    load_centroid_ops,
    next_label,
    write_centroid_ops,
)


def test_op_round_trips_through_dict() -> None:
    op = CentroidOp.add(7, 10.5, 20.25, notes="hello")
    restored = CentroidOp.from_dict(op.to_jsonable())
    assert restored.op == "add"
    assert restored.label == 7
    assert restored.y == 10.5
    assert restored.x == 20.25
    assert restored.notes == "hello"


def test_apply_add_inserts_new_centroid() -> None:
    base = {1: (0.0, 0.0)}
    op = CentroidOp.add(2, 10.0, 20.0)
    effective, warnings = apply_centroid_ops(base, [op])
    assert effective == {1: (0.0, 0.0), 2: (10.0, 20.0)}
    assert warnings == []


def test_apply_delete_removes_centroid() -> None:
    base = {1: (0.0, 0.0), 2: (10.0, 20.0)}
    op = CentroidOp.delete(1)
    effective, warnings = apply_centroid_ops(base, [op])
    assert effective == {2: (10.0, 20.0)}
    assert warnings == []


def test_apply_move_updates_position() -> None:
    base = {1: (0.0, 0.0)}
    op = CentroidOp.move(1, 5.0, 5.0)
    effective, warnings = apply_centroid_ops(base, [op])
    assert effective == {1: (5.0, 5.0)}
    assert warnings == []


def test_delete_then_add_does_not_reuse_label() -> None:
    # Adding label 3, then deleting it, must not free 3 back up — a link
    # log elsewhere may already reference (session, 3) meaning something
    # else entirely by the time a later add runs.
    base = {1: (0.0, 0.0), 2: (5.0, 5.0)}
    ops = [CentroidOp.add(3, 1.0, 1.0), CentroidOp.delete(3)]
    assert next_label(base, ops) == 4


def test_next_label_empty_base_and_ops() -> None:
    assert next_label({}, []) == 1


def test_next_label_ignores_nothing_from_base() -> None:
    base = {1: (0.0, 0.0), 5: (1.0, 1.0)}
    assert next_label(base, []) == 6


def test_replay_is_deterministic() -> None:
    base = {1: (0.0, 0.0)}
    ops = [
        CentroidOp.add(2, 10.0, 10.0),
        CentroidOp.move(1, 1.0, 1.0),
        CentroidOp.delete(2),
    ]
    first, warnings_1 = apply_centroid_ops(base, ops)
    second, warnings_2 = apply_centroid_ops(base, ops)
    assert first == second
    assert warnings_1 == warnings_2 == []


def test_apply_centroid_ops_does_not_mutate_base() -> None:
    base = {1: (0.0, 0.0)}
    apply_centroid_ops(base, [CentroidOp.add(2, 1.0, 1.0)])
    assert base == {1: (0.0, 0.0)}


def test_append_then_load_preserves_order(tmp_path: Path) -> None:
    ops = [
        CentroidOp.add(1, 0.0, 0.0),
        CentroidOp.move(1, 1.0, 1.0),
        CentroidOp.delete(1),
    ]
    for op in ops:
        append_centroid_op(tmp_path, op)
    loaded = load_centroid_ops(tmp_path)
    assert [o.op for o in loaded] == ["add", "move", "delete"]
    assert [o.label for o in loaded] == [1, 1, 1]


def test_load_centroid_ops_missing_log_returns_empty(tmp_path: Path) -> None:
    assert load_centroid_ops(tmp_path) == []


def test_write_centroid_ops_empty_removes_log(tmp_path: Path) -> None:
    append_centroid_op(tmp_path, CentroidOp.add(1, 0.0, 0.0))
    assert centroid_log_path(tmp_path).exists()
    write_centroid_ops(tmp_path, [])
    assert not centroid_log_path(tmp_path).exists()


def test_write_centroid_ops_rewrites_log(tmp_path: Path) -> None:
    ops = [CentroidOp.add(1, 0.0, 0.0), CentroidOp.add(2, 1.0, 1.0)]
    write_centroid_ops(tmp_path, ops)
    loaded = load_centroid_ops(tmp_path)
    assert [o.label for o in loaded] == [1, 2]


def test_centroid_log_path_is_under_corrections_dir(tmp_path: Path) -> None:
    path = centroid_log_path(tmp_path)
    assert path == tmp_path / "corrections" / "centroids.jsonl"


# ── warning cases — each skips only the offending op ────────────────────────


def test_add_on_occupied_label_is_skipped_with_warning() -> None:
    base = {1: (0.0, 0.0)}
    effective, warnings = apply_centroid_ops(base, [CentroidOp.add(1, 9.0, 9.0)])
    assert effective == base
    assert len(warnings) == 1
    assert "add" in warnings[0] and "1" in warnings[0]


def test_delete_of_absent_label_is_skipped_with_warning() -> None:
    base = {1: (0.0, 0.0)}
    effective, warnings = apply_centroid_ops(base, [CentroidOp.delete(99)])
    assert effective == base
    assert len(warnings) == 1
    assert "delete" in warnings[0] and "99" in warnings[0]


def test_move_of_absent_label_is_skipped_with_warning() -> None:
    base = {1: (0.0, 0.0)}
    effective, warnings = apply_centroid_ops(
        base, [CentroidOp.move(99, 9.0, 9.0)])
    assert effective == base
    assert len(warnings) == 1
    assert "move" in warnings[0] and "99" in warnings[0]


def test_unknown_op_is_skipped_with_warning() -> None:
    base = {1: (0.0, 0.0)}
    bad = CentroidOp.from_dict({"op": "frobnicate", "label": 1})
    effective, warnings = apply_centroid_ops(base, [bad])
    assert effective == base
    assert len(warnings) == 1
    assert "frobnicate" in warnings[0] or "unknown" in warnings[0]


def test_add_missing_coords_is_skipped_with_warning() -> None:
    base = {1: (0.0, 0.0)}
    bad = CentroidOp.from_dict({"op": "add", "label": 2})
    effective, warnings = apply_centroid_ops(base, [bad])
    assert effective == base
    assert len(warnings) == 1
    assert "add" in warnings[0] and "2" in warnings[0]


def test_move_missing_coords_is_skipped_with_warning() -> None:
    base = {1: (0.0, 0.0)}
    bad = CentroidOp.from_dict({"op": "move", "label": 1})
    effective, warnings = apply_centroid_ops(base, [bad])
    assert effective == base
    assert len(warnings) == 1
    assert "move" in warnings[0] and "1" in warnings[0]


def test_bad_op_does_not_break_the_rest_of_the_replay() -> None:
    base = {1: (0.0, 0.0)}
    ops = [
        CentroidOp.delete(99),                 # bad — skipped
        CentroidOp.add(2, 5.0, 5.0),            # good
    ]
    effective, warnings = apply_centroid_ops(base, ops)
    assert effective == {1: (0.0, 0.0), 2: (5.0, 5.0)}
    assert len(warnings) == 1


def test_op_ids_and_timestamps_are_populated_and_json_serializable() -> None:
    op = CentroidOp.add(1, 0.0, 0.0)
    payload = op.to_jsonable()
    assert "id" in payload and "ts" in payload
    json.dumps(payload)  # must not raise
