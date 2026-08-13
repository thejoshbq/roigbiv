"""Tests for :mod:`roigbiv.registry.cell_edits`."""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.centroid_edits import CentroidOp, append_centroid_op
from roigbiv.registry.cell_edits import (
    MatchOp,
    append_match_op,
    apply_match_ops,
    apply_tracking_edits,
    load_match_ops,
    match_log_path,
    write_match_ops,
)
from roigbiv.registry.store.base import FOVRecord, SessionRecord
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore

ORDER = {"s1": 0, "s2": 1, "s3": 2}


def _assignment(**members) -> dict:
    """``{("s1", 1): "cell-a", ...}`` from ``s1_1="cell-a"`` kwargs."""
    out = {}
    for key, gcid in members.items():
        stem, label = key.rsplit("_", 1)
        out[(stem, int(label))] = gcid
    return out


# ── op round-trip ────────────────────────────────────────────────────────


def test_op_round_trips_through_dict() -> None:
    op = MatchOp.link("fov-1", [("s1", 1), ("s2", 2)], notes="hi")
    restored = MatchOp.from_dict(op.to_jsonable())
    assert restored.op == "link"
    assert restored.fov_id == "fov-1"
    assert restored.members == [["s1", 1], ["s2", 2]]
    assert restored.notes == "hi"


def test_unlink_op_round_trips() -> None:
    op = MatchOp.unlink("fov-1", ("s1", 1))
    restored = MatchOp.from_dict(op.to_jsonable())
    assert restored.op == "unlink"
    assert restored.member == ["s1", 1]


# ── log I/O ──────────────────────────────────────────────────────────────


def test_append_then_load_preserves_order(tmp_path: Path) -> None:
    ops = [
        MatchOp.link("fov-1", [("s1", 1), ("s2", 1)]),
        MatchOp.unlink("fov-1", ("s1", 1)),
    ]
    for op in ops:
        append_match_op(tmp_path, op)
    loaded = load_match_ops(tmp_path, "fov-1")
    assert [o.op for o in loaded] == ["link", "unlink"]


def test_undo_last_on_one_fov_cannot_touch_another(tmp_path: Path) -> None:
    append_match_op(tmp_path, MatchOp.link("fov-A", [("s1", 1), ("s2", 1)]))
    append_match_op(tmp_path, MatchOp.link("fov-B", [("s1", 2), ("s2", 2)]))

    write_match_ops(tmp_path, "fov-A", [])  # undo fov-A's only op

    assert load_match_ops(tmp_path, "fov-A") == []
    assert len(load_match_ops(tmp_path, "fov-B")) == 1


def test_load_missing_log_returns_empty(tmp_path: Path) -> None:
    assert load_match_ops(tmp_path, "fov-none") == []


def test_write_empty_removes_log(tmp_path: Path) -> None:
    append_match_op(tmp_path, MatchOp.link("fov-1", [("s1", 1), ("s2", 1)]))
    assert match_log_path(tmp_path, "fov-1").exists()
    write_match_ops(tmp_path, "fov-1", [])
    assert not match_log_path(tmp_path, "fov-1").exists()


# ── link: basic merge ────────────────────────────────────────────────────


def test_link_merges_two_cells() -> None:
    assignment = _assignment(s1_1="cell-a", s2_1="cell-b")
    op = MatchOp.link("fov-1", [("s1", 1), ("s2", 1)])
    effective, warnings = apply_match_ops(assignment, [op], order=ORDER)
    assert warnings == []
    assert effective[("s1", 1)] == effective[("s2", 1)]


def test_link_survivor_is_the_earliest_sequence_index() -> None:
    # cell-b's only member is in s1 (seq 0); cell-a's is in s2 (seq 1).
    # cell-b must survive even though it's alphabetically "later".
    assignment = _assignment(s1_1="cell-b", s2_1="cell-a")
    op = MatchOp.link("fov-1", [("s1", 1), ("s2", 1)])
    effective, _ = apply_match_ops(assignment, [op], order=ORDER)
    assert effective[("s1", 1)] == "cell-b"
    assert effective[("s2", 1)] == "cell-b"


def test_link_tie_break_is_smaller_gcid_string() -> None:
    # Equal sequence_index on both sides — smaller gcid string wins.
    order = {"s1": 0, "s2": 0}
    assignment = _assignment(s1_1="cell-z", s2_1="cell-a")
    op = MatchOp.link("fov-1", [("s1", 1), ("s2", 1)])
    effective, _ = apply_match_ops(assignment, [op], order=order)
    assert effective[("s1", 1)] == "cell-a"
    assert effective[("s2", 1)] == "cell-a"


def test_link_merges_whole_cells_not_just_clicked_members() -> None:
    # cell-a also has a member in s3 that nobody clicked — it must come along.
    assignment = _assignment(s1_1="cell-a", s3_1="cell-a", s2_1="cell-b")
    op = MatchOp.link("fov-1", [("s1", 1), ("s2", 1)])
    effective, _ = apply_match_ops(assignment, [op], order=ORDER)
    survivor = effective[("s1", 1)]
    assert effective[("s2", 1)] == survivor
    assert effective[("s3", 1)] == survivor


def test_three_way_merge_chain() -> None:
    assignment = _assignment(s1_1="cell-a", s2_1="cell-b", s3_1="cell-c")
    ops = [
        MatchOp.link("fov-1", [("s1", 1), ("s2", 1)]),
        MatchOp.link("fov-1", [("s2", 1), ("s3", 1)]),
    ]
    effective, warnings = apply_match_ops(assignment, ops, order=ORDER)
    assert warnings == []
    survivor = effective[("s1", 1)]
    assert effective[("s2", 1)] == survivor
    assert effective[("s3", 1)] == survivor


# ── link: rejection and drops ────────────────────────────────────────────


def test_link_naming_a_deleted_member_drops_it_and_keeps_the_rest() -> None:
    # (s2, 99) is not in `assignment` — its centroid was deleted.
    assignment = _assignment(s1_1="cell-a", s3_1="cell-b")
    op = MatchOp.link("fov-1", [("s1", 1), ("s2", 99), ("s3", 1)])
    effective, warnings = apply_match_ops(assignment, [op], order=ORDER)
    assert len(warnings) == 1
    assert "s2" in warnings[0] and "99" in warnings[0]
    assert effective[("s1", 1)] == effective[("s3", 1)]


def test_link_with_two_members_from_one_stem_is_rejected_whole() -> None:
    assignment = _assignment(s1_1="cell-a", s1_2="cell-b")
    op = MatchOp.link("fov-1", [("s1", 1), ("s1", 2)])
    effective, warnings = apply_match_ops(assignment, [op], order=ORDER)
    assert effective == assignment  # no change at all
    assert len(warnings) == 1
    assert "s1" in warnings[0]


def test_link_rejected_when_existing_cell_members_collide_by_stem() -> None:
    # cell-a already has s1:1 and s1:2 (pre-existing — not from this op).
    # Linking s1:2's cell to s2:1's cell would still put two s1 members in
    # one cell, so it must be rejected even though the op itself names s1
    # only once.
    assignment = _assignment(s1_1="cell-a", s1_2="cell-a", s2_1="cell-b")
    op = MatchOp.link("fov-1", [("s1", 2), ("s2", 1)])
    effective, warnings = apply_match_ops(assignment, [op], order=ORDER)
    assert effective == assignment
    assert len(warnings) == 1


def test_link_with_fewer_than_two_members_is_skipped() -> None:
    assignment = _assignment(s1_1="cell-a")
    op = MatchOp.link("fov-1", [("s1", 1)])
    effective, warnings = apply_match_ops(assignment, [op], order=ORDER)
    assert effective == assignment
    assert len(warnings) == 1


def test_link_dropping_down_to_one_present_member_is_skipped() -> None:
    assignment = _assignment(s1_1="cell-a")
    op = MatchOp.link("fov-1", [("s1", 1), ("s2", 99)])
    effective, warnings = apply_match_ops(assignment, [op], order=ORDER)
    assert effective == assignment
    # one warning for the drop, one for the resulting skip
    assert len(warnings) == 2


# ── unlink ───────────────────────────────────────────────────────────────


def test_unlink_gives_a_member_its_own_cell() -> None:
    assignment = _assignment(s1_1="cell-a", s2_1="cell-a")
    op = MatchOp.unlink("fov-1", ("s1", 1))
    effective, warnings = apply_match_ops(assignment, [op], order=ORDER)
    assert warnings == []
    assert effective[("s1", 1)] != effective[("s2", 1)]


def test_unlink_of_a_singleton_is_a_noop_with_warning() -> None:
    assignment = _assignment(s1_1="cell-a")
    op = MatchOp.unlink("fov-1", ("s1", 1))
    effective, warnings = apply_match_ops(assignment, [op], order=ORDER)
    assert effective == assignment
    assert len(warnings) == 1
    assert "only member" in warnings[0]


def test_unlink_of_absent_member_is_skipped_with_warning() -> None:
    assignment = _assignment(s1_1="cell-a")
    op = MatchOp.unlink("fov-1", ("s2", 99))
    effective, warnings = apply_match_ops(assignment, [op], order=ORDER)
    assert effective == assignment
    assert len(warnings) == 1


def test_unlink_new_gcid_is_deterministic_across_calls() -> None:
    assignment = _assignment(s1_1="cell-a", s2_1="cell-a")
    op = MatchOp.unlink("fov-1", ("s1", 1))
    first, _ = apply_match_ops(assignment, [op], order=ORDER)
    second, _ = apply_match_ops(assignment, [op], order=ORDER)
    assert first[("s1", 1)] == second[("s1", 1)]


def test_unlink_then_link_back() -> None:
    assignment = _assignment(s1_1="cell-a", s2_1="cell-a")
    ops = [
        MatchOp.unlink("fov-1", ("s1", 1)),
        MatchOp.link("fov-1", [("s1", 1), ("s2", 1)]),
    ]
    effective, warnings = apply_match_ops(assignment, ops, order=ORDER)
    assert warnings == []
    assert effective[("s1", 1)] == effective[("s2", 1)]


# ── determinism / purity ─────────────────────────────────────────────────


def test_same_log_applied_twice_gives_the_same_result() -> None:
    assignment = _assignment(s1_1="cell-a", s2_1="cell-b", s3_1="cell-c")
    ops = [
        MatchOp.link("fov-1", [("s1", 1), ("s2", 1)]),
        MatchOp.unlink("fov-1", ("s2", 1)),
        MatchOp.link("fov-1", [("s2", 1), ("s3", 1)]),
    ]
    first, warnings_1 = apply_match_ops(assignment, ops, order=ORDER)
    second, warnings_2 = apply_match_ops(assignment, ops, order=ORDER)
    assert first == second
    assert warnings_1 == warnings_2


def test_apply_match_ops_does_not_mutate_assignment() -> None:
    assignment = _assignment(s1_1="cell-a", s2_1="cell-b")
    snapshot = dict(assignment)
    apply_match_ops(assignment, [MatchOp.link("fov-1", [("s1", 1), ("s2", 1)])],
                    order=ORDER)
    assert assignment == snapshot


def test_unknown_op_is_skipped_with_warning() -> None:
    assignment = _assignment(s1_1="cell-a")
    bad = MatchOp.from_dict({"op": "frobnicate", "fov_id": "fov-1"})
    effective, warnings = apply_match_ops(assignment, [bad], order=ORDER)
    assert effective == assignment
    assert len(warnings) == 1


# ── apply_tracking_edits — the DB materializer ────────────────────────────


def _write_centroid_session(out_dir: Path, points: list) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_dir / "summary" / "mean_M.tif"),
                     np.zeros((64, 64), dtype=np.float32))
    (out_dir / "centroids.json").write_text(json.dumps({
        "stem": out_dir.name, "schema": 4,
        "centroids": [
            {"label_id": i, "y": float(y), "x": float(x), "npix": 50,
             "cellpose_prob": 0.9}
            for i, (y, x) in enumerate(points)
        ],
    }))


@pytest.fixture
def two_sessions(tmp_path):
    """Two sessions, each two centroids far enough apart not to collide."""
    store = SQLAlchemyStore(dsn=f"sqlite:///{tmp_path / 'registry.db'}")
    store.ensure_schema()
    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id, fingerprint_hash="a" * 64, animal_id="X", region="Y",
        mean_m_uri="file:///m", centroid_table_uri="file:///c",
        created_at=datetime.now(timezone.utc)))

    stems = ["sess-a", "sess-b"]
    points = [(10.0, 10.0), (40.0, 40.0)]
    session_ids = []
    for i, stem in enumerate(stems):
        out_dir = tmp_path / stem
        _write_centroid_session(out_dir, points)
        sid = str(uuid.uuid4())
        store.upsert_session(SessionRecord(
            session_id=sid, fov_id=fov_id, session_date=date(2026, 1, 1),
            output_dir=str(out_dir), created_at=datetime.now(timezone.utc),
            sequence_index=i))
        session_ids.append(sid)
    return store, fov_id, tmp_path, stems, session_ids


def test_apply_tracking_edits_creates_a_cell_per_present_label(two_sessions):
    store, fov_id, root, stems, session_ids = two_sessions
    report = apply_tracking_edits(fov_id, root, store)

    assert report.n_sessions == 2
    assert report.n_observations == 4       # 2 labels x 2 sessions
    assert report.n_cells_created == 4       # nothing links them yet
    assert report.warnings == []
    for sid in session_ids:
        assert len(store.list_observations_for_session(sid)) == 2


def test_apply_tracking_edits_preserves_an_existing_gcid(two_sessions):
    store, fov_id, root, stems, session_ids = two_sessions
    apply_tracking_edits(fov_id, root, store)
    before = {(stems[i], o.local_label_id): o.global_cell_id
             for i, sid in enumerate(session_ids)
             for o in store.list_observations_for_session(sid)}

    report = apply_tracking_edits(fov_id, root, store)

    after = {(stems[i], o.local_label_id): o.global_cell_id
            for i, sid in enumerate(session_ids)
            for o in store.list_observations_for_session(sid)}
    assert after == before
    assert report.n_cells_created == 0  # every gcid already existed


def test_apply_tracking_edits_applies_a_link_op(two_sessions):
    store, fov_id, root, stems, session_ids = two_sessions
    apply_tracking_edits(fov_id, root, store)  # establish baseline gcids

    append_match_op(root, MatchOp.link(fov_id, [(stems[0], 1), (stems[1], 1)]))
    apply_tracking_edits(fov_id, root, store)

    obs_a = {o.local_label_id: o.global_cell_id
            for o in store.list_observations_for_session(session_ids[0])}
    obs_b = {o.local_label_id: o.global_cell_id
            for o in store.list_observations_for_session(session_ids[1])}
    assert obs_a[1] == obs_b[1]


def test_apply_tracking_edits_reflects_a_centroid_delete(two_sessions):
    store, fov_id, root, stems, session_ids = two_sessions
    append_centroid_op(root / stems[0], CentroidOp.delete(1))

    report = apply_tracking_edits(fov_id, root, store)

    obs_a = store.list_observations_for_session(session_ids[0])
    assert [o.local_label_id for o in obs_a] == [2]
    assert report.n_observations == 3  # 1 (sess-a) + 2 (sess-b)


def test_apply_tracking_edits_warns_when_an_observation_loses_its_centroid(
        two_sessions):
    store, fov_id, root, stems, session_ids = two_sessions
    apply_tracking_edits(fov_id, root, store)  # observation for label 1 exists

    append_centroid_op(root / stems[0], CentroidOp.delete(1))
    report = apply_tracking_edits(fov_id, root, store)

    assert any("dropped" in w for w in report.warnings)
