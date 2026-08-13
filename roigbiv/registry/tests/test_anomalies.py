"""
Cross-session ROI anomaly reporting (:mod:`roigbiv.registry.anomalies`).

The report is derived from ``CellObservation`` rows rather than stored, so
these build observation sets by hand and assert the three patterns are named
correctly — including the one that matters most, a cell absent from a session
*between* two sightings, which no per-session counter can express.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from roigbiv.registry.anomalies import cell_timeline
from roigbiv.registry.store.base import (
    CellRecord,
    FOVRecord,
    ObservationRecord,
    SessionRecord,
)
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore


def _store() -> SQLAlchemyStore:
    store = SQLAlchemyStore(dsn="sqlite://")
    store.ensure_schema()
    return store


def _fov(store) -> str:
    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id,
        fingerprint_hash="a" * 64,
        animal_id="DS-Prism-3",
        region="DS-Prism",
        mean_m_uri="file:///m.npy",
        centroid_table_uri="file:///c.npy",
        created_at=datetime.now(timezone.utc),
    ))
    return fov_id


def _timeline(store, fov_id, n=3, ordered=True) -> list[str]:
    """*n* sessions on one date, ordered 0..n-1 unless *ordered* is False."""
    ids = []
    for i in range(n):
        session_id = str(uuid.uuid4())
        store.upsert_session(SessionRecord(
            session_id=session_id,
            fov_id=fov_id,
            session_date=date(2026, 5, 21),
            output_dir=f"/out/s{i}",
            created_at=datetime.now(timezone.utc),
            sequence_index=i if ordered else None,
        ))
        ids.append(session_id)
    return ids


def _cell(store, fov_id, session_ids, present, label_id=1) -> str:
    """A cell observed in the sessions where *present* is True.

    ``label_id`` must differ between cells sharing a session — the schema
    enforces one observation per (session, local_label_id).
    """
    gid = str(uuid.uuid4())
    store.insert_cell(CellRecord(global_cell_id=gid, fov_id=fov_id))
    store.insert_observations([
        ObservationRecord(
            observation_id=str(uuid.uuid4()),
            global_cell_id=gid,
            session_id=sid,
            local_label_id=label_id,
        )
        for sid, seen in zip(session_ids, present) if seen
    ])
    return gid


def test_cell_present_everywhere_has_no_anomaly():
    store = _store()
    fov_id = _fov(store)
    sessions = _timeline(store, fov_id)
    _cell(store, fov_id, sessions, [True, True, True])

    report = cell_timeline(store, fov_id)

    assert report.counts["n_cells"] == 1
    assert report.counts["n_complete"] == 1
    assert report.cells[0].anomalies == []


def test_late_arrival_is_flagged():
    store = _store()
    fov_id = _fov(store)
    sessions = _timeline(store, fov_id)
    _cell(store, fov_id, sessions, [False, True, True])

    report = cell_timeline(store, fov_id)
    cell = report.cells[0]

    assert cell.anomalies == ["late_arrival"]
    assert cell.first_seen == 1
    assert report.counts["late_arrival"] == 1


def test_dropout_is_flagged():
    store = _store()
    fov_id = _fov(store)
    sessions = _timeline(store, fov_id)
    _cell(store, fov_id, sessions, [True, True, False])

    report = cell_timeline(store, fov_id)
    cell = report.cells[0]

    assert cell.anomalies == ["dropout"]
    assert cell.last_seen == 1
    assert report.counts["dropout"] == 1


def test_intermittent_cell_is_flagged_and_names_the_gap():
    """A cell cannot un-exist and exist again — this is a missed detection."""
    store = _store()
    fov_id = _fov(store)
    sessions = _timeline(store, fov_id)
    _cell(store, fov_id, sessions, [True, False, True])

    report = cell_timeline(store, fov_id)
    cell = report.cells[0]

    assert cell.anomalies == ["intermittent"]
    assert cell.gap_indices == [1]
    assert not cell.is_late_arrival
    assert not cell.is_dropout


def test_a_cell_can_be_both_late_and_a_dropout():
    store = _store()
    fov_id = _fov(store)
    sessions = _timeline(store, fov_id, n=4)
    _cell(store, fov_id, sessions, [False, True, True, False])

    cell = cell_timeline(store, fov_id).cells[0]

    assert cell.anomalies == ["late_arrival", "dropout"]


def test_all_three_can_coexist():
    store = _store()
    fov_id = _fov(store)
    sessions = _timeline(store, fov_id, n=5)
    _cell(store, fov_id, sessions, [False, True, False, True, False])

    cell = cell_timeline(store, fov_id).cells[0]

    assert cell.anomalies == ["late_arrival", "dropout", "intermittent"]
    assert cell.gap_indices == [2]


def test_report_follows_human_order_not_insertion_order():
    """Anomaly classes are defined by position, so ordering is load-bearing."""
    store = _store()
    fov_id = _fov(store)
    sessions = _timeline(store, fov_id)
    # Seen only in the session a human placed last.
    _cell(store, fov_id, [sessions[2]], [True])

    cell = cell_timeline(store, fov_id).cells[0]

    assert cell.first_seen == 2
    assert cell.is_late_arrival
    assert not cell.is_dropout


def test_unconfirmed_ordering_is_surfaced():
    store = _store()
    fov_id = _fov(store)
    _timeline(store, fov_id, ordered=False)

    report = cell_timeline(store, fov_id)

    assert report.ordering_is_confirmed is False


def test_confirmed_ordering_is_surfaced():
    store = _store()
    fov_id = _fov(store)
    _timeline(store, fov_id, ordered=True)

    assert cell_timeline(store, fov_id).ordering_is_confirmed is True


def test_local_label_ids_are_carried_per_session():
    """The label id is how a caller finds the ROI in that session's masks."""
    store = _store()
    fov_id = _fov(store)
    sessions = _timeline(store, fov_id)
    gid = str(uuid.uuid4())
    store.insert_cell(CellRecord(global_cell_id=gid, fov_id=fov_id))
    store.insert_observations([
        ObservationRecord(observation_id=str(uuid.uuid4()), global_cell_id=gid,
                          session_id=sessions[0], local_label_id=4),
        ObservationRecord(observation_id=str(uuid.uuid4()), global_cell_id=gid,
                          session_id=sessions[2], local_label_id=9),
    ])

    cell = cell_timeline(store, fov_id).cells[0]

    assert cell.local_label_ids == [4, None, 9]


def test_fov_with_no_sessions_reports_empty():
    store = _store()
    fov_id = _fov(store)

    report = cell_timeline(store, fov_id)

    assert report.sessions == []
    assert report.cells == []
    assert report.ordering_is_confirmed is False


def test_cells_are_ordered_by_first_appearance():
    store = _store()
    fov_id = _fov(store)
    sessions = _timeline(store, fov_id)
    late = _cell(store, fov_id, sessions, [False, False, True], label_id=1)
    early = _cell(store, fov_id, sessions, [True, True, True], label_id=2)

    report = cell_timeline(store, fov_id)

    assert [c.global_cell_id for c in report.cells] == [early, late]
