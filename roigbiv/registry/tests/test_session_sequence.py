"""Human-assigned session ordering (``session.sequence_index``, Alembic 0006).

Filename dates cannot order this lab's timelines — six-digit groups are
ambiguous between two conventions, and the reference prism workspace records
``pre`` / ``beh`` / ``post`` on one date. These cover that a human order is
stored, wins over dates, and degrades to date order when unset.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from roigbiv.registry.store.base import FOVRecord, SessionRecord
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore


def _new_store() -> SQLAlchemyStore:
    store = SQLAlchemyStore(dsn="sqlite://")
    store.ensure_schema()
    return store


def _fov(store: SQLAlchemyStore) -> str:
    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id,
        fingerprint_hash="a" * 64,
        animal_id="DS-Prism-3",
        region="DS-Prism",
        mean_m_uri="file:///tmp/mean.npy",
        centroid_table_uri="file:///tmp/c.npy",
        created_at=datetime.now(timezone.utc),
    ))
    return fov_id


def _session(store, fov_id, output_dir, session_date, sequence_index=None) -> str:
    session_id = str(uuid.uuid4())
    store.upsert_session(SessionRecord(
        session_id=session_id,
        fov_id=fov_id,
        session_date=session_date,
        output_dir=output_dir,
        created_at=datetime.now(timezone.utc),
        sequence_index=sequence_index,
    ))
    return session_id


def test_sequence_index_defaults_to_null():
    store = _new_store()
    fov_id = _fov(store)
    _session(store, fov_id, "/out/a", date(2026, 5, 21))

    assert store.list_sessions(fov_id)[0].sequence_index is None


def test_sequence_index_round_trips_through_insert():
    store = _new_store()
    fov_id = _fov(store)
    _session(store, fov_id, "/out/a", date(2026, 5, 21), sequence_index=2)

    assert store.list_sessions(fov_id)[0].sequence_index == 2


def test_update_session_sequence_sets_and_clears():
    store = _new_store()
    fov_id = _fov(store)
    session_id = _session(store, fov_id, "/out/a", date(2026, 5, 21))

    store.update_session_sequence(session_id, 3)
    assert store.list_sessions(fov_id)[0].sequence_index == 3

    store.update_session_sequence(session_id, None)
    assert store.list_sessions(fov_id)[0].sequence_index is None


def test_update_session_sequence_on_unknown_id_is_a_no_op():
    store = _new_store()
    store.update_session_sequence(str(uuid.uuid4()), 1)  # must not raise


def test_human_order_wins_over_dates():
    """The reference case: three same-day sessions no date can order."""
    store = _new_store()
    fov_id = _fov(store)
    same_day = date(2026, 5, 21)
    _session(store, fov_id, "/out/post-007", same_day, sequence_index=2)
    _session(store, fov_id, "/out/pre-005", same_day, sequence_index=0)
    _session(store, fov_id, "/out/beh-006", same_day, sequence_index=1)

    ordered = [s.output_dir for s in store.list_sessions(fov_id)]
    assert ordered == ["/out/pre-005", "/out/beh-006", "/out/post-007"]


def test_human_order_overrides_conflicting_dates():
    store = _new_store()
    fov_id = _fov(store)
    _session(store, fov_id, "/out/late", date(2026, 6, 1), sequence_index=0)
    _session(store, fov_id, "/out/early", date(2026, 5, 21), sequence_index=1)

    ordered = [s.output_dir for s in store.list_sessions(fov_id)]
    assert ordered == ["/out/late", "/out/early"]


def test_unordered_sessions_fall_back_to_date_order():
    store = _new_store()
    fov_id = _fov(store)
    _session(store, fov_id, "/out/b", date(2026, 6, 1))
    _session(store, fov_id, "/out/a", date(2026, 5, 21))

    ordered = [s.output_dir for s in store.list_sessions(fov_id)]
    assert ordered == ["/out/a", "/out/b"]


def test_partially_ordered_fov_puts_ordered_sessions_first():
    """A half-ordered FOV must still read sensibly rather than interleave."""
    store = _new_store()
    fov_id = _fov(store)
    _session(store, fov_id, "/out/unordered", date(2026, 5, 1))
    _session(store, fov_id, "/out/ordered", date(2026, 6, 1), sequence_index=0)

    ordered = [s.output_dir for s in store.list_sessions(fov_id)]
    assert ordered == ["/out/ordered", "/out/unordered"]
