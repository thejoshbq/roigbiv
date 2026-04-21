from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from roigbiv.registry.store.base import (
    CellRecord,
    FOVRecord,
    ObservationRecord,
    SessionRecord,
)
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore


def _new_store() -> SQLAlchemyStore:
    store = SQLAlchemyStore(dsn="sqlite://")
    store.ensure_schema()
    return store


def test_insert_and_get_fov():
    store = _new_store()
    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id,
        fingerprint_hash="a" * 64,
        animal_id="PrL-NAc-G6-5M_HI-D1",
        region="PrL-NAc",
        mean_m_uri="file:///tmp/mean.npy",
        centroid_table_uri="file:///tmp/c.npy",
        created_at=datetime.now(timezone.utc),
        latest_session_date=date(2022, 12, 9),
    ))
    by_hash = store.get_fov_by_hash("a" * 64)
    assert by_hash is not None
    assert by_hash.fov_id == fov_id
    by_id = store.get_fov(fov_id)
    assert by_id is not None
    assert by_id.animal_id == "PrL-NAc-G6-5M_HI-D1"


def test_find_candidates_filters_by_animal_and_region():
    store = _new_store()
    for animal, region in [
        ("A", "PrL-NAc"),
        ("A", "PVT"),
        ("B", "PrL-NAc"),
    ]:
        store.insert_fov(FOVRecord(
            fov_id=str(uuid.uuid4()),
            fingerprint_hash=str(uuid.uuid4()).replace("-", "")[:64].ljust(64, "0"),
            animal_id=animal,
            region=region,
            mean_m_uri="file:///x",
            centroid_table_uri="file:///x",
            created_at=datetime.now(timezone.utc),
        ))
    assert len(store.find_candidates("A", "PrL-NAc")) == 1
    assert len(store.find_candidates("A", "PVT")) == 1
    assert len(store.find_candidates("B", "PVT")) == 0


def test_session_observation_cell_roundtrip():
    store = _new_store()
    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id,
        fingerprint_hash="b" * 64,
        animal_id="X",
        region="PrL",
        mean_m_uri="file:///x",
        centroid_table_uri="file:///x",
        created_at=datetime.now(timezone.utc),
    ))
    session_id = str(uuid.uuid4())
    store.insert_session(SessionRecord(
        session_id=session_id,
        fov_id=fov_id,
        session_date=date(2022, 12, 9),
        output_dir="/tmp/out",
        n_matched=0, n_new=2, n_missing=0,
        created_at=datetime.now(timezone.utc),
    ))
    cell1 = str(uuid.uuid4())
    cell2 = str(uuid.uuid4())
    for gid in (cell1, cell2):
        store.insert_cell(CellRecord(
            global_cell_id=gid,
            fov_id=fov_id,
            first_seen_session_id=session_id,
            morphology_summary={"first_local_label_id": 1 if gid == cell1 else 2},
        ))
    store.insert_observations([
        ObservationRecord(observation_id=str(uuid.uuid4()),
                          global_cell_id=cell1, session_id=session_id,
                          local_label_id=1, match_score=None),
        ObservationRecord(observation_id=str(uuid.uuid4()),
                          global_cell_id=cell2, session_id=session_id,
                          local_label_id=2, match_score=None),
    ])

    assert len(store.list_cells(fov_id)) == 2
    assert len(store.list_sessions(fov_id)) == 1
    obs = store.list_observations_for_cell(cell1)
    assert len(obs) == 1
    assert obs[0].local_label_id == 1

    store.update_fov_latest_session(fov_id, date(2023, 1, 1))
    assert store.get_fov(fov_id).latest_session_date == date(2023, 1, 1)
    store.update_fov_latest_session(fov_id, date(2022, 1, 1))
    assert store.get_fov(fov_id).latest_session_date == date(2023, 1, 1)
