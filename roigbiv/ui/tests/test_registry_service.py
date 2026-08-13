"""
Registry reads the UI depends on (:mod:`roigbiv.ui.services.registry_service`).

The Track page's anomaly panel must work for a workspace tracked *elsewhere* —
from ``roigbiv-pipeline --track`` or in an earlier browser session — so these
go through a real store rather than a mock: the interesting failure is a
lookup that misses because the path was spelled differently, which no mock
would ever reproduce.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from roigbiv.registry.store.base import (
    CellRecord,
    FOVRecord,
    ObservationRecord,
    SessionRecord,
)
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore
from roigbiv.ui.services.registry_service import anomaly_payload, workspace_anomalies

STEMS = ["fov_pre-005", "fov_beh-006", "fov_post-007"]


def _store() -> SQLAlchemyStore:
    store = SQLAlchemyStore(dsn="sqlite://")
    store.ensure_schema()
    return store


def _tracked_fov(store, output_root: Path, presence: dict[str, list[bool]]) -> str:
    """One FOV whose sessions live under *output_root*, in STEMS order."""
    fov_id = str(uuid.uuid4())
    store.insert_fov(FOVRecord(
        fov_id=fov_id, fingerprint_hash="a" * 64, animal_id="DS-Prism-3",
        region="DS-Prism", mean_m_uri="file:///m.npy",
        centroid_table_uri="file:///c.npy",
        created_at=datetime.now(timezone.utc),
    ))
    session_ids = []
    for i, stem in enumerate(STEMS):
        session_id = str(uuid.uuid4())
        store.upsert_session(SessionRecord(
            session_id=session_id, fov_id=fov_id, session_date=date(2026, 5, 21),
            output_dir=str(output_root / stem),
            created_at=datetime.now(timezone.utc), sequence_index=i,
        ))
        session_ids.append(session_id)

    for label_id, (_name, present) in enumerate(presence.items(), start=1):
        gcid = str(uuid.uuid4())
        store.insert_cell(CellRecord(global_cell_id=gcid, fov_id=fov_id))
        store.insert_observations([
            ObservationRecord(observation_id=str(uuid.uuid4()),
                              global_cell_id=gcid, session_id=sid,
                              local_label_id=label_id)
            for sid, seen in zip(session_ids, present) if seen
        ])
    return fov_id


def _dirs(output_root: Path) -> list[Path]:
    return [output_root / stem for stem in STEMS]


def test_untracked_workspace_reports_nothing(tmp_path):
    store = _store()
    with patch("roigbiv.registry.build_store", return_value=store):
        assert workspace_anomalies(_dirs(tmp_path / "output")) == {}


def test_a_tracked_workspace_reports_its_fov(tmp_path):
    output_root = tmp_path / "output"
    store = _store()
    fov_id = _tracked_fov(store, output_root, {
        "everywhere": [True, True, True],
        "dropout": [True, True, False],
    })

    with patch("roigbiv.registry.build_store", return_value=store):
        reports = workspace_anomalies(_dirs(output_root))

    assert list(reports) == [fov_id]
    counts = reports[fov_id]["counts"]
    assert counts == {"n_sessions": 3, "n_cells": 2, "n_complete": 1,
                      "late_arrival": 0, "dropout": 1, "intermittent": 0}


def test_only_anomalous_cells_are_carried_to_the_browser(tmp_path):
    """A clean FOV would otherwise ship every cell on every poll."""
    output_root = tmp_path / "output"
    store = _store()
    fov_id = _tracked_fov(store, output_root, {
        "everywhere": [True, True, True],
        "late": [False, True, True],
    })

    with patch("roigbiv.registry.build_store", return_value=store):
        reports = workspace_anomalies(_dirs(output_root))

    cells = reports[fov_id]["cells"]
    assert [c["anomalies"] for c in cells] == [["late_arrival"]]
    assert cells[0]["present"] == [False, True, True]


def test_sessions_come_back_in_the_human_confirmed_order(tmp_path):
    output_root = tmp_path / "output"
    store = _store()
    fov_id = _tracked_fov(store, output_root, {"everywhere": [True, True, True]})

    with patch("roigbiv.registry.build_store", return_value=store):
        reports = workspace_anomalies(_dirs(output_root))

    sessions = reports[fov_id]["sessions"]
    assert [s["sequence_index"] for s in sessions] == [0, 1, 2]
    assert [Path(s["output_dir"]).name for s in sessions] == STEMS
    assert reports[fov_id]["ordering_is_confirmed"] is True


def test_an_unresolved_output_path_still_finds_its_session(tmp_path):
    """Sessions are keyed by whatever string the registering run passed in.

    A workspace opened through a symlink or a relative path would otherwise
    look untracked, and the panel would claim there is nothing to report.
    """
    output_root = tmp_path / "output"
    for stem in STEMS:
        (output_root / stem).mkdir(parents=True)

    store = _store()
    _tracked_fov(store, output_root, {"everywhere": [True, True, True]})

    link = tmp_path / "via-link"
    link.symlink_to(output_root)

    with patch("roigbiv.registry.build_store", return_value=store):
        reports = workspace_anomalies([link / stem for stem in STEMS])

    assert len(reports) == 1


def test_payload_is_json_safe(tmp_path):
    """It crosses into a dcc component tree, so no dates or dataclasses."""
    import json

    from roigbiv.registry.anomalies import cell_timeline

    output_root = tmp_path / "output"
    store = _store()
    fov_id = _tracked_fov(store, output_root, {"late": [False, True, True]})

    payload = anomaly_payload(cell_timeline(store, fov_id))
    assert json.loads(json.dumps(payload)) == payload
