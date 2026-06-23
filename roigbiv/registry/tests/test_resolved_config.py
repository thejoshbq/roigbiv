"""Tests for the v5 fov.resolved_config_uri column (optics auto-adapt memory)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from roigbiv.registry.store.base import FOVRecord
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore


def _store() -> SQLAlchemyStore:
    store = SQLAlchemyStore(dsn="sqlite://")
    store.ensure_schema()      # applies migrations through head (0005)
    return store


def _fov(**kw) -> FOVRecord:
    base = dict(
        fov_id=str(uuid.uuid4()),
        fingerprint_hash="b" * 64,
        animal_id="VI15", region="DS",
        mean_m_uri="file:///tmp/m.npy",
        centroid_table_uri="file:///tmp/c.npy",
        created_at=datetime.now(timezone.utc),
    )
    base.update(kw)
    return FOVRecord(**base)


def test_resolved_config_uri_defaults_none_and_roundtrips():
    store = _store()
    fid = str(uuid.uuid4())
    store.insert_fov(_fov(fov_id=fid, fingerprint_hash="c" * 64))
    assert store.get_fov(fid).resolved_config_uri is None      # nullable default

    fid2 = str(uuid.uuid4())
    store.insert_fov(_fov(fov_id=fid2, fingerprint_hash="d" * 64,
                          resolved_config_uri="file:///blobs/x/resolved_config.json"))
    assert store.get_fov(fid2).resolved_config_uri == \
        "file:///blobs/x/resolved_config.json"


def test_update_fov_resolved_config():
    store = _store()
    fid = str(uuid.uuid4())
    store.insert_fov(_fov(fov_id=fid))
    store.update_fov_resolved_config(fid, "file:///blobs/y/resolved_config.json")
    assert store.get_fov(fid).resolved_config_uri == \
        "file:///blobs/y/resolved_config.json"


def test_update_fov_resolved_config_missing_is_noop():
    store = _store()
    # No row → silent no-op (mirrors update_fov_embeddings behavior).
    store.update_fov_resolved_config(str(uuid.uuid4()), "file:///nope.json")
