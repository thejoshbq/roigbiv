"""Tests for the v3 register_or_match orchestrator.

These tests exercise the code paths that do NOT require running ROICaT:
  * new_fov (empty registry → mint a new FOV)
  * hash_match (identical footprint hash → session re-registration)

Cross-session clustering (auto_match / review / reject) flows through the
ROICaT adapter and is covered end-to-end by Phase 4's three-session test,
where real-sized data is available. The synthetic inputs here are too
small for ROICaT's similarity-crossover inference and would need fixture
data that's beyond what this suite is meant to verify.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from roigbiv.registry.blob.local import LocalBlobStore
from roigbiv.registry.orchestrator import register_or_match
from roigbiv.registry.roicat_adapter import SessionInput
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore


def _mask(centers: list[tuple[int, int]], H: int = 64, W: int = 64) -> np.ndarray:
    out = np.zeros((H, W), dtype=np.uint16)
    yy, xx = np.ogrid[:H, :W]
    for lbl, (cy, cx) in enumerate(centers, start=1):
        out[((yy - cy) ** 2 + (xx - cx) ** 2) <= 9] = lbl
    return out


def _mean(H: int = 64, W: int = 64, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((H, W)).astype(np.float32)


def _session(session_key: str, centers: list[tuple[int, int]]) -> SessionInput:
    return SessionInput(session_key=session_key, mean_m=_mean(), merged_masks=_mask(centers))


def _build_backends(tmp_path: Path):
    store = SQLAlchemyStore(dsn="sqlite://")
    store.ensure_schema()
    blob = LocalBlobStore(root=tmp_path / "blobs")
    return store, blob


def test_first_run_mints_new_fov(tmp_path: Path):
    store, blob = _build_backends(tmp_path)
    query = _session("q", [(16, 16), (32, 32), (48, 48)])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    report = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH",
        query=query,
        output_dir=out_dir,
        store=store,
        blob_store=blob,
    )
    assert report["decision"] == "new_fov"
    fovs = store.list_fovs()
    assert len(fovs) == 1
    fov = fovs[0]
    # Phase-1 filename-parser fix: animal_id is "T1", not the session-condition tail.
    assert fov.animal_id == "T1"
    assert fov.region == "PrL-NAc"
    assert fov.latest_session_date == date(2022, 12, 9)
    assert fov.fingerprint_version == 3
    assert len(store.list_cells(fov.fov_id)) == 3
    assert report["n_new_cells"] == 3


def test_same_input_hash_matches_without_reclustering(tmp_path: Path):
    """Identical masks → hash shortcut — no ROICaT invocation needed."""
    store, blob = _build_backends(tmp_path)
    centers = [(16, 16), (32, 32), (48, 48)]
    query1 = _session("q1", centers)

    out1 = tmp_path / "out1"
    out1.mkdir()
    r1 = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1",
        query=query1,
        output_dir=out1,
        store=store,
        blob_store=blob,
    )
    # Second session with identical masks → fingerprint hash collides.
    query2 = _session("q2", centers)
    out2 = tmp_path / "out2"
    out2.mkdir()
    r2 = register_or_match(
        fov_stem="T1_221215_PrL-NAc-G6-5M_HI-D1_FOV1",
        query=query2,
        output_dir=out2,
        store=store,
        blob_store=blob,
    )
    assert r2["decision"] == "hash_match"
    assert r2["fov_id"] == r1["fov_id"]
    assert len(store.list_fovs()) == 1
    sessions = store.list_sessions(r1["fov_id"])
    assert len(sessions) == 2
    # Session row for r2 records posterior = 1.0 for the hash-match shortcut.
    assert sessions[-1].fov_posterior == 1.0


def test_unrelated_fov_mints_new_because_candidates_mismatch_animal(tmp_path: Path):
    """Different animal_id → no candidate retrieval → new FOV without ROICaT."""
    store, blob = _build_backends(tmp_path)
    out1 = tmp_path / "out1"
    out1.mkdir()
    register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1",
        query=_session("q1", [(10, 10), (20, 40), (40, 20)]),
        output_dir=out1,
        store=store,
        blob_store=blob,
    )
    out2 = tmp_path / "out2"
    out2.mkdir()
    r2 = register_or_match(
        fov_stem="T2_221210_PrL-NAc-G6-5M_HI-D1_FOV1",  # different animal
        query=_session("q2", [(30, 30), (45, 45), (15, 50)]),
        output_dir=out2,
        store=store,
        blob_store=blob,
    )
    assert r2["decision"] == "new_fov"
    assert r2["fov_id"] != store.list_fovs()[0].fov_id or len(store.list_fovs()) == 2
    assert len(store.list_fovs()) == 2
