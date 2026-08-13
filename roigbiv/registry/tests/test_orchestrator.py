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


def test_new_fov_persists_resolved_config_blob(tmp_path: Path):
    import json

    store, blob = _build_backends(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg_payload = {"profile": "prism", "auto_scale": True, "diameter": 56,
                   "auto_adapt": {"scale_ok": True, "n_somata": 14}}
    report = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH",
        query=_session("q", [(16, 16), (32, 32), (48, 48)]),
        output_dir=out_dir,
        store=store,
        blob_store=blob,
        resolved_config=cfg_payload,
    )
    assert report["decision"] == "new_fov"
    fov = store.get_fov(report["fov_id"])
    assert fov.resolved_config_uri is not None
    restored = json.loads(blob.get(fov.resolved_config_uri).decode())
    assert restored["profile"] == "prism"
    assert restored["auto_adapt"]["scale_ok"] is True


def test_new_fov_without_resolved_config_leaves_column_null(tmp_path: Path):
    store, blob = _build_backends(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    report = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH",
        query=_session("q", [(16, 16), (32, 32), (48, 48)]),
        output_dir=out_dir, store=store, blob_store=blob,
    )
    assert store.get_fov(report["fov_id"]).resolved_config_uri is None


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


def test_repeat_call_on_same_output_dir_is_idempotent(tmp_path: Path):
    """Calling register_or_match twice with the same output_dir writes one session row.

    This guards the workspace flow: the pipeline's per-TIF pass registers a
    session, then ``_safety_backfill`` sweeps the same output dirs — without
    idempotency that would double every session.
    """
    store, blob = _build_backends(tmp_path)
    query = _session("q", [(16, 16), (32, 32), (48, 48)])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    r1 = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1",
        query=query,
        output_dir=out_dir,
        store=store,
        blob_store=blob,
    )
    assert r1["decision"] == "new_fov"
    fov_id = r1["fov_id"]
    assert len(store.list_sessions(fov_id)) == 1

    r2 = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1",
        query=query,
        output_dir=out_dir,
        store=store,
        blob_store=blob,
    )
    assert r2["decision"] == "already_registered"
    assert r2["fov_id"] == fov_id
    assert r2["session_id"] == r1["session_id"]
    assert len(store.list_sessions(fov_id)) == 1
    assert len(store.list_fovs()) == 1


def test_override_bypasses_idempotency_and_replaces(tmp_path: Path):
    """override=True re-registers a same-masks re-run instead of no-op'ing.

    Without override the second call short-circuits to ``already_registered``
    (see the test above). With override the prior session is superseded (and
    its now-orphaned FOV dropped), so the run re-mints — leaving exactly one
    FOV + one session for that output_dir rather than accumulating.
    """
    store, blob = _build_backends(tmp_path)
    query = _session("q", [(16, 16), (32, 32), (48, 48)])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    stem = "T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1"
    r1 = register_or_match(fov_stem=stem, query=query, output_dir=out_dir,
                           store=store, blob_store=blob)
    assert r1["decision"] == "new_fov"

    r2 = register_or_match(
        fov_stem=stem,
        query=_session("q", [(16, 16), (32, 32), (48, 48)]),
        output_dir=out_dir, store=store, blob_store=blob,
        override=True)
    # Not the idempotent short-circuit — a fresh registration replaced it.
    assert r2["decision"] != "already_registered"
    assert len(store.list_fovs()) == 1
    assert len(store.list_sessions(r2["fov_id"])) == 1


def test_override_changed_masks_drops_orphaned_prior_fov(tmp_path: Path):
    """A changed-mask re-run with override leaves exactly one FOV/session.

    Uses a different ``animal_id`` on the second run so candidate retrieval is
    empty (no ROICaT) — the point under test is supersede + orphan cleanup,
    not cross-session matching.
    """
    store, blob = _build_backends(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    r1 = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1",
        query=_session("q1", [(10, 10), (20, 40), (40, 20)]),
        output_dir=out_dir, store=store, blob_store=blob)
    assert r1["decision"] == "new_fov"
    old_fov_id = r1["fov_id"]

    r2 = register_or_match(
        fov_stem="T2_221210_PrL-NAc-G6-5M_HI-D1_FOV1",   # different animal
        query=_session("q2", [(30, 30), (45, 45), (15, 50)]),
        output_dir=out_dir, store=store, blob_store=blob,
        override=True)
    assert r2["decision"] == "new_fov"
    # Old FOV was orphaned by the supersede (its only session was at out_dir)
    # and dropped; only the fresh one remains.
    assert store.get_fov(old_fov_id) is None
    assert len(store.list_fovs()) == 1
    assert store.list_fovs()[0].fov_id == r2["fov_id"]
    assert len(store.list_sessions(r2["fov_id"])) == 1


def test_a_failed_candidate_comparison_is_recorded_on_the_report(tmp_path: Path):
    """A crashed matcher and a genuinely-new FOV both decide "new_fov".

    Nothing downstream can tell them apart unless the failure is carried on
    the report — the logger this used to go to is not configured by any
    entry point, so the crash was invisible in every artifact.
    """
    from unittest.mock import patch

    store, blob = _build_backends(tmp_path)
    stem = "T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH"

    first = tmp_path / "s1"
    first.mkdir()
    register_or_match(fov_stem=stem, query=_session("q1", [(16, 16), (32, 32)]),
                      output_dir=first, store=store, blob_store=blob)

    # Second session, same animal/region → the first FOV is a candidate. Its
    # comparison raises the way ROICaT's clustering does on unalignable FOVs.
    second = tmp_path / "s2"
    second.mkdir()
    with patch("roigbiv.registry.orchestrator._load_candidate_session_inputs",
               return_value=[_session("c1", [(16, 16), (32, 32)])]), \
         patch("roigbiv.registry.orchestrator.match_fov",
               side_effect=ValueError("zero-size array to reduction "
                                      "operation fmin which has no identity")):
        report = register_or_match(
            fov_stem=stem, query=_session("q2", [(20, 20), (40, 40)]),
            output_dir=second, store=store, blob_store=blob)

    assert report["decision"] == "new_fov"
    assert len(report["match_errors"]) == 1
    err = report["match_errors"][0]
    assert "zero-size array" in err["error"]
    assert err["candidate_fov_id"] == store.list_fovs()[0].fov_id


def test_a_clean_new_fov_records_no_match_errors(tmp_path: Path):
    store, blob = _build_backends(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    report = register_or_match(
        fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH",
        query=_session("q", [(16, 16), (32, 32)]),
        output_dir=out_dir, store=store, blob_store=blob)
    assert report["match_errors"] == []


# ── Re-registering the same output directory ───────────────────────────────


def _auto_match(candidate_counts: list[int], query_count: int) -> object:
    """A ``match_fov`` result that pairs each query ROI with its candidate twin.

    Stands in for ROICaT so the auto_match write path can be tested without
    real-sized fixture data. Cluster label ``k`` holds ROI ``k`` of every
    session, which is the shape a successful three-session match produces.
    """
    from roigbiv.registry.calibration import FOVFeatures
    from roigbiv.registry.match import FOVMatchResult
    from roigbiv.registry.roicat_adapter import ClusterResult

    counts = list(candidate_counts) + [query_count]
    labels = np.concatenate([np.arange(n, dtype=np.int32) for n in counts])
    session_bool = np.zeros((int(sum(counts)), len(counts)), dtype=bool)
    offset = 0
    for j, n in enumerate(counts):
        session_bool[offset:offset + n, j] = True
        offset += n
    return FOVMatchResult(
        decision="auto_match",
        fov_posterior=0.95,
        features=FOVFeatures(n_shared_clusters=min(counts),
                             fraction_query_clustered=1.0,
                             alignment_quality=1.0,
                             mean_cluster_cohesion=0.8),
        cluster_result=ClusterResult(
            labels=labels,
            session_bool=session_bool,
            per_session_label_ids=[np.arange(1, n + 1, dtype=np.int64)
                                   for n in counts],
            per_session_roi_count=counts,
            quality_metrics={},
            alignment_method="stub",
            alignment_inlier_rate=1.0,
            fov_height=64,
            fov_width=64,
        ),
        query_session_idx=len(candidate_counts),
    )


def _orphaned_observations(store) -> list[str]:
    """Observation ids whose ``session_id`` resolves to no session row."""
    live = {
        s.session_id
        for fov in store.list_fovs()
        for s in store.list_sessions(fov.fov_id)
    }
    orphans = []
    for fov in store.list_fovs():
        for cell in store.list_cells(fov.fov_id):
            for obs in store.list_observations_for_cell(cell.global_cell_id):
                if obs.session_id not in live:
                    orphans.append(obs.observation_id)
    return orphans


def test_re_registering_an_output_dir_replaces_its_session(tmp_path: Path):
    """Re-running tracking over a directory must not strand its observations.

    ``(fov_id, output_dir)`` is unique, so the second registration cannot get
    a row of its own. It used to mint a fresh ``session_id`` anyway, write
    every observation against it, and lose the row at insert time — leaving
    observations referencing a session that does not exist and a session row
    still advertising the *previous* run's counts.
    """
    from unittest.mock import patch

    store, blob = _build_backends(tmp_path)
    stem = "T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH"

    first = tmp_path / "s1"
    first.mkdir()
    register_or_match(fov_stem=stem, query=_session("q1", [(16, 16), (32, 32)]),
                      output_dir=first, store=store, blob_store=blob)
    fov_id = store.list_fovs()[0].fov_id

    second = tmp_path / "s2"
    second.mkdir()
    candidate = _session("c1", [(16, 16), (32, 32)])

    with patch("roigbiv.registry.orchestrator._load_candidate_session_inputs",
               return_value=[candidate]), \
         patch("roigbiv.registry.orchestrator.match_fov",
               return_value=_auto_match([2], 2)):
        first_report = register_or_match(
            fov_stem=stem, query=_session("q2", [(17, 17), (33, 33)]),
            output_dir=second, store=store, blob_store=blob)
    assert first_report["decision"] == "auto_match"
    assert first_report["n_matched"] == 2

    # Re-run over the same directory with different masks — a re-stamp at a
    # new radius does exactly this, and the changed fingerprint means the
    # idempotency guard does not fire.
    with patch("roigbiv.registry.orchestrator._load_candidate_session_inputs",
               return_value=[candidate]), \
         patch("roigbiv.registry.orchestrator.match_fov",
               return_value=_auto_match([2], 3)):
        second_report = register_or_match(
            fov_stem=stem,
            query=_session("q2", [(17, 17), (33, 33), (48, 48)]),
            output_dir=second, store=store, blob_store=blob)

    assert second_report["decision"] == "auto_match"
    assert _orphaned_observations(store) == []

    sessions = [s for s in store.list_sessions(fov_id)
                if s.output_dir == str(second)]
    assert len(sessions) == 1
    row = sessions[0]
    assert row.session_id == second_report["session_id"]
    # Counts describe the re-run (3 ROIs: 2 matched + 1 new), not the first.
    assert (row.n_matched, row.n_new) == (2, 1)
    assert len(store.list_observations_for_session(row.session_id)) == 3


# ── Forced FOV grouping ────────────────────────────────────────────────────


def test_force_fov_id_registers_into_that_fov_despite_a_rejecting_posterior(
    tmp_path: Path,
):
    """A human-confirmed session order outranks the matcher's opinion.

    The posterior is derived from the cell clustering it gates, so a matcher
    that cannot cluster these ROIs used to split the FOV — turning a cell
    matching failure into a grouping failure and leaving the sessions
    incomparable. Forced grouping keeps them together and reports the
    disagreement instead.
    """
    from unittest.mock import patch

    store, blob = _build_backends(tmp_path)
    stem = "T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH"

    first = tmp_path / "s1"
    first.mkdir()
    r1 = register_or_match(fov_stem=stem,
                           query=_session("q1", [(16, 16), (32, 32)]),
                           output_dir=first, store=store, blob_store=blob)
    fov_id = r1["fov_id"]

    rejecting = _auto_match([2], 2)
    rejecting.decision = "reject"
    rejecting.fov_posterior = 0.02

    second = tmp_path / "s2"
    second.mkdir()
    with patch("roigbiv.registry.orchestrator._load_candidate_session_inputs",
               return_value=[_session("c1", [(16, 16), (32, 32)])]), \
         patch("roigbiv.registry.orchestrator.match_fov",
               return_value=rejecting):
        report = register_or_match(
            fov_stem=stem, query=_session("q2", [(17, 17), (33, 33)]),
            output_dir=second, store=store, blob_store=blob,
            force_fov_id=fov_id)

    assert report["decision"] == "forced_fov"
    assert report["fov_id"] == fov_id
    assert len(store.list_fovs()) == 1
    assert len(store.list_sessions(fov_id)) == 2
    # Cells were still assigned from the clustering, not abandoned.
    assert report["n_matched"] == 2
    # ...and the matcher's disagreement is on the record, not swallowed.
    assert "0.020" in report["grouping_warning"]


def test_force_fov_id_still_registers_when_clustering_raises(tmp_path: Path):
    """A crashed matcher must not drop an ordered session from its own FOV."""
    from unittest.mock import patch

    store, blob = _build_backends(tmp_path)
    stem = "T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH"

    first = tmp_path / "s1"
    first.mkdir()
    r1 = register_or_match(fov_stem=stem,
                           query=_session("q1", [(16, 16), (32, 32)]),
                           output_dir=first, store=store, blob_store=blob)
    fov_id = r1["fov_id"]

    second = tmp_path / "s2"
    second.mkdir()
    with patch("roigbiv.registry.orchestrator._load_candidate_session_inputs",
               return_value=[_session("c1", [(16, 16), (32, 32)])]), \
         patch("roigbiv.registry.orchestrator.match_fov",
               side_effect=RuntimeError("ROICaT could not infer a distance cutoff")):
        report = register_or_match(
            fov_stem=stem, query=_session("q2", [(17, 17), (33, 33)]),
            output_dir=second, store=store, blob_store=blob,
            force_fov_id=fov_id)

    assert report["decision"] == "forced_fov"
    assert len(store.list_sessions(fov_id)) == 2
    assert report["n_matched"] == 0
    assert report["n_new"] == 2
    assert "distance cutoff" in report["match_errors"][0]["error"]


def test_force_fov_id_rejects_an_unknown_fov(tmp_path: Path):
    import pytest

    store, blob = _build_backends(tmp_path)
    out = tmp_path / "s1"
    out.mkdir()
    with pytest.raises(ValueError, match="not in the registry"):
        register_or_match(fov_stem="T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH",
                          query=_session("q", [(16, 16)]),
                          output_dir=out, store=store, blob_store=blob,
                          force_fov_id="no-such-fov")
