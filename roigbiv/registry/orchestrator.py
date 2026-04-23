"""register_or_match — v3 top-level registry entry point.

Called by the pipeline (with ``--registry``), by the CLI, and by the
backfill command. Operates on a query :class:`SessionInput` (the mean
projection + merged mask from a freshly-processed session) and resolves
one of four outcomes:

    * **hash_match** — a FOV with the identical footprint-derived hash
      already exists; register a new session against it.
    * **auto_match** — best candidate's calibrated posterior is above the
      accept threshold; register a new session with ROICaT-derived
      cluster labels mapped back to existing global cell IDs.
    * **review** — best posterior falls in the review band; no DB writes,
      user resolves in the Streamlit Registry tab.
    * **new_fov** — otherwise; mint a new FOV with fresh global cell IDs.

The global cell ID assignment semantics follow Phase 1 design doc Q2(b):
existing ``global_cell_id`` values are immutable. When a query ROI's
cluster contains existing observations, the query joins the earliest-
created cell of that cluster; divergent global IDs within a single cluster
are logged as merge warnings but never rewritten.

A ``registry_match.json`` report is written to ``output_dir`` in every
branch, preserving the Streamlit tab's expected contract.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.registry.blob.base import BlobStore
from roigbiv.registry.calibration import CalibrationModel
from roigbiv.registry.fingerprint import (
    Fingerprint,
    compute_fingerprint,
    serialize_array,
)
from roigbiv.registry.filename import FilenameMetadata, parse_filename_metadata
from roigbiv.registry.match import (
    AUTO_ACCEPT_THRESHOLD,
    REVIEW_THRESHOLD,
    FOVMatchResult,
    match_fov,
)
from roigbiv.registry.roicat_adapter import (
    AdapterConfig,
    ClusterResult,
    SessionInput,
    load_session_input,
)
from roigbiv.registry.store.base import (
    CellRecord,
    FOVRecord,
    ObservationRecord,
    RegistryStore,
    SessionRecord,
)

log = logging.getLogger(__name__)


def register_or_match(
    *,
    fov_stem: str,
    query: SessionInput,
    output_dir: Path,
    store: RegistryStore,
    blob_store: BlobStore,
    session_date_override: Optional[date] = None,
    calibration: Optional[CalibrationModel] = None,
    adapter_config: Optional[AdapterConfig] = None,
    accept_threshold: float = AUTO_ACCEPT_THRESHOLD,
    review_threshold: float = REVIEW_THRESHOLD,
) -> dict:
    """Register a newly-processed session against the cross-session registry.

    Parameters
    ----------
    fov_stem : str
        Filename stem used to parse animal/region/date metadata.
    query : SessionInput
        The new session's mean projection + merged mask (in-memory).
    output_dir : Path
        Where the query session's output lives on disk. Stored on the
        session row so future re-clustering can reload the mask.
    store, blob_store : registry back-ends.
    session_date_override : Optional[date]
        Overrides the date parsed from ``fov_stem`` (useful when the
        filename date is wrong).
    calibration, adapter_config : tunables (both have sensible defaults).
    accept_threshold, review_threshold : decision banding.
    """
    store.ensure_schema()

    meta = parse_filename_metadata(fov_stem)
    if session_date_override is not None:
        meta = FilenameMetadata(
            animal_id=meta.animal_id,
            region=meta.region,
            session_date=session_date_override,
            fov_number=meta.fov_number,
        )
    session_date = meta.session_date or date.today()
    calibration = calibration or CalibrationModel()
    adapter_config = adapter_config or AdapterConfig()

    fp = compute_fingerprint(query.merged_masks, query.mean_m)

    # 0. Idempotency guard — if a session row already points at this exact
    #    output_dir and the FOV it's tied to still carries the same
    #    fingerprint hash we're about to register, this is a duplicate call
    #    (backfill re-registering what the per-TIF pass already wrote, or a
    #    user running `roigbiv-registry match` twice on the same directory).
    #    Short-circuit: no DB writes, return the cached report.
    existing_report = _load_idempotent_report(
        store=store, output_dir=output_dir, fingerprint_hash=fp.fingerprint_hash
    )
    if existing_report is not None:
        return existing_report

    # 1. Hash pre-filter — exact re-run shortcut.
    hit = store.get_fov_by_hash(fp.fingerprint_hash)
    if hit is not None:
        report = _register_hash_match(
            store=store,
            blob_store=blob_store,
            fov_record=hit,
            meta=meta,
            session_date=session_date,
            query=query,
            output_dir=output_dir,
            fp=fp,
        )
        return _write_report(output_dir, report)

    # 2. Find candidate FOVs (scoped by animal_id + region).
    candidates = store.find_candidates(meta.animal_id, meta.region)
    # Only v3 candidates can be matched via ROICaT — v1 / v2 FOVs have no
    # footprint blob + no session output dir contract we can rely on here.
    v3_candidates = [c for c in candidates if (c.fingerprint_version or 1) >= 3]
    if len(v3_candidates) != len(candidates):
        skipped = len(candidates) - len(v3_candidates)
        log.warning(
            "Skipping %d legacy (v1/v2) candidate FOV(s) for %s/%s — "
            "they predate the ROICaT-clusterable schema.",
            skipped,
            meta.animal_id,
            meta.region,
        )

    # 3. Iterate candidate FOVs; keep the highest-posterior result.
    best_record: Optional[FOVRecord] = None
    best_result: Optional[FOVMatchResult] = None
    best_sessions: Optional[list[SessionRecord]] = None
    best_inputs: Optional[list[SessionInput]] = None
    for cand in v3_candidates:
        cand_sessions = store.list_sessions(cand.fov_id)
        cand_inputs = _load_candidate_session_inputs(cand_sessions)
        if not cand_inputs:
            log.warning(
                "Candidate FOV %s has no reachable session outputs; skipping.",
                cand.fov_id,
            )
            continue
        try:
            result = match_fov(
                query=query,
                candidate_sessions=cand_inputs,
                calibration=calibration,
                adapter_config=adapter_config,
                accept_threshold=accept_threshold,
                review_threshold=review_threshold,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception(
                "ROICaT match_fov failed for candidate %s: %s", cand.fov_id, exc
            )
            continue
        if best_result is None or result.fov_posterior > best_result.fov_posterior:
            best_record = cand
            best_result = result
            best_sessions = cand_sessions
            best_inputs = cand_inputs

    # 4. Branch on the best result.
    if best_result is not None and best_result.decision == "auto_match":
        assert best_record is not None and best_sessions is not None
        report = _register_auto_match(
            store=store,
            blob_store=blob_store,
            fov_record=best_record,
            candidate_sessions=best_sessions,
            meta=meta,
            session_date=session_date,
            query=query,
            output_dir=output_dir,
            fp=fp,
            match_result=best_result,
        )
        return _write_report(output_dir, report)

    if best_result is not None and best_result.decision == "review":
        report = _review_report(
            fov_record=best_record,
            match_result=best_result,
            meta=meta,
            session_date=session_date,
            fp=fp,
            review_threshold=review_threshold,
            accept_threshold=accept_threshold,
        )
        return _write_report(output_dir, report)

    # 5. Otherwise mint a new FOV. Carry the best-candidate diagnostics forward
    #    so post-hoc inspection can see *why* we minted (vs. silent discard).
    report = _mint_new_fov(
        store=store,
        blob_store=blob_store,
        fingerprint=fp,
        meta=meta,
        session_date=session_date,
        query=query,
        output_dir=output_dir,
        best_candidate_fov_id=best_record.fov_id if best_record else None,
        best_match_result=best_result,
    )
    return _write_report(output_dir, report)


# ── Idempotency ────────────────────────────────────────────────────────────


def _load_idempotent_report(
    *,
    store: RegistryStore,
    output_dir: Path,
    fingerprint_hash: str,
) -> Optional[dict]:
    """Detect a duplicate ``register_or_match`` call and return a cached report.

    A call is considered a duplicate when:
      1. a session row already exists with the same ``output_dir``, AND
      2. the FOV it points at still carries the same fingerprint hash we're
         about to register (i.e. the outputs on disk haven't been overwritten
         with different masks since the row was written).

    When both hold we reuse the persisted ``registry_match.json`` in
    ``output_dir`` and stamp the decision as ``already_registered`` so callers
    can log the no-op. If the cached report can't be read, we synthesize a
    minimal one from the DB so downstream code keeps a consistent shape.
    """
    existing = store.get_session_by_output_dir(str(output_dir))
    if existing is None:
        return None
    fov = store.get_fov(existing.fov_id)
    if fov is None or fov.fingerprint_hash != fingerprint_hash:
        return None

    cached_path = Path(output_dir) / "registry_match.json"
    report: dict
    if cached_path.exists():
        try:
            report = json.loads(cached_path.read_text())
        except Exception:  # noqa: BLE001
            report = {}
    else:
        report = {}

    report.update({
        "decision": "already_registered",
        "fov_id": existing.fov_id,
        "session_id": existing.session_id,
        "fingerprint_hash": fingerprint_hash,
    })
    return report


# ── Candidate loading ──────────────────────────────────────────────────────


def _load_candidate_session_inputs(
    sessions: list[SessionRecord],
) -> list[SessionInput]:
    """Read each candidate session's merged_masks + mean_M from its output_dir."""
    inputs: list[SessionInput] = []
    for sess in sessions:
        output_dir = Path(sess.output_dir)
        if not output_dir.exists():
            log.warning(
                "Session %s output_dir %s missing — dropping from candidate bundle.",
                sess.session_id,
                output_dir,
            )
            continue
        try:
            inputs.append(load_session_input(output_dir, session_key=sess.session_id))
        except FileNotFoundError as exc:
            log.warning("Session %s: %s — dropping.", sess.session_id, exc)
    return inputs


# ── Branches ───────────────────────────────────────────────────────────────


def _register_hash_match(
    *,
    store: RegistryStore,
    blob_store: BlobStore,
    fov_record: FOVRecord,
    meta: FilenameMetadata,
    session_date: date,
    query: SessionInput,
    output_dir: Path,
    fp: Fingerprint,
) -> dict:
    """The exact-rerun shortcut — same fingerprint, write observations 1:1."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    # Hash match means label_ids + centroids are identical to the first
    # observation of this FOV, so we map each query label_id to the existing
    # global_cell_id (same first_local_label_id).
    existing_cells = store.list_cells(fov_record.fov_id)
    first_label_to_gid = {
        int(c.morphology_summary.get("first_local_label_id", -1)): c.global_cell_id
        for c in existing_cells
    }

    observations: list[ObservationRecord] = []
    cell_assignments: list[dict] = []
    matched = 0
    missing = 0
    new_cells: list[CellRecord] = []

    for label_id in fp.label_ids.tolist():
        gid = first_label_to_gid.get(int(label_id))
        if gid is None:
            # Shouldn't happen for a hash match, but guard anyway.
            gid = str(uuid.uuid4())
            new_cells.append(CellRecord(
                global_cell_id=gid,
                fov_id=fov_record.fov_id,
                first_seen_session_id=session_id,
                morphology_summary={"first_local_label_id": int(label_id)},
            ))
            kind = "new"
            missing += 1
        else:
            kind = "matched"
            matched += 1
        observations.append(ObservationRecord(
            observation_id=str(uuid.uuid4()),
            global_cell_id=gid,
            session_id=session_id,
            local_label_id=int(label_id),
            match_score=1.0,
            cluster_label=None,
        ))
        cell_assignments.append({
            "local_label_id": int(label_id),
            "global_cell_id": gid,
            "match_kind": kind,
        })

    store.insert_session(SessionRecord(
        session_id=session_id,
        fov_id=fov_record.fov_id,
        session_date=session_date,
        output_dir=str(output_dir),
        fov_sim=1.0,
        fov_posterior=1.0,
        n_matched=matched,
        n_new=0,
        n_missing=0,
        created_at=now,
    ))
    for cell in new_cells:
        store.insert_cell(cell)
    store.insert_observations(observations)
    store.update_fov_latest_session(fov_record.fov_id, session_date)

    return {
        "decision": "hash_match",
        "fov_id": fov_record.fov_id,
        "session_id": session_id,
        "animal_id": meta.animal_id,
        "region": meta.region,
        "session_date": session_date.isoformat(),
        "fingerprint_hash": fov_record.fingerprint_hash,
        "fingerprint_version": fp.fingerprint_version,
        "fov_sim": 1.0,
        "fov_posterior": 1.0,
        "n_matched": matched,
        "n_new": 0,
        "n_missing": missing,
        "cell_assignments": cell_assignments,
    }


def _register_auto_match(
    *,
    store: RegistryStore,
    blob_store: BlobStore,
    fov_record: FOVRecord,
    candidate_sessions: list[SessionRecord],
    meta: FilenameMetadata,
    session_date: date,
    query: SessionInput,
    output_dir: Path,
    fp: Fingerprint,
    match_result: FOVMatchResult,
) -> dict:
    """Write a new session mapped via ROICaT cluster labels → global cell IDs."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    cr = match_result.cluster_result
    q_idx = match_result.query_session_idx

    # 1. Slice query labels out of the concatenated cluster_result.
    per_session_counts = cr.per_session_roi_count
    offset = int(sum(per_session_counts[:q_idx]))
    query_n = int(per_session_counts[q_idx])
    query_cluster_labels = cr.labels[offset : offset + query_n].astype(np.int32)
    query_label_ids = cr.per_session_label_ids[q_idx]  # original label_ids in the mask

    # 2. Build (session_id, local_label_id) → cluster_label map for candidate sessions
    #    so we can resolve cluster_label → existing global_cell_id via the DB.
    candidate_cluster_by_session: dict[str, np.ndarray] = {}
    candidate_label_ids_by_session: dict[str, np.ndarray] = {}
    cursor = 0
    for sess_idx, sess_rec in enumerate(candidate_sessions):
        n = int(per_session_counts[sess_idx])
        candidate_cluster_by_session[sess_rec.session_id] = (
            cr.labels[cursor : cursor + n].astype(np.int32)
        )
        candidate_label_ids_by_session[sess_rec.session_id] = cr.per_session_label_ids[sess_idx]
        cursor += n

    # 3. For each cluster_label seen in the query, find the oldest global_cell_id
    #    among any observation whose (session_id, local_label_id) sits in that cluster.
    existing_cells = store.list_cells(fov_record.fov_id)
    cell_by_gid = {c.global_cell_id: c for c in existing_cells}
    cluster_to_gid: dict[int, str] = {}

    for sess_rec in candidate_sessions:
        cluster_labels = candidate_cluster_by_session.get(sess_rec.session_id)
        label_ids = candidate_label_ids_by_session.get(sess_rec.session_id)
        if cluster_labels is None or label_ids is None:
            continue
        obs = store.list_observations_for_session(sess_rec.session_id)
        obs_by_label = {int(o.local_label_id): o for o in obs}
        for idx, label_id in enumerate(label_ids.tolist()):
            clabel = int(cluster_labels[idx])
            if clabel == -1:
                continue
            o = obs_by_label.get(int(label_id))
            if o is None:
                continue
            existing = cluster_to_gid.get(clabel)
            if existing is None:
                cluster_to_gid[clabel] = o.global_cell_id
            elif existing != o.global_cell_id:
                # Preserve the earliest-created cell's ID; log the merge.
                e_cell = cell_by_gid.get(existing)
                o_cell = cell_by_gid.get(o.global_cell_id)
                if (
                    e_cell is not None
                    and o_cell is not None
                    and (o_cell.first_seen_session_id or "") < (e_cell.first_seen_session_id or "")
                ):
                    cluster_to_gid[clabel] = o.global_cell_id
                log.warning(
                    "Cluster %d in FOV %s spans multiple existing global_cell_ids "
                    "(%s vs %s); keeping earliest-created.",
                    clabel,
                    fov_record.fov_id,
                    existing,
                    o.global_cell_id,
                )

    # 4. Build observations for the query session.
    observations: list[ObservationRecord] = []
    cell_assignments: list[dict] = []
    new_cells: list[CellRecord] = []
    n_matched = 0
    n_new = 0

    for idx, label_id in enumerate(query_label_ids.tolist()):
        clabel = int(query_cluster_labels[idx])
        gid: Optional[str] = None
        match_kind = "new"
        if clabel != -1:
            gid = cluster_to_gid.get(clabel)
            if gid is not None:
                match_kind = "matched"
        if gid is None:
            gid = str(uuid.uuid4())
            new_cells.append(CellRecord(
                global_cell_id=gid,
                fov_id=fov_record.fov_id,
                first_seen_session_id=session_id,
                morphology_summary={
                    "first_local_label_id": int(label_id),
                    "area": int(fp.areas[idx]) if idx < len(fp.areas) else 0,
                    "first_cluster_label": clabel,
                },
            ))
            n_new += 1
            # Propagate the new global_cell_id so other ROIs in the same cluster
            # (shouldn't happen — ROICaT enforces one ROI per session per cluster
            # — but defensive).
            if clabel != -1:
                cluster_to_gid[clabel] = gid
        else:
            n_matched += 1
        observations.append(ObservationRecord(
            observation_id=str(uuid.uuid4()),
            global_cell_id=gid,
            session_id=session_id,
            local_label_id=int(label_id),
            match_score=float(match_result.fov_posterior),
            cluster_label=int(clabel) if clabel != -1 else None,
        ))
        cell_assignments.append({
            "local_label_id": int(label_id),
            "global_cell_id": gid,
            "match_kind": match_kind,
            "cluster_label": int(clabel) if clabel != -1 else None,
        })

    # 5. Count missing candidate cells (clusters present in candidates but no
    #    query member). Cheap: walk unique candidate cluster labels and check.
    missing_payload: list[dict] = []
    query_clusters = set(int(c) for c in query_cluster_labels.tolist() if c != -1)
    seen_missing_gids: set[str] = set()
    for sess_rec in candidate_sessions:
        cluster_labels = candidate_cluster_by_session.get(sess_rec.session_id)
        label_ids = candidate_label_ids_by_session.get(sess_rec.session_id)
        if cluster_labels is None or label_ids is None:
            continue
        for idx, label_id in enumerate(label_ids.tolist()):
            clabel = int(cluster_labels[idx])
            if clabel == -1 or clabel in query_clusters:
                continue
            gid = cluster_to_gid.get(clabel)
            if gid is None or gid in seen_missing_gids:
                continue
            seen_missing_gids.add(gid)
            missing_payload.append({
                "global_cell_id": gid,
                "candidate_local_label_id": int(label_id),
                "candidate_session_id": sess_rec.session_id,
            })
    n_missing = len(missing_payload)

    # 6. Persist the query session's cluster labels as a blob.
    cluster_labels_uri = blob_store.put(
        f"{fov_record.fov_id}/sessions/{session_id}/cluster_labels.npy",
        serialize_array(query_cluster_labels),
    )

    store.insert_session(SessionRecord(
        session_id=session_id,
        fov_id=fov_record.fov_id,
        session_date=session_date,
        output_dir=str(output_dir),
        fov_sim=float(match_result.fov_posterior),
        fov_posterior=float(match_result.fov_posterior),
        n_matched=n_matched,
        n_new=n_new,
        n_missing=n_missing,
        created_at=now,
        cluster_labels_uri=cluster_labels_uri,
    ))
    for cell in new_cells:
        store.insert_cell(cell)
    store.insert_observations(observations)
    store.update_fov_latest_session(fov_record.fov_id, session_date)

    features = match_result.features
    return {
        "decision": "auto_match",
        "fov_id": fov_record.fov_id,
        "session_id": session_id,
        "animal_id": meta.animal_id,
        "region": meta.region,
        "session_date": session_date.isoformat(),
        "fingerprint_hash": fov_record.fingerprint_hash,
        "fingerprint_version": fp.fingerprint_version,
        "fov_sim": float(match_result.fov_posterior),
        "fov_posterior": float(match_result.fov_posterior),
        "alignment_method": cr.alignment_method,
        "alignment_inlier_rate": float(cr.alignment_inlier_rate),
        "n_shared_clusters": int(features.n_shared_clusters),
        "fraction_query_clustered": float(features.fraction_query_clustered),
        "mean_cluster_cohesion": float(features.mean_cluster_cohesion),
        "n_matched": n_matched,
        "n_new": n_new,
        "n_missing": n_missing,
        "cluster_labels_uri": cluster_labels_uri,
        "cell_assignments": cell_assignments,
        "missing_cells": missing_payload,
    }


def _review_report(
    *,
    fov_record: Optional[FOVRecord],
    match_result: FOVMatchResult,
    meta: FilenameMetadata,
    session_date: date,
    fp: Fingerprint,
    review_threshold: float,
    accept_threshold: float,
) -> dict:
    """Review-band payload — no DB writes; user resolves in Streamlit."""
    cr = match_result.cluster_result
    features = match_result.features
    return {
        "decision": "review",
        "fov_id": None,
        "candidate_fov_id": fov_record.fov_id if fov_record else None,
        "fov_posterior": float(match_result.fov_posterior),
        "fov_sim": float(match_result.fov_posterior),
        "alignment_method": cr.alignment_method,
        "alignment_inlier_rate": float(cr.alignment_inlier_rate),
        "n_shared_clusters": int(features.n_shared_clusters),
        "fraction_query_clustered": float(features.fraction_query_clustered),
        "mean_cluster_cohesion": float(features.mean_cluster_cohesion),
        "animal_id": meta.animal_id,
        "region": meta.region,
        "session_date": session_date.isoformat(),
        "fingerprint_hash": fp.fingerprint_hash,
        "fingerprint_version": fp.fingerprint_version,
        "message": (
            f"posterior={match_result.fov_posterior:.3f} in review band "
            f"[{review_threshold}, {accept_threshold}); "
            "no session written. Resolve in the Streamlit Registry tab."
        ),
    }


def _mint_new_fov(
    *,
    store: RegistryStore,
    blob_store: BlobStore,
    fingerprint: Fingerprint,
    meta: FilenameMetadata,
    session_date: date,
    query: SessionInput,
    output_dir: Path,
    best_candidate_fov_id: Optional[str] = None,
    best_match_result: Optional[FOVMatchResult] = None,
) -> dict:
    """Create a new FOV + session + cells keyed to fresh global IDs."""
    fov_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    merged_masks_uri = blob_store.put(
        f"{fov_id}/merged_masks.npy", fingerprint.merged_masks_blob
    )
    mean_m_uri = blob_store.put(f"{fov_id}/mean_M.npy", fingerprint.mean_m_blob)
    centroids_uri = blob_store.put(f"{fov_id}/centroids.npy", fingerprint.centroids_blob)

    # Single-session FOV: each ROI is its own cluster label.
    query_cluster_labels = np.arange(fingerprint.label_ids.size, dtype=np.int32)
    cluster_labels_uri = blob_store.put(
        f"{fov_id}/sessions/{session_id}/cluster_labels.npy",
        serialize_array(query_cluster_labels),
    )

    store.insert_fov(FOVRecord(
        fov_id=fov_id,
        fingerprint_hash=fingerprint.fingerprint_hash,
        animal_id=meta.animal_id,
        region=meta.region,
        mean_m_uri=mean_m_uri,
        centroid_table_uri=centroids_uri,
        created_at=now,
        latest_session_date=session_date,
        fingerprint_version=fingerprint.fingerprint_version,
        # v3 does not populate the legacy fov/roi embedding URIs. We stash the
        # merged_masks blob URI under roi_embeddings_uri for round-trip
        # convenience — it is opaque to the legacy code path.
        fov_embedding_uri=None,
        roi_embeddings_uri=merged_masks_uri,
    ))
    store.insert_session(SessionRecord(
        session_id=session_id,
        fov_id=fov_id,
        session_date=session_date,
        output_dir=str(output_dir),
        fov_sim=None,
        fov_posterior=None,
        n_matched=0,
        n_new=int(fingerprint.label_ids.size),
        n_missing=0,
        created_at=now,
        cluster_labels_uri=cluster_labels_uri,
    ))

    global_ids: dict[int, str] = {}
    observations: list[ObservationRecord] = []
    for idx, label_id in enumerate(fingerprint.label_ids.tolist()):
        gid = str(uuid.uuid4())
        global_ids[int(label_id)] = gid
        store.insert_cell(CellRecord(
            global_cell_id=gid,
            fov_id=fov_id,
            first_seen_session_id=session_id,
            morphology_summary={
                "first_local_label_id": int(label_id),
                "area": int(fingerprint.areas[idx]) if idx < len(fingerprint.areas) else 0,
                "centroid_y": int(fingerprint.centroids[idx, 0]),
                "centroid_x": int(fingerprint.centroids[idx, 1]),
                "first_cluster_label": int(query_cluster_labels[idx]),
            },
        ))
        observations.append(ObservationRecord(
            observation_id=str(uuid.uuid4()),
            global_cell_id=gid,
            session_id=session_id,
            local_label_id=int(label_id),
            match_score=None,
            cluster_label=int(query_cluster_labels[idx]),
        ))
    store.insert_observations(observations)

    report: dict = {
        "decision": "new_fov",
        "fov_id": fov_id,
        "session_id": session_id,
        "animal_id": meta.animal_id,
        "region": meta.region,
        "session_date": session_date.isoformat(),
        "fingerprint_hash": fingerprint.fingerprint_hash,
        "fingerprint_version": fingerprint.fingerprint_version,
        "n_new_cells": int(fingerprint.label_ids.size),
        "best_candidate_fov_id": best_candidate_fov_id,
        "best_candidate_posterior": (
            best_match_result.fov_posterior if best_match_result else None
        ),
        "cluster_labels_uri": cluster_labels_uri,
        "cell_assignments": [
            {
                "local_label_id": int(label_id),
                "global_cell_id": global_ids[int(label_id)],
                "match_kind": "new",
                "cluster_label": int(query_cluster_labels[i]),
            }
            for i, label_id in enumerate(fingerprint.label_ids.tolist())
        ],
    }
    if best_match_result is not None:
        cr = best_match_result.cluster_result
        feats = best_match_result.features
        report["best_candidate_decision"] = best_match_result.decision
        report["best_candidate_alignment_method"] = cr.alignment_method
        report["best_candidate_alignment_inlier_rate"] = float(cr.alignment_inlier_rate)
        report["best_candidate_n_shared_clusters"] = int(feats.n_shared_clusters)
        report["best_candidate_fraction_query_clustered"] = float(
            feats.fraction_query_clustered
        )
        report["best_candidate_mean_cluster_cohesion"] = float(
            feats.mean_cluster_cohesion
        )
    return report


def _write_report(output_dir: Path, report: dict) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "registry_match.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return report
