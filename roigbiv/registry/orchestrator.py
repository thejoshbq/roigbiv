"""register_or_match — the top-level registry entry point.

Called by the pipeline (with --registry) and by the backfill command. Takes
the mean_M summary image + a list of CellFeature per ROI, plus metadata parsed
from the filename stem, and resolves one of three outcomes:

    * hash pre-filter hit → register a new session against the known FOV.
    * candidate matched above AUTO_ACCEPT_THRESHOLD → same.
    * candidate in review band (0.60 \u2264 sim < 0.85) → status "review";
      observations NOT written; user resolves in Streamlit.
    * otherwise → mint a new FOV with fresh cells.

A registry_match.json report is written to the output dir either way.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from roigbiv.registry.blob.base import BlobStore
from roigbiv.registry.fingerprint import (
    CellFeature,
    Fingerprint,
    compute_fingerprint,
    deserialize_cells,
    deserialize_mean_m,
)
from roigbiv.registry.filename import FilenameMetadata, parse_filename_metadata
from roigbiv.registry.match import (
    AUTO_ACCEPT_THRESHOLD,
    REVIEW_THRESHOLD,
    CellMatchResult,
    FOVMatchResult,
    match_cells,
    match_fov,
)
from roigbiv.registry.store.base import (
    CellRecord,
    FOVRecord,
    ObservationRecord,
    RegistryStore,
    SessionRecord,
)


def register_or_match(
    *,
    fov_stem: str,
    mean_m: np.ndarray,
    cells: list[CellFeature],
    output_dir: Path,
    store: RegistryStore,
    blob_store: BlobStore,
    session_date_override: Optional[date] = None,
) -> dict:
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

    fp = compute_fingerprint(mean_m, cells)

    # Hash pre-filter: exact re-run shortcut.
    hit = store.get_fov_by_hash(fp.fingerprint_hash)
    if hit is not None:
        report = _register_session_on_existing(
            store=store,
            blob_store=blob_store,
            fov_record=hit,
            meta=meta,
            session_date=session_date,
            query_cells=cells,
            translation=(0.0, 0.0),
            fov_sim=1.0,
            peak=1.0,
            output_dir=output_dir,
            decision="hash_match",
        )
        return _write_report(output_dir, report)

    # Candidate matching.
    candidates = store.find_candidates(meta.animal_id, meta.region)
    best: tuple[Optional[FOVRecord], Optional[FOVMatchResult]] = (None, None)
    for cand in candidates:
        cand_mean_m = deserialize_mean_m(blob_store.get(cand.mean_m_uri))
        cand_cells = deserialize_cells(blob_store.get(cand.centroid_table_uri))
        result = match_fov(mean_m, cells, cand_mean_m, cand_cells)
        if best[1] is None or result.fov_sim > best[1].fov_sim:
            best = (cand, result)

    if best[1] is not None and best[1].fov_sim >= AUTO_ACCEPT_THRESHOLD:
        report = _register_session_on_existing(
            store=store,
            blob_store=blob_store,
            fov_record=best[0],
            meta=meta,
            session_date=session_date,
            query_cells=cells,
            translation=best[1].translation_yx,
            fov_sim=best[1].fov_sim,
            peak=best[1].peak_correlation,
            output_dir=output_dir,
            decision="auto_match",
        )
        return _write_report(output_dir, report)

    if best[1] is not None and best[1].fov_sim >= REVIEW_THRESHOLD:
        report = {
            "decision": "review",
            "fov_id": None,
            "candidate_fov_id": best[0].fov_id if best[0] else None,
            "fov_sim": best[1].fov_sim,
            "translation_yx": list(best[1].translation_yx),
            "peak_correlation": best[1].peak_correlation,
            "animal_id": meta.animal_id,
            "region": meta.region,
            "session_date": session_date.isoformat(),
            "fingerprint_hash": fp.fingerprint_hash,
            "message": (
                f"fov_sim={best[1].fov_sim:.3f} in review band "
                f"[{REVIEW_THRESHOLD}, {AUTO_ACCEPT_THRESHOLD}); "
                "no session written. Resolve in the Streamlit Registry tab."
            ),
        }
        return _write_report(output_dir, report)

    report = _mint_new_fov(
        store=store,
        blob_store=blob_store,
        fingerprint=fp,
        meta=meta,
        session_date=session_date,
        cells=cells,
        output_dir=output_dir,
        best_candidate_sim=best[1].fov_sim if best[1] else None,
    )
    return _write_report(output_dir, report)


def _mint_new_fov(
    *,
    store: RegistryStore,
    blob_store: BlobStore,
    fingerprint: Fingerprint,
    meta: FilenameMetadata,
    session_date: date,
    cells: list[CellFeature],
    output_dir: Path,
    best_candidate_sim: Optional[float],
) -> dict:
    fov_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    mean_m_uri = blob_store.put(f"{fov_id}/mean_M.npy", fingerprint.mean_m_blob)
    centroid_uri = blob_store.put(f"{fov_id}/centroids.npy", fingerprint.centroid_blob)

    store.insert_fov(FOVRecord(
        fov_id=fov_id,
        fingerprint_hash=fingerprint.fingerprint_hash,
        animal_id=meta.animal_id,
        region=meta.region,
        mean_m_uri=mean_m_uri,
        centroid_table_uri=centroid_uri,
        created_at=now,
        latest_session_date=session_date,
    ))
    store.insert_session(SessionRecord(
        session_id=session_id,
        fov_id=fov_id,
        session_date=session_date,
        output_dir=str(output_dir),
        fov_sim=None,
        n_matched=0,
        n_new=len(cells),
        n_missing=0,
        created_at=now,
    ))

    global_ids: dict[int, str] = {}
    observations: list[ObservationRecord] = []
    for c in cells:
        gid = str(uuid.uuid4())
        global_ids[c.local_label_id] = gid
        store.insert_cell(CellRecord(
            global_cell_id=gid,
            fov_id=fov_id,
            first_seen_session_id=session_id,
            morphology_summary={
                "first_local_label_id": int(c.local_label_id),
                "area": int(c.area),
                "solidity": float(c.solidity),
                "eccentricity": float(c.eccentricity),
                "nuclear_shadow_score": float(c.nuclear_shadow_score),
                "soma_surround_contrast": float(c.soma_surround_contrast),
            },
        ))
        observations.append(ObservationRecord(
            observation_id=str(uuid.uuid4()),
            global_cell_id=gid,
            session_id=session_id,
            local_label_id=int(c.local_label_id),
            match_score=None,
        ))
    store.insert_observations(observations)

    return {
        "decision": "new_fov",
        "fov_id": fov_id,
        "session_id": session_id,
        "animal_id": meta.animal_id,
        "region": meta.region,
        "session_date": session_date.isoformat(),
        "fingerprint_hash": fingerprint.fingerprint_hash,
        "n_new_cells": len(cells),
        "best_candidate_sim": best_candidate_sim,
        "cell_assignments": [
            {"local_label_id": c.local_label_id, "global_cell_id": global_ids[c.local_label_id]}
            for c in cells
        ],
    }


def _register_session_on_existing(
    *,
    store: RegistryStore,
    blob_store: BlobStore,
    fov_record: FOVRecord,
    meta: FilenameMetadata,
    session_date: date,
    query_cells: list[CellFeature],
    translation: tuple[float, float],
    fov_sim: float,
    peak: float,
    output_dir: Path,
    decision: str,
) -> dict:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    cand_cells_feat = deserialize_cells(blob_store.get(fov_record.centroid_table_uri))
    translated_query = [
        CellFeature(
            local_label_id=c.local_label_id,
            centroid_y=c.centroid_y + translation[0],
            centroid_x=c.centroid_x + translation[1],
            area=c.area,
            solidity=c.solidity,
            eccentricity=c.eccentricity,
            nuclear_shadow_score=c.nuclear_shadow_score,
            soma_surround_contrast=c.soma_surround_contrast,
        )
        for c in query_cells
    ]
    cell_result: CellMatchResult = match_cells(translated_query, cand_cells_feat)

    existing_cells = store.list_cells(fov_record.fov_id)
    first_label_to_gid = {
        int(c.morphology_summary.get("first_local_label_id", -1)): c.global_cell_id
        for c in existing_cells
    }

    observations: list[ObservationRecord] = []
    cell_assignments: list[dict] = []
    new_cells_for_db: list[CellRecord] = []

    for m in cell_result.matches:
        gid = first_label_to_gid.get(int(m.candidate_label_id))
        if gid is None:
            gid = str(uuid.uuid4())
            new_cells_for_db.append(CellRecord(
                global_cell_id=gid,
                fov_id=fov_record.fov_id,
                first_seen_session_id=session_id,
                morphology_summary={"first_local_label_id": int(m.query_label_id)},
            ))
        observations.append(ObservationRecord(
            observation_id=str(uuid.uuid4()),
            global_cell_id=gid,
            session_id=session_id,
            local_label_id=int(m.query_label_id),
            match_score=float(m.score),
        ))
        cell_assignments.append({
            "local_label_id": int(m.query_label_id),
            "global_cell_id": gid,
            "match_kind": "matched",
            "score": float(m.score),
        })

    for q_label in cell_result.new_query_labels:
        query_cell = next(c for c in query_cells if c.local_label_id == q_label)
        gid = str(uuid.uuid4())
        new_cells_for_db.append(CellRecord(
            global_cell_id=gid,
            fov_id=fov_record.fov_id,
            first_seen_session_id=session_id,
            morphology_summary={
                "first_local_label_id": int(q_label),
                "area": int(query_cell.area),
                "solidity": float(query_cell.solidity),
                "eccentricity": float(query_cell.eccentricity),
                "nuclear_shadow_score": float(query_cell.nuclear_shadow_score),
                "soma_surround_contrast": float(query_cell.soma_surround_contrast),
            },
        ))
        observations.append(ObservationRecord(
            observation_id=str(uuid.uuid4()),
            global_cell_id=gid,
            session_id=session_id,
            local_label_id=int(q_label),
            match_score=None,
        ))
        cell_assignments.append({
            "local_label_id": int(q_label),
            "global_cell_id": gid,
            "match_kind": "new",
            "score": None,
        })

    missing_payload = []
    for cand_label in cell_result.missing_candidate_labels:
        gid = first_label_to_gid.get(int(cand_label))
        if gid is None:
            continue
        missing_payload.append({"global_cell_id": gid, "candidate_local_label_id": int(cand_label)})

    store.insert_session(SessionRecord(
        session_id=session_id,
        fov_id=fov_record.fov_id,
        session_date=session_date,
        output_dir=str(output_dir),
        fov_sim=float(fov_sim),
        n_matched=len(cell_result.matches),
        n_new=len(cell_result.new_query_labels),
        n_missing=len(cell_result.missing_candidate_labels),
        created_at=now,
    ))
    for cell in new_cells_for_db:
        store.insert_cell(cell)
    store.insert_observations(observations)
    store.update_fov_latest_session(fov_record.fov_id, session_date)

    return {
        "decision": decision,
        "fov_id": fov_record.fov_id,
        "session_id": session_id,
        "animal_id": meta.animal_id,
        "region": meta.region,
        "session_date": session_date.isoformat(),
        "fingerprint_hash": fov_record.fingerprint_hash,
        "fov_sim": float(fov_sim),
        "peak_correlation": float(peak),
        "translation_yx": list(translation),
        "n_matched": len(cell_result.matches),
        "n_new": len(cell_result.new_query_labels),
        "n_missing": len(cell_result.missing_candidate_labels),
        "cell_assignments": cell_assignments,
        "missing_cells": missing_payload,
    }


def _write_report(output_dir: Path, report: dict) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "registry_match.json").write_text(json.dumps(report, indent=2, default=str))
    return report
