"""SQLAlchemy-backed RegistryStore.

Works against both SQLite (Phase A) and Postgres (Phase B) with zero code
change — the only difference is the DSN passed to the constructor.
"""
from __future__ import annotations

import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import create_engine, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SASession
from sqlalchemy.orm import sessionmaker

from roigbiv.registry import models as m
from roigbiv.registry.store.base import (
    CellRecord,
    FOVRecord,
    ObservationRecord,
    SessionRecord,
)


# Alembic's EnvironmentContext stores its `script` proxy in module globals,
# so two threads overlapping in `command.upgrade` race on `del globals_['script']`
# during __exit__ → KeyError('script'). Serialise migrations across threads and
# memoise per-DSN so the hot path (Dash callbacks) doesn't re-run alembic.
# In-memory SQLite (``sqlite://`` / ``sqlite:///:memory:``) is excluded from the
# memo because each engine gets its own private DB even when the DSN string is
# identical — caching would skip schema creation on a fresh blank engine.
_MIGRATE_LOCK = threading.Lock()
_MIGRATED_DSNS: set[str] = set()


def _is_memory_sqlite(dsn: str) -> bool:
    return dsn == "sqlite://" or dsn.startswith("sqlite:///:memory:")


class SQLAlchemyStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        connect_args = {"check_same_thread": False} if dsn.startswith("sqlite") else {}
        self.engine: Engine = create_engine(
            dsn, future=True, connect_args=connect_args,
        )
        self._Session = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def ensure_schema(self) -> None:
        cacheable = not _is_memory_sqlite(self.dsn)
        if cacheable and self.dsn in _MIGRATED_DSNS:
            return
        with _MIGRATE_LOCK:
            if cacheable and self.dsn in _MIGRATED_DSNS:
                return
            m.Base.metadata.create_all(self.engine)
            from roigbiv.registry.migrate import ensure_alembic_head

            ensure_alembic_head()
            if cacheable:
                _MIGRATED_DSNS.add(self.dsn)

    # ── FOV ───────────────────────────────────────────────────────────────
    def get_fov_by_hash(self, fingerprint_hash: str) -> Optional[FOVRecord]:
        with self._Session() as s:
            row = s.scalar(select(m.FOV).where(m.FOV.fingerprint_hash == fingerprint_hash))
            return _fov_to_record(row) if row else None

    def get_fov(self, fov_id: str) -> Optional[FOVRecord]:
        with self._Session() as s:
            row = s.get(m.FOV, fov_id)
            return _fov_to_record(row) if row else None

    def find_candidates(self, animal_id: str, region: str) -> list[FOVRecord]:
        with self._Session() as s:
            rows = s.scalars(
                select(m.FOV).where(
                    m.FOV.animal_id == animal_id,
                    m.FOV.region == region,
                )
            ).all()
            return [_fov_to_record(r) for r in rows]

    def find_candidates_by_embedding(
        self,
        animal_id: str,
        region: str,
        fov_embedding: np.ndarray,
        blob_store,
        top_k: int = 10,
        min_cosine: float = 0.0,
    ) -> list[FOVRecord]:
        """Rank (animal_id, region) FOVs by cosine similarity of pooled embedding.

        Only FOVs with a populated ``fov_embedding_uri`` are considered; FOVs
        without an embedding (v1 rows) are silently skipped. Returns up to
        ``top_k`` records sorted by descending similarity. If no v2 FOVs exist
        in the candidate pool this returns an empty list, letting the caller
        fall back to the region-only ``find_candidates``.
        """
        candidates = self.find_candidates(animal_id, region)
        query = np.asarray(fov_embedding, dtype=np.float32).ravel()
        q_norm = float(np.linalg.norm(query))
        if q_norm <= 0:
            return []
        query = query / q_norm
        scored: list[tuple[float, FOVRecord]] = []
        for cand in candidates:
            if not cand.fov_embedding_uri:
                continue
            try:
                blob = blob_store.get(cand.fov_embedding_uri)
            except Exception:
                continue
            import io
            vec = np.load(io.BytesIO(blob), allow_pickle=False).astype(np.float32).ravel()
            if vec.shape != query.shape:
                continue
            n = float(np.linalg.norm(vec))
            if n <= 0:
                continue
            sim = float(np.dot(query, vec / n))
            if sim < min_cosine:
                continue
            scored.append((sim, cand))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [rec for _, rec in scored[:top_k]]

    def list_fovs(self, filters: Optional[dict] = None) -> list[FOVRecord]:
        with self._Session() as s:
            stmt = select(m.FOV)
            for key, val in (filters or {}).items():
                stmt = stmt.where(getattr(m.FOV, key) == val)
            rows = s.scalars(stmt).all()
            return [_fov_to_record(r) for r in rows]

    def insert_fov(self, fov: FOVRecord) -> None:
        with self._Session() as s:
            s.add(m.FOV(
                fov_id=fov.fov_id,
                fingerprint_hash=fov.fingerprint_hash,
                animal_id=fov.animal_id,
                region=fov.region,
                mean_m_uri=fov.mean_m_uri,
                centroid_table_uri=fov.centroid_table_uri,
                created_at=fov.created_at,
                latest_session_date=fov.latest_session_date,
                fingerprint_version=fov.fingerprint_version,
                fov_embedding_uri=fov.fov_embedding_uri,
                roi_embeddings_uri=fov.roi_embeddings_uri,
                resolved_config_uri=fov.resolved_config_uri,
            ))
            s.commit()

    def update_fov_resolved_config(
        self, fov_id: str, resolved_config_uri: str
    ) -> None:
        """Attach the optics auto-adaptation config blob URI to an existing FOV."""
        with self._Session() as s:
            row = s.get(m.FOV, fov_id)
            if row is None:
                return
            row.resolved_config_uri = resolved_config_uri
            s.commit()

    def update_fov_embeddings(
        self,
        fov_id: str,
        fov_embedding_uri: str,
        roi_embeddings_uri: str,
        fingerprint_version: int,
    ) -> None:
        """Attach embedding blob URIs to an existing FOV row (v1 → v2 upgrade)."""
        with self._Session() as s:
            row = s.get(m.FOV, fov_id)
            if row is None:
                return
            row.fov_embedding_uri = fov_embedding_uri
            row.roi_embeddings_uri = roi_embeddings_uri
            row.fingerprint_version = fingerprint_version
            s.commit()

    def update_fov_latest_session(self, fov_id: str, session_date: date) -> None:
        with self._Session() as s:
            row = s.get(m.FOV, fov_id)
            if row is None:
                return
            if row.latest_session_date is None or session_date > row.latest_session_date:
                row.latest_session_date = session_date
                s.commit()

    # ── Session ───────────────────────────────────────────────────────────
    def get_session(
        self, fov_id: str, output_dir: str
    ) -> Optional[SessionRecord]:
        """The session row for exactly this ``(fov_id, output_dir)``, if any.

        Keyed on the pair the unique constraint uses, so a caller can learn the
        id a re-registration will land on *before* it builds anything that
        references it. ``get_session_by_output_dir`` answers a different
        question (newest row across all FOVs) and is not a substitute.
        """
        with self._Session() as s:
            row = s.scalar(
                select(m.Session).where(
                    m.Session.fov_id == fov_id,
                    m.Session.output_dir == output_dir,
                )
            )
            return _session_to_record(row) if row else None

    def upsert_session(self, session: SessionRecord) -> str:
        """Insert *session*, or refresh the row already holding its key.

        Returns the ``session_id`` now authoritative for
        ``(fov_id, output_dir)`` — **not necessarily the one passed in**. The
        unique constraint means a re-registration of the same output directory
        cannot get its own row, and the previous version of this method dealt
        with that by returning silently: the caller went on to write
        observations against a ``session_id`` that was never inserted, leaving
        rows referencing a session that does not exist (44 of them in the
        reference workspace) and stale counts on the row that survived.

        Adopting the existing row also means refreshing it. The counts,
        posterior and cluster-labels blob all describe *this* registration, so
        leaving the previous run's values in place would make the row disagree
        with the ``registry_match.json`` written beside it.
        """
        with self._Session() as s:
            row = s.scalar(
                select(m.Session).where(
                    m.Session.fov_id == session.fov_id,
                    m.Session.output_dir == session.output_dir,
                )
            )
            if row is not None:
                row.session_date = session.session_date
                row.fov_sim = session.fov_sim
                row.fov_posterior = session.fov_posterior
                row.n_matched = session.n_matched
                row.n_new = session.n_new
                row.n_missing = session.n_missing
                if session.cluster_labels_uri is not None:
                    row.cluster_labels_uri = session.cluster_labels_uri
                if session.sequence_index is not None:
                    row.sequence_index = session.sequence_index
                s.commit()
                return row.session_id
            s.add(m.Session(
                session_id=session.session_id,
                fov_id=session.fov_id,
                session_date=session.session_date,
                output_dir=session.output_dir,
                fov_sim=session.fov_sim,
                fov_posterior=session.fov_posterior,
                n_matched=session.n_matched,
                n_new=session.n_new,
                n_missing=session.n_missing,
                created_at=session.created_at or datetime.now(timezone.utc),
                cluster_labels_uri=session.cluster_labels_uri,
                sequence_index=session.sequence_index,
            ))
            s.commit()
            return session.session_id

    def delete_observations_for_session(self, session_id: str) -> int:
        """Drop a session's observations so a re-registration can replace them.

        ``(session_id, local_label_id)`` is unique, so re-registering an output
        directory without this raises on the first repeated label.
        """
        with self._Session() as s:
            result = s.execute(
                delete(m.CellObservation).where(
                    m.CellObservation.session_id == session_id
                )
            )
            s.commit()
            return int(result.rowcount or 0)

    def update_session_cluster_labels(
        self, session_id: str, cluster_labels_uri: str
    ) -> None:
        """Attach a cluster-labels blob URI to an existing session row."""
        with self._Session() as s:
            row = s.get(m.Session, session_id)
            if row is None:
                return
            row.cluster_labels_uri = cluster_labels_uri
            s.commit()

    def update_session_sequence(
        self, session_id: str, sequence_index: Optional[int]
    ) -> None:
        """Set (or clear) a session's human-assigned timeline position."""
        with self._Session() as s:
            row = s.get(m.Session, session_id)
            if row is None:
                return
            row.sequence_index = sequence_index
            s.commit()

    def list_sessions(self, fov_id: str) -> list[SessionRecord]:
        """Sessions in timeline order.

        A human-assigned ``sequence_index`` wins where one exists; unordered
        sessions fall back to ``session_date`` and sort after the ordered ones,
        so a partially-ordered FOV still reads sensibly.
        """
        with self._Session() as s:
            rows = s.scalars(
                select(m.Session)
                .where(m.Session.fov_id == fov_id)
                .order_by(
                    m.Session.sequence_index.is_(None),
                    m.Session.sequence_index,
                    m.Session.session_date,
                )
            ).all()
            return [_session_to_record(r) for r in rows]

    def get_session_by_output_dir(
        self, output_dir: str
    ) -> Optional[SessionRecord]:
        """Return the most recent session row keyed to ``output_dir``.

        Multiple rows can exist during the transition window before the
        workspace DB has been deduped; callers treat the newest one as
        authoritative.
        """
        with self._Session() as s:
            row = s.scalars(
                select(m.Session)
                .where(m.Session.output_dir == output_dir)
                .order_by(m.Session.created_at.desc())
            ).first()
            return _session_to_record(row) if row else None

    def supersede_session(self, output_dir: str) -> dict:
        """Delete prior run rows for ``output_dir`` so a re-run replaces them.

        Deletes, in FK-safe order: the ``cell_observation`` rows of every
        session tied to ``output_dir``, then those ``session`` rows. For each
        FOV thereby left with no remaining sessions (an orphan — its only run
        was just superseded), deletes its ``cell`` rows and the ``fov`` row.
        A FOV that still has sessions from other output dirs is left intact.

        SQLite honours ``ON DELETE CASCADE`` only with ``PRAGMA
        foreign_keys=ON`` per connection, so we cascade explicitly (matching
        :mod:`roigbiv.registry.dedupe`).

        Assumes the superseded session is the cell origin for ``output_dir``
        (the intended same-output_dir re-run case). When a FOV survives because
        it carries sessions from *other* output dirs, its ``Cell`` rows are
        kept; a ``Cell.first_seen_session_id`` that pointed at the dropped
        session becomes a dangling soft-reference (the column is nullable with
        no FK, read only as a string tie-break in matching, so it degrades
        gracefully rather than erroring).

        Returns deletion counts: ``{"sessions", "observations", "fovs",
        "cells"}``.
        """
        counts = {"sessions": 0, "observations": 0, "fovs": 0, "cells": 0}
        with self._Session() as s:
            sess_rows = s.scalars(
                select(m.Session).where(m.Session.output_dir == output_dir)
            ).all()
            if not sess_rows:
                return counts
            session_ids = [r.session_id for r in sess_rows]
            fov_ids = {r.fov_id for r in sess_rows}

            obs_ids = s.scalars(
                select(m.CellObservation.observation_id).where(
                    m.CellObservation.session_id.in_(session_ids)
                )
            ).all()
            counts["observations"] = len(obs_ids)
            if obs_ids:
                s.execute(
                    delete(m.CellObservation).where(
                        m.CellObservation.session_id.in_(session_ids)
                    )
                )
            s.execute(
                delete(m.Session).where(m.Session.session_id.in_(session_ids))
            )
            counts["sessions"] = len(session_ids)

            # Drop any FOV the supersede just orphaned (zero sessions left).
            for fov_id in fov_ids:
                remaining = s.scalar(
                    select(m.Session).where(m.Session.fov_id == fov_id)
                )
                if remaining is not None:
                    continue
                cell_ids = s.scalars(
                    select(m.Cell.global_cell_id).where(m.Cell.fov_id == fov_id)
                ).all()
                if cell_ids:
                    s.execute(delete(m.Cell).where(m.Cell.fov_id == fov_id))
                    counts["cells"] += len(cell_ids)
                s.execute(delete(m.FOV).where(m.FOV.fov_id == fov_id))
                counts["fovs"] += 1
            s.commit()
        return counts

    # ── Cell ──────────────────────────────────────────────────────────────
    def insert_cell(self, cell: CellRecord) -> None:
        with self._Session() as s:
            s.add(m.Cell(
                global_cell_id=cell.global_cell_id,
                fov_id=cell.fov_id,
                first_seen_session_id=cell.first_seen_session_id,
                morphology_summary=cell.morphology_summary,
            ))
            s.commit()

    def list_cells(self, fov_id: str) -> list[CellRecord]:
        with self._Session() as s:
            rows = s.scalars(
                select(m.Cell).where(m.Cell.fov_id == fov_id)
            ).all()
            return [_cell_to_record(r) for r in rows]

    # ── Observation ───────────────────────────────────────────────────────
    def insert_observations(self, observations: list[ObservationRecord]) -> None:
        if not observations:
            return
        with self._Session() as s:
            for obs in observations:
                s.add(m.CellObservation(
                    observation_id=obs.observation_id,
                    global_cell_id=obs.global_cell_id,
                    session_id=obs.session_id,
                    local_label_id=obs.local_label_id,
                    match_score=obs.match_score,
                    cluster_label=obs.cluster_label,
                ))
            s.commit()

    def replace_observations(
        self, session_ids: list[str], observations: list[ObservationRecord]
    ) -> int:
        """Atomically swap every observation of ``session_ids`` for ``observations``.

        ``(session_id, local_label_id)`` is unique, so moving a label from one
        cell to another is necessarily a delete followed by an insert — no
        in-place update avoids a transient constraint violation. Every other
        method in this class commits its own single statement, so that pair
        cannot otherwise be made atomic. That matters more here than it does
        for a fresh registration: the engine is built with
        ``check_same_thread=False`` and the Dash UI serves callbacks on
        multiple threads, so an interactive edit can genuinely interleave with
        another write. One session and one commit means a failed insert rolls
        the delete back with it, rather than leaving the affected sessions
        with no observations at all.

        Returns the number of rows deleted.
        """
        with self._Session() as s:
            result = s.execute(
                delete(m.CellObservation).where(
                    m.CellObservation.session_id.in_(session_ids)
                )
            )
            for obs in observations:
                s.add(m.CellObservation(
                    observation_id=obs.observation_id,
                    global_cell_id=obs.global_cell_id,
                    session_id=obs.session_id,
                    local_label_id=obs.local_label_id,
                    match_score=obs.match_score,
                    cluster_label=obs.cluster_label,
                ))
            s.commit()
            return int(result.rowcount or 0)

    def list_observations_for_cell(self, global_cell_id: str) -> list[ObservationRecord]:
        with self._Session() as s:
            rows = s.scalars(
                select(m.CellObservation).where(
                    m.CellObservation.global_cell_id == global_cell_id
                )
            ).all()
            return [_obs_to_record(r) for r in rows]

    def list_observations_for_session(self, session_id: str) -> list[ObservationRecord]:
        with self._Session() as s:
            rows = s.scalars(
                select(m.CellObservation).where(m.CellObservation.session_id == session_id)
            ).all()
            return [_obs_to_record(r) for r in rows]


def _fov_to_record(row: m.FOV) -> FOVRecord:
    return FOVRecord(
        fov_id=row.fov_id,
        fingerprint_hash=row.fingerprint_hash,
        animal_id=row.animal_id,
        region=row.region,
        mean_m_uri=row.mean_m_uri,
        centroid_table_uri=row.centroid_table_uri,
        created_at=row.created_at,
        latest_session_date=row.latest_session_date,
        fingerprint_version=row.fingerprint_version or 1,
        fov_embedding_uri=row.fov_embedding_uri,
        roi_embeddings_uri=row.roi_embeddings_uri,
        resolved_config_uri=row.resolved_config_uri,
    )


def _session_to_record(row: m.Session) -> SessionRecord:
    return SessionRecord(
        session_id=row.session_id,
        fov_id=row.fov_id,
        session_date=row.session_date,
        output_dir=row.output_dir,
        fov_sim=row.fov_sim,
        fov_posterior=row.fov_posterior,
        n_matched=row.n_matched,
        n_new=row.n_new,
        n_missing=row.n_missing,
        created_at=row.created_at,
        cluster_labels_uri=row.cluster_labels_uri,
        sequence_index=row.sequence_index,
    )


def _cell_to_record(row: m.Cell) -> CellRecord:
    return CellRecord(
        global_cell_id=row.global_cell_id,
        fov_id=row.fov_id,
        first_seen_session_id=row.first_seen_session_id,
        morphology_summary=row.morphology_summary or {},
    )


def _obs_to_record(row: m.CellObservation) -> ObservationRecord:
    return ObservationRecord(
        observation_id=row.observation_id,
        global_cell_id=row.global_cell_id,
        session_id=row.session_id,
        local_label_id=row.local_label_id,
        match_score=row.match_score,
        cluster_label=row.cluster_label,
    )
