"""SQLAlchemy-backed RegistryStore.

Works against both SQLite (Phase A) and Postgres (Phase B) with zero code
change — the only difference is the DSN passed to the constructor.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import create_engine, select
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


class SQLAlchemyStore:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn
        connect_args = {"check_same_thread": False} if dsn.startswith("sqlite") else {}
        self.engine: Engine = create_engine(
            dsn, future=True, connect_args=connect_args,
        )
        self._Session = sessionmaker(self.engine, expire_on_commit=False, future=True)

    def ensure_schema(self) -> None:
        m.Base.metadata.create_all(self.engine)
        from roigbiv.registry.migrate import ensure_alembic_head

        ensure_alembic_head()

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
            ))
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
    def insert_session(self, session: SessionRecord) -> None:
        with self._Session() as s:
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
            ))
            s.commit()

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

    def list_sessions(self, fov_id: str) -> list[SessionRecord]:
        with self._Session() as s:
            rows = s.scalars(
                select(m.Session)
                .where(m.Session.fov_id == fov_id)
                .order_by(m.Session.session_date)
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
