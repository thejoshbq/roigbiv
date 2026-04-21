"""SQLAlchemy-backed RegistryStore.

Works against both SQLite (Phase A) and Postgres (Phase B) with zero code
change — the only difference is the DSN passed to the constructor.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

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
            ))
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
                n_matched=session.n_matched,
                n_new=session.n_new,
                n_missing=session.n_missing,
                created_at=session.created_at or datetime.now(timezone.utc),
            ))
            s.commit()

    def list_sessions(self, fov_id: str) -> list[SessionRecord]:
        with self._Session() as s:
            rows = s.scalars(
                select(m.Session)
                .where(m.Session.fov_id == fov_id)
                .order_by(m.Session.session_date)
            ).all()
            return [_session_to_record(r) for r in rows]

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
    )


def _session_to_record(row: m.Session) -> SessionRecord:
    return SessionRecord(
        session_id=row.session_id,
        fov_id=row.fov_id,
        session_date=row.session_date,
        output_dir=row.output_dir,
        fov_sim=row.fov_sim,
        n_matched=row.n_matched,
        n_new=row.n_new,
        n_missing=row.n_missing,
        created_at=row.created_at,
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
    )
