"""Registry queries and maintenance actions exposed to the Dash UI.

Thin wrappers over :mod:`roigbiv.registry`. Each function opens a fresh
store so the current ``ROIGBIV_REGISTRY_DSN`` (which the workspace runner
keeps in sync with the selected input root) is always honored.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional


@dataclass
class FOVRow:
    fov_id: str
    animal_id: Optional[str]
    region: Optional[str]
    created_at: Optional[str]
    latest_session_date: Optional[str]
    fingerprint_version: Optional[int]
    n_sessions: int


@dataclass
class SessionRow:
    session_id: str
    fov_id: str
    session_date: Optional[str]
    output_dir: str
    fov_posterior: Optional[float]
    n_matched: int
    n_new: int
    n_missing: int


def list_fovs() -> list[FOVRow]:
    from roigbiv.registry import build_store

    store = build_store()
    store.ensure_schema()
    rows: list[FOVRow] = []
    # Find all FOVs by scanning observations' distinct fov list; simplest is
    # to use find_candidates with empty filters — but not every backend supports
    # that. Fall back to listing per (animal_id, region) pair we see.
    seen: set[str] = set()
    for (animal_id, region) in _known_animal_region_pairs(store):
        for fov in store.find_candidates(animal_id, region):
            if fov.fov_id in seen:
                continue
            seen.add(fov.fov_id)
            sessions = store.list_sessions(fov.fov_id)
            rows.append(FOVRow(
                fov_id=fov.fov_id,
                animal_id=fov.animal_id,
                region=fov.region,
                created_at=str(fov.created_at) if fov.created_at else None,
                latest_session_date=_fmt_date(fov.latest_session_date),
                fingerprint_version=fov.fingerprint_version,
                n_sessions=len(sessions),
            ))
    rows.sort(key=lambda r: (r.animal_id or "", r.region or "", r.fov_id))
    return rows


def list_sessions_for_fov(fov_id: str) -> list[SessionRow]:
    from roigbiv.registry import build_store

    store = build_store()
    store.ensure_schema()
    sessions = sorted(
        store.list_sessions(fov_id),
        key=lambda s: s.session_date or date.min,
    )
    return [
        SessionRow(
            session_id=s.session_id,
            fov_id=s.fov_id,
            session_date=_fmt_date(s.session_date),
            output_dir=str(s.output_dir),
            fov_posterior=s.fov_posterior,
            n_matched=int(s.n_matched or 0),
            n_new=int(s.n_new or 0),
            n_missing=int(s.n_missing or 0),
        )
        for s in sessions
    ]


def run_migrations() -> str:
    """Alembic upgrade — idempotent."""
    from roigbiv.registry.migrate import ensure_alembic_head

    return ensure_alembic_head()


def run_backfill_now(root: Path, dry_run: bool = False) -> list[dict]:
    from roigbiv.registry.backfill import run_backfill
    from roigbiv.registry.config import RegistryConfig

    cfg = RegistryConfig.from_env()
    return run_backfill(root, dry_run=dry_run, cfg=cfg)


# ── internals ──────────────────────────────────────────────────────────────


def _known_animal_region_pairs(store) -> set[tuple[str, str]]:
    """Collect distinct (animal_id, region) pairs from the FOV table.

    The SQLAlchemy store exposes ``find_candidates(animal_id, region)`` but
    not a blanket list; we run a small raw query to enumerate what's in the
    DB. For very large deployments this is still cheap (FOVs scale like
    sessions, not ROIs).
    """
    from sqlalchemy import distinct, select

    from roigbiv.registry import models as m

    pairs: set[tuple[str, str]] = set()
    with store.engine.connect() as conn:
        result = conn.execute(
            select(distinct(m.FOV.animal_id), m.FOV.region)
        )
        for animal_id, region in result:
            pairs.add((animal_id or "", region or ""))
    return pairs


def _fmt_date(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return value.isoformat()
    except AttributeError:
        return str(value)
