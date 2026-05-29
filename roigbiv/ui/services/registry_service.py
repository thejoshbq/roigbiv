"""Registry queries exposed to the Dash UI.

Thin wrappers over :mod:`roigbiv.registry`. Each function opens a fresh
store so the current ``ROIGBIV_REGISTRY_DSN`` (which the workspace runner
keeps in sync with the selected input root) is always honored.

Maintenance actions (migrate, backfill, dedupe) live in the
``roigbiv-registry`` CLI rather than the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
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


def list_fovs(cfg=None) -> list[FOVRow]:
    from roigbiv.registry import build_store

    store = build_store(cfg=cfg)
    store.ensure_schema()
    rows: list[FOVRow] = []
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


# ── internals ──────────────────────────────────────────────────────────────


def _known_animal_region_pairs(store) -> set[tuple[str, str]]:
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
