"""Registry queries exposed to the Dash UI.

Thin wrappers over :mod:`roigbiv.registry`. Each function opens a fresh
store so the current ``ROIGBIV_REGISTRY_DSN`` (which the workspace runner
keeps in sync with the selected input root) is always honored.

Maintenance actions (migrate, backfill, dedupe) live in the
``roigbiv-registry`` CLI rather than the UI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


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


def anomaly_payload(report) -> dict:
    """Flatten a :class:`~roigbiv.registry.anomalies.FOVAnomalyReport` for Dash.

    Only cells *with* an anomaly are carried — a fully-present cell needs no
    explanation and a large FOV would otherwise ship hundreds of rows into the
    browser on every poll.
    """
    return {
        "counts": report.counts,
        "ordering_is_confirmed": report.ordering_is_confirmed,
        "cells": [
            {
                "global_cell_id": c.global_cell_id,
                "present": list(c.present),
                "anomalies": c.anomalies,
                "first_seen": c.first_seen,
                "last_seen": c.last_seen,
            }
            for c in report.cells if c.anomalies
        ],
        "sessions": [
            {
                "sequence_index": s.sequence_index,
                "session_date": (s.session_date.isoformat()
                                 if s.session_date else None),
                "output_dir": s.output_dir,
            }
            for s in report.sessions
        ],
    }


def workspace_anomalies(output_dirs: Iterable[Path], cfg=None) -> dict[str, dict]:
    """Anomaly reports for every FOV this workspace's sessions belong to.

    Read from the registry rather than from a run's in-memory results, so a
    workspace tracked in an earlier UI session — or from ``roigbiv-pipeline
    --track`` — still reports. Returns ``{fov_id: anomaly_payload}``, empty when
    none of these output directories has been registered yet.
    """
    from roigbiv.registry import build_store
    from roigbiv.registry.anomalies import cell_timeline

    store = build_store(cfg=cfg)
    store.ensure_schema()

    fov_ids: list[str] = []
    for out_dir in output_dirs:
        session = _session_for(store, out_dir)
        if session is not None and session.fov_id not in fov_ids:
            fov_ids.append(session.fov_id)
    return {fid: anomaly_payload(cell_timeline(store, fid)) for fid in fov_ids}


# ── internals ──────────────────────────────────────────────────────────────


def _session_for(store, out_dir: Path):
    """Look a session up by output directory, tolerating path spelling.

    Sessions are keyed by the exact string the registering run passed in, which
    may or may not have been resolved; try both rather than silently reporting
    a tracked workspace as untracked.
    """
    for key in dict.fromkeys([str(out_dir), str(Path(out_dir).resolve())]):
        session = store.get_session_by_output_dir(key)
        if session is not None:
            return session
    return None


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
