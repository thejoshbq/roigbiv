"""One-shot utility to collapse duplicate ``session`` rows.

Exists because pre-fix workspaces (where :func:`register_or_match` was not
idempotent at the session level) accumulated two rows per pipeline run: one
from the per-TIF registration, one from the end-of-run backfill sweep.

Grouping key: ``(fov_id, output_dir)``.

Winner selection: the earliest ``created_at`` in each group. The earliest
row is the one that minted the Cell records (``Cell.first_seen_session_id``
points at that ``session_id``), so keeping it preserves referential
consistency. Later rows are the no-op duplicates introduced by the backfill
pass on the same output directory.

Losers' ``cell_observation`` rows are deleted first (SQLite enforces
``ON DELETE CASCADE`` only when ``PRAGMA foreign_keys=ON`` per connection —
we don't rely on it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import delete, select

from roigbiv.registry import models as m
from roigbiv.registry.store.sqlalchemy_store import SQLAlchemyStore


@dataclass
class DedupeSummary:
    n_groups_with_duplicates: int
    n_session_rows_deleted: int
    n_observation_rows_deleted: int


def dedupe_sessions(
    store: SQLAlchemyStore, *, dry_run: bool = False
) -> DedupeSummary:
    """Collapse duplicate ``session`` rows keyed by ``(fov_id, output_dir)``.

    Parameters
    ----------
    store : SQLAlchemyStore
        The registry store to dedupe. Must be the concrete SQLAlchemy
        backend — this routine reaches into the ORM session for bulk
        deletes rather than going through the RegistryStore Protocol.
    dry_run : bool
        When True, only counts what would be deleted; no writes.
    """
    n_groups = 0
    n_sessions_deleted = 0
    n_obs_deleted = 0

    with store._Session() as s:  # noqa: SLF001 — intentional: ORM access
        rows = s.scalars(
            select(m.Session).order_by(m.Session.created_at)
        ).all()
        groups: dict[tuple[str, str], list] = {}
        for r in rows:
            groups.setdefault((r.fov_id, r.output_dir), []).append(r)

        loser_ids: list[str] = []
        for key, sess_rows in groups.items():
            if len(sess_rows) <= 1:
                continue
            n_groups += 1
            # Oldest (first in the list; the select ordered by created_at asc)
            # wins; the rest are duplicates.
            for dup in sess_rows[1:]:
                loser_ids.append(dup.session_id)

        if not loser_ids:
            return DedupeSummary(
                n_groups_with_duplicates=0,
                n_session_rows_deleted=0,
                n_observation_rows_deleted=0,
            )

        obs_count_stmt = select(m.CellObservation).where(
            m.CellObservation.session_id.in_(loser_ids)
        )
        n_obs_deleted = len(s.scalars(obs_count_stmt).all())
        n_sessions_deleted = len(loser_ids)

        if dry_run:
            return DedupeSummary(
                n_groups_with_duplicates=n_groups,
                n_session_rows_deleted=n_sessions_deleted,
                n_observation_rows_deleted=n_obs_deleted,
            )

        s.execute(
            delete(m.CellObservation).where(
                m.CellObservation.session_id.in_(loser_ids)
            )
        )
        s.execute(
            delete(m.Session).where(m.Session.session_id.in_(loser_ids))
        )
        s.commit()

    return DedupeSummary(
        n_groups_with_duplicates=n_groups,
        n_session_rows_deleted=n_sessions_deleted,
        n_observation_rows_deleted=n_obs_deleted,
    )


def dedupe_from_env(dry_run: bool = False) -> DedupeSummary:
    """Convenience wrapper that resolves the store from the env DSN."""
    from roigbiv.registry import build_store

    store = build_store()
    store.ensure_schema()
    if not isinstance(store, SQLAlchemyStore):
        raise RuntimeError(
            "dedupe_sessions requires the SQLAlchemyStore backend; "
            f"got {type(store).__name__}"
        )
    return dedupe_sessions(store, dry_run=dry_run)


__all__ = ["DedupeSummary", "dedupe_sessions", "dedupe_from_env"]
