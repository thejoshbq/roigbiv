"""Programmatic Alembic access.

`ensure_alembic_head()` is idempotent: it inspects the live schema, stamps at
the correct starting revision for DBs that pre-date alembic tracking, then
runs `upgrade head`. Used by the `roigbiv-registry migrate` CLI, the Streamlit
"Run database migrations" button, and auto-invoked from
`SQLAlchemyStore.ensure_schema()` so that every store open self-heals.
"""
from __future__ import annotations

from pathlib import Path


V2_FOV_COLUMNS = ("fingerprint_version", "fov_embedding_uri", "roi_embeddings_uri")


def _alembic_config():
    from alembic.config import Config

    cfg_path = Path(__file__).resolve().parent / "alembic.ini"
    return Config(str(cfg_path))


def ensure_alembic_head() -> str:
    """Move the active DB to alembic head, whichever state it started in.

    Cases:
      * No `fov` table (fresh DB)           → `upgrade head`.
      * `fov` table, no alembic version:
        - v2 columns present                → stamp at head.
        - v2 columns missing (legacy v1 DB) → stamp at `0001`, then
                                              `upgrade head`.
      * Alembic version already recorded    → `upgrade head` (no-op if
                                              already at head).

    Returns a short description of the action taken.
    """
    from alembic import command
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import inspect

    from roigbiv.registry import build_store

    store = build_store()
    cfg = _alembic_config()

    with store.engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        current = ctx.get_current_revision()

    if current is None:
        insp = inspect(store.engine)
        tables = set(insp.get_table_names())
        if "fov" in tables:
            fov_cols = {c["name"] for c in insp.get_columns("fov")}
            session_cols = (
                {c["name"] for c in insp.get_columns("session")}
                if "session" in tables
                else set()
            )
            # v3 → latest: adds session.cluster_labels_uri.
            if "cluster_labels_uri" in session_cols:
                command.stamp(cfg, "head")
                return "stamped at head (pre-existing v3 schema)"
            # v2 schema (has embedding columns but not yet the v3 cluster column).
            if all(col in fov_cols for col in V2_FOV_COLUMNS):
                command.stamp(cfg, "0002")
                command.upgrade(cfg, "head")
                return "stamped at 0002, upgraded to head (pre-existing v2 schema)"
            command.stamp(cfg, "0001")
            command.upgrade(cfg, "head")
            return "stamped at 0001, upgraded to head (legacy v1 schema)"

    command.upgrade(cfg, "head")
    return "upgraded to head"


def current_revision() -> str:
    from alembic.runtime.migration import MigrationContext

    from roigbiv.registry import build_store

    store = build_store()
    with store.engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        return ctx.get_current_revision() or ""
