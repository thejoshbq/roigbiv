"""Programmatic Alembic access.

`ensure_alembic_head()` is idempotent: it stamps pre-existing schemas at head
(for DBs initialized via SQLAlchemyStore.ensure_schema's create_all path) and
runs `upgrade head` otherwise. Used by the `roigbiv-registry migrate` CLI
subcommand.
"""
from __future__ import annotations

from pathlib import Path


def _alembic_config():
    from alembic.config import Config

    cfg_path = Path(__file__).resolve().parent / "alembic.ini"
    return Config(str(cfg_path))


def ensure_alembic_head() -> str:
    """Move the active DB to alembic head, whichever state it started in.

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
        if "fov" in set(insp.get_table_names()):
            command.stamp(cfg, "head")
            return "stamped at head (pre-existing schema)"

    command.upgrade(cfg, "head")
    return "upgraded to head"


def current_revision() -> str:
    from alembic.runtime.migration import MigrationContext

    from roigbiv.registry import build_store

    store = build_store()
    with store.engine.connect() as conn:
        ctx = MigrationContext.configure(conn)
        return ctx.get_current_revision() or ""
