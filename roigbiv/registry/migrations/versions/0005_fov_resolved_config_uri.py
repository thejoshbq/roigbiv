"""add fov.resolved_config_uri for optics auto-adaptation memory

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-22

Adds:
  * fov.resolved_config_uri (nullable) — blob URI for the optics auto-adaptation
    config (resolved profile + measured soma scale + derived gates) that
    succeeded for this FOV. Reused as a prior when a repeat FOV is matched, so
    recurring experiments skip re-discovery (and the pause-to-confirm) instead
    of starting from defaults.

Additive only — existing v1..v4 FOV rows remain fully readable with the new
column NULL until the FOV is re-registered with a resolved config persisted.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("fov") as batch:
        batch.add_column(
            sa.Column("resolved_config_uri", sa.String(512), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("fov") as batch:
        batch.drop_column("resolved_config_uri")
