"""add ROICaT cluster columns (v3)

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-22

Adds:
  * session.cluster_labels_uri (nullable) — blob URI for the per-session int32
    ROICaT cluster label array (length == number of ROIs in the session;
    -1 = unclustered).
  * cell_observation.cluster_label (nullable) — ROICaT cluster label assigned
    to this observation. Mutable (may change when the FOV is re-clustered);
    canonical per-ROI identity remains ``global_cell_id``.

Additive only — existing v1 / v2 FOV rows remain readable and the new columns
stay NULL until the parent session / observation is re-registered under the
v3 pipeline.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("session") as batch:
        batch.add_column(sa.Column("cluster_labels_uri", sa.String(512), nullable=True))

    with op.batch_alter_table("cell_observation") as batch:
        batch.add_column(sa.Column("cluster_label", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("cell_observation") as batch:
        batch.drop_column("cluster_label")

    with op.batch_alter_table("session") as batch:
        batch.drop_column("cluster_labels_uri")
