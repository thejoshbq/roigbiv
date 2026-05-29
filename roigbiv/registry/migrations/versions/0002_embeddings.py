"""add learned embedding columns

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-21

Adds:
  * fov.fingerprint_version (NOT NULL, default 1) — schema version of stored
    fingerprint.
  * fov.fov_embedding_uri (nullable) — blob URI for the pooled FOV embedding
    written alongside mean_M and centroids.
  * fov.roi_embeddings_uri (nullable) — blob URI for the (N, D) per-ROI
    embedding matrix, rows aligned with the centroid table.
  * session.fov_posterior (nullable) — calibrated probability-of-same-FOV from
    the v2 probabilistic matcher.

Existing v1 FOV rows remain fully readable — the embedding URIs are left NULL
and ``fingerprint_version`` defaults to 1 so the orchestrator falls back to
the geometric-only match path for those FOVs.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("fov") as batch:
        batch.add_column(
            sa.Column(
                "fingerprint_version",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )
        batch.add_column(sa.Column("fov_embedding_uri", sa.String(512), nullable=True))
        batch.add_column(sa.Column("roi_embeddings_uri", sa.String(512), nullable=True))

    with op.batch_alter_table("session") as batch:
        batch.add_column(sa.Column("fov_posterior", sa.Float(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("session") as batch:
        batch.drop_column("fov_posterior")

    with op.batch_alter_table("fov") as batch:
        batch.drop_column("roi_embeddings_uri")
        batch.drop_column("fov_embedding_uri")
        batch.drop_column("fingerprint_version")
