"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-21
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "fov",
        sa.Column("fov_id", sa.CHAR(36), primary_key=True),
        sa.Column("fingerprint_hash", sa.CHAR(64), nullable=False),
        sa.Column("animal_id", sa.String(128), nullable=False),
        sa.Column("region", sa.String(64), nullable=False),
        sa.Column("mean_m_uri", sa.String(512), nullable=False),
        sa.Column("centroid_table_uri", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False),
        sa.Column("latest_session_date", sa.Date, nullable=True),
        sa.UniqueConstraint("fingerprint_hash", name="uq_fov_fingerprint_hash"),
    )
    op.create_index("ix_fov_animal_region", "fov", ["animal_id", "region"])

    op.create_table(
        "cell",
        sa.Column("global_cell_id", sa.CHAR(36), primary_key=True),
        sa.Column(
            "fov_id",
            sa.CHAR(36),
            sa.ForeignKey("fov.fov_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("first_seen_session_id", sa.CHAR(36), nullable=True),
        sa.Column("morphology_summary", sa.JSON, nullable=False),
    )
    op.create_index("ix_cell_fov_id", "cell", ["fov_id"])

    op.create_table(
        "session",
        sa.Column("session_id", sa.CHAR(36), primary_key=True),
        sa.Column(
            "fov_id",
            sa.CHAR(36),
            sa.ForeignKey("fov.fov_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_date", sa.Date, nullable=False),
        sa.Column("output_dir", sa.String(1024), nullable=False),
        sa.Column("fov_sim", sa.Float, nullable=True),
        sa.Column("n_matched", sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_new", sa.Integer, nullable=False, server_default="0"),
        sa.Column("n_missing", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime, nullable=False),
    )
    op.create_index("ix_session_fov_date", "session", ["fov_id", "session_date"])

    op.create_table(
        "cell_observation",
        sa.Column("observation_id", sa.CHAR(36), primary_key=True),
        sa.Column(
            "global_cell_id",
            sa.CHAR(36),
            sa.ForeignKey("cell.global_cell_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "session_id",
            sa.CHAR(36),
            sa.ForeignKey("session.session_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("local_label_id", sa.Integer, nullable=False),
        sa.Column("match_score", sa.Float, nullable=True),
        sa.UniqueConstraint("session_id", "local_label_id", name="uq_obs_session_label"),
    )
    op.create_index("ix_obs_cell", "cell_observation", ["global_cell_id"])


def downgrade() -> None:
    op.drop_index("ix_obs_cell", table_name="cell_observation")
    op.drop_table("cell_observation")
    op.drop_index("ix_session_fov_date", table_name="session")
    op.drop_table("session")
    op.drop_index("ix_cell_fov_id", table_name="cell")
    op.drop_table("cell")
    op.drop_index("ix_fov_animal_region", table_name="fov")
    op.drop_table("fov")
