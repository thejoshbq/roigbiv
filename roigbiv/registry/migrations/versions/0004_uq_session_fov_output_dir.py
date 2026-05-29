"""add unique constraint on (fov_id, output_dir) in session table

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-07

Removes any pre-existing duplicate session rows (keeping the newest by
created_at per (fov_id, output_dir) pair), then adds a UNIQUE constraint
on those two columns to prevent future duplication.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()

    # Identify duplicate session_ids to remove.  For each (fov_id, output_dir)
    # group, keep the row with the most recent created_at; delete the rest.
    result = conn.execute(sa.text(
        "SELECT session_id FROM ("
        "  SELECT session_id,"
        "         ROW_NUMBER() OVER ("
        "             PARTITION BY fov_id, output_dir"
        "             ORDER BY COALESCE(created_at, '1900-01-01') DESC"
        "         ) AS rn"
        "  FROM session"
        ") t WHERE rn > 1"
    ))
    dup_ids = [row[0] for row in result.fetchall()]
    if dup_ids:
        conn.execute(
            sa.text("DELETE FROM session WHERE session_id = :sid"),
            [{"sid": sid} for sid in dup_ids],
        )

    with op.batch_alter_table("session") as batch:
        batch.create_unique_constraint(
            "uq_session_fov_output_dir", ["fov_id", "output_dir"]
        )


def downgrade() -> None:
    with op.batch_alter_table("session") as batch:
        batch.drop_constraint("uq_session_fov_output_dir", type_="unique")
