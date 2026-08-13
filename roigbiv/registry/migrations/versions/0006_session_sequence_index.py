"""add session.sequence_index for human-set chronological ordering

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-10

Adds:
  * session.sequence_index (nullable) — the position a human assigned this
    session in its FOV's timeline, via the Track page's reorder step.

``session_date`` alone cannot order a timeline. Six-digit filename dates are
ambiguous between the lab's two conventions (see registry/filename.py), and
several sessions routinely share one date — the reference prism workspace has
``pre-005`` / ``beh-006`` / ``post-007`` all recorded on the same day, in a
sequence no date can express.

Additive only: existing rows keep NULL and continue to sort by session_date.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("session") as batch:
        batch.add_column(sa.Column("sequence_index", sa.Integer, nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("session") as batch:
        batch.drop_column("sequence_index")
