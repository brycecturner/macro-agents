"""Add intake fields to theses table

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Rename intake_unconfirmed → thesis_confirmed and invert semantics.
    # Old default False meant "no problem"; new default True means the same
    # thing without a double-negative.
    op.alter_column("theses", "intake_unconfirmed", new_column_name="thesis_confirmed")
    op.execute("UPDATE theses SET thesis_confirmed = NOT thesis_confirmed")

    op.add_column("theses", sa.Column("intake_message", sa.Text(), nullable=True))
    op.add_column(
        "theses",
        sa.Column("intake_sent_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("theses", sa.Column("intake_user_response", sa.Text(), nullable=True))
    op.add_column(
        "theses",
        sa.Column("intake_responded_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("theses", "intake_responded_at")
    op.drop_column("theses", "intake_user_response")
    op.drop_column("theses", "intake_sent_at")
    op.drop_column("theses", "intake_message")

    op.execute("UPDATE theses SET thesis_confirmed = NOT thesis_confirmed")
    op.alter_column("theses", "thesis_confirmed", new_column_name="intake_unconfirmed")
