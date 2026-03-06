"""fix economic_calendar schema

Renames release_date → scheduled_date (DateTime → Date), drops description,
adds actual_date (Date, nullable), and adds a unique constraint on
(release_type, scheduled_date) to enforce idempotent syncs.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-05
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns (nullable initially so existing rows don't violate constraints)
    op.add_column(
        "economic_calendar",
        sa.Column("scheduled_date", sa.Date(), nullable=True),
    )
    op.add_column(
        "economic_calendar",
        sa.Column("actual_date", sa.Date(), nullable=True),
    )

    # Migrate existing data: cast release_date (DateTime) to Date
    op.execute("UPDATE economic_calendar SET scheduled_date = release_date::date")

    # Now enforce non-nullable on scheduled_date
    op.alter_column("economic_calendar", "scheduled_date", nullable=False)

    # Drop old columns
    op.drop_column("economic_calendar", "release_date")
    op.drop_column("economic_calendar", "description")

    # Unique constraint to support idempotent syncs
    op.create_unique_constraint(
        "uq_economic_calendar_type_date",
        "economic_calendar",
        ["release_type", "scheduled_date"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_economic_calendar_type_date", "economic_calendar", type_="unique"
    )

    op.add_column(
        "economic_calendar",
        sa.Column(
            "release_date",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "economic_calendar",
        sa.Column("description", sa.String(500), nullable=True),
    )

    op.execute(
        "UPDATE economic_calendar"
        " SET release_date = scheduled_date::timestamp with time zone"
    )

    op.alter_column("economic_calendar", "release_date", nullable=False)

    op.drop_column("economic_calendar", "actual_date")
    op.drop_column("economic_calendar", "scheduled_date")
