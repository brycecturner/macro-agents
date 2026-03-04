"""Initial schema — all tables

Revision ID: 0001
Revises:
Create Date: 2026-03-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # 1. Enable pgvector extension (idempotent)
    # ------------------------------------------------------------------
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------
    # 2. Create tables (FK dependency order)
    # Enum types are created automatically by SQLAlchemy on first use
    # (create_type=True is the default). Subsequent tables that reuse
    # the same enum set create_type=False to avoid duplicate creation.
    # ------------------------------------------------------------------

    # pods — top-level entity; all other core tables reference it
    op.create_table(
        "pods",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # users
    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    # pod_configs — one row per pod, all operational parameters
    # First use of: trading_mode_enum, kill_authority_enum
    op.create_table(
        "pod_configs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "pod_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pods.id"),
            nullable=False,
        ),
        sa.Column(
            "trading_mode",
            sa.Enum("paper", "real", name="trading_mode_enum"),
            nullable=False,
        ),
        sa.Column("target_vol_per_position", sa.Float, nullable=False),
        sa.Column("max_position_pct", sa.Float, nullable=False),
        sa.Column("rebalance_threshold_pct", sa.Float, nullable=False),
        sa.Column("rebalance_day", sa.Integer, nullable=False),
        sa.Column("intake_timeout_hours", sa.Integer, nullable=False),
        sa.Column(
            "kill_authority_default",
            sa.Enum("alert_only", "auto_close", name="kill_authority_enum"),
            nullable=False,
        ),
        sa.Column("vol_lookback_days", sa.Integer, nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("pod_id", name="uq_pod_configs_pod_id"),
    )

    # pod_memberships — join table for users ↔ pods
    # First use of: pod_membership_role_enum
    op.create_table(
        "pod_memberships",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "pod_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pods.id"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column(
            "role",
            sa.Enum("pm", "analyst", "readonly", name="pod_membership_role_enum"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # theses — core idea entity; pgvector embedding for semantic search
    # First use of: direction_enum, thesis_status_enum
    # Reuses: kill_authority_enum (create_type=False)
    op.create_table(
        "theses",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "pod_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pods.id"),
            nullable=False,
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("time_horizon", sa.String(100), nullable=False),
        sa.Column(
            "direction",
            sa.Enum("long", "short", name="direction_enum"),
            nullable=False,
        ),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "draft",
                "intake_sent",
                "researched",
                "approved",
                "active",
                "closed",
                "rejected",
                name="thesis_status_enum",
            ),
            nullable=False,
        ),
        sa.Column(
            "kill_authority",
            sa.Enum(
                "alert_only",
                "auto_close",
                name="kill_authority_enum",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "intake_unconfirmed", sa.Boolean, nullable=False, server_default="false"
        ),
        # pgvector column — 1536 dimensions for semantic search
        sa.Column(
            "embedding",
            sa.Text,  # placeholder — replaced by raw SQL below
            nullable=True,
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # Drop the placeholder Text column and add the proper vector column
    op.drop_column("theses", "embedding")
    op.execute("ALTER TABLE theses ADD COLUMN embedding vector(1536)")

    # thesis_instruments — ETF(s) associated with a thesis
    # First use of: instrument_role_enum
    # Reuses: direction_enum (create_type=False)
    op.create_table(
        "thesis_instruments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "thesis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("theses.id"),
            nullable=False,
        ),
        sa.Column("instrument", sa.String(20), nullable=False),
        sa.Column(
            "direction",
            sa.Enum("long", "short", name="direction_enum", create_type=False),
            nullable=False,
        ),
        sa.Column(
            "role",
            sa.Enum("primary", "hedge", "secondary", name="instrument_role_enum"),
            nullable=False,
        ),
    )

    # falsification_conditions — kill conditions for each thesis
    # First use of: condition_type_enum, chain_operator_enum
    op.create_table(
        "falsification_conditions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "thesis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("theses.id"),
            nullable=False,
        ),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column(
            "condition_type",
            sa.Enum("state", "event", name="condition_type_enum"),
            nullable=False,
        ),
        sa.Column("trigger_type", sa.String(50), nullable=True),
        sa.Column("measurable_proxy", sa.Text, nullable=False),
        sa.Column("evaluation_logic", sa.Text, nullable=False),
        # Nullable in v1 — reserved for v2 condition chain logic
        sa.Column(
            "chain_operator",
            sa.Enum("AND", "OR", name="chain_operator_enum"),
            nullable=True,
        ),
        # Nullable in v1 — reserved for v2 condition chain grouping
        sa.Column("chain_group", sa.String(100), nullable=True),
        sa.Column("last_evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # condition_evaluations — daily sweep log per condition
    # First use of: condition_result_enum
    op.create_table(
        "condition_evaluations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "condition_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("falsification_conditions.id"),
            nullable=False,
        ),
        sa.Column(
            "thesis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("theses.id"),
            nullable=False,
        ),
        sa.Column(
            "result",
            sa.Enum("passing", "failing", "no_trigger", name="condition_result_enum"),
            nullable=False,
        ),
        sa.Column("data_point", sa.Text, nullable=True),
        sa.Column("citation", sa.Text, nullable=True),
        sa.Column(
            "evaluated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # workflow_registry — registered workflow classes
    op.create_table(
        "workflow_registry",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("name", name="uq_workflow_registry_name"),
    )

    # workflow_runs — log of all workflow executions
    # First use of: workflow_status_enum
    op.create_table(
        "workflow_runs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "thesis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("theses.id"),
            nullable=False,
        ),
        sa.Column("workflow_name", sa.String(100), nullable=False),
        sa.Column(
            "status",
            sa.Enum("completed", "failed", "partial", name="workflow_status_enum"),
            nullable=False,
        ),
        sa.Column("structured_output", postgresql.JSONB, nullable=True),
        sa.Column("citations", postgresql.JSONB, nullable=True),
        sa.Column("agent_inferences", postgresql.JSONB, nullable=True),
        sa.Column("raw_output", sa.Text, nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # further_reading — curated sources per thesis
    op.create_table(
        "further_reading",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "thesis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("theses.id"),
            nullable=False,
        ),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("url", sa.Text, nullable=False),
        sa.Column("source_type", sa.String(50), nullable=False),
        sa.Column("annotation", sa.Text, nullable=False),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column(
            "is_cited",
            sa.Boolean,
            nullable=False,
            server_default="false",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # positions — real/paper positions
    # Reuses: direction_enum, trading_mode_enum (both create_type=False)
    op.create_table(
        "positions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "pod_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pods.id"),
            nullable=False,
        ),
        sa.Column(
            "thesis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("theses.id"),
            nullable=False,
        ),
        sa.Column("instrument", sa.String(20), nullable=False),
        sa.Column(
            "direction",
            sa.Enum("long", "short", name="direction_enum", create_type=False),
            nullable=False,
        ),
        sa.Column("quantity", sa.Float, nullable=False),
        sa.Column("entry_price", sa.Float, nullable=False),
        sa.Column("current_price", sa.Float, nullable=False),
        sa.Column(
            "trading_mode",
            sa.Enum(
                "paper",
                "real",
                name="trading_mode_enum",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "opened_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # trades — execution log
    # First use of: order_type_enum, close_reason_enum
    # Reuses: direction_enum (create_type=False)
    op.create_table(
        "trades",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "pod_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pods.id"),
            nullable=False,
        ),
        sa.Column(
            "thesis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("theses.id"),
            nullable=False,
        ),
        sa.Column(
            "position_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("positions.id"),
            nullable=False,
        ),
        sa.Column("instrument", sa.String(20), nullable=False),
        sa.Column(
            "direction",
            sa.Enum("long", "short", name="direction_enum", create_type=False),
            nullable=False,
        ),
        sa.Column(
            "order_type",
            sa.Enum("limit", "market", name="order_type_enum"),
            nullable=False,
        ),
        sa.Column("submitted_price", sa.Float, nullable=False),
        sa.Column("fill_price", sa.Float, nullable=False),
        sa.Column("quantity", sa.Float, nullable=False),
        sa.Column("slippage", sa.Float, nullable=False),
        sa.Column(
            "close_reason",
            sa.Enum(
                "rebalance",
                "kill_condition",
                "auto_close",
                "human_manual",
                name="close_reason_enum",
            ),
            nullable=True,
        ),
        sa.Column("fill_time", sa.DateTime(timezone=True), nullable=False),
    )

    # portfolio_snapshots — daily pod-level performance
    op.create_table(
        "portfolio_snapshots",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "pod_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pods.id"),
            nullable=False,
        ),
        sa.Column("nav", sa.Float, nullable=False),
        sa.Column("nav_change_daily", sa.Float, nullable=False),
        sa.Column("gross_exposure", sa.Float, nullable=False),
        sa.Column("net_exposure", sa.Float, nullable=False),
        sa.Column("sharpe_30d", sa.Float, nullable=True),
        sa.Column("sharpe_90d", sa.Float, nullable=True),
        sa.Column("sharpe_1y", sa.Float, nullable=True),
        sa.Column("max_drawdown_rolling", sa.Float, nullable=True),
        sa.Column("max_drawdown_inception", sa.Float, nullable=True),
        sa.Column("snapshot_date", sa.Date, nullable=False),
    )

    # economic_calendar — scheduled macro release dates from FRED
    op.create_table(
        "economic_calendar",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("release_type", sa.String(50), nullable=False),
        sa.Column("release_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("description", sa.String(500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # news_events — unscheduled events from IBKR news + LLM classifier
    op.create_table(
        "news_events",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column("headline", sa.Text, nullable=False),
        sa.Column("trigger_type", sa.String(50), nullable=True),
        sa.Column("classifier_confidence", sa.Float, nullable=True),
        sa.Column(
            "is_override",
            sa.Boolean,
            nullable=False,
            server_default="false",
        ),
        sa.Column("event_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # alerts — alert log with delivery status
    op.create_table(
        "alerts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "pod_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pods.id"),
            nullable=False,
        ),
        sa.Column(
            "thesis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("theses.id"),
            nullable=True,
        ),
        sa.Column(
            "condition_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("falsification_conditions.id"),
            nullable=True,
        ),
        sa.Column("alert_type", sa.String(100), nullable=False),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("delivered", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("delivery_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # audit_log — append-only state change log
    # No UPDATE or DELETE permitted on this table — enforced at application layer.
    op.create_table(
        "audit_log",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "pod_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pods.id"),
            nullable=False,
        ),
        # Not a FK — entity_id + entity_type together identify what changed
        sa.Column("entity_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("entity_type", sa.String(100), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("previous_value", postgresql.JSONB, nullable=True),
        sa.Column("new_value", postgresql.JSONB, nullable=True),
        sa.Column("changed_by", sa.String(255), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # llm_usage_log — append-only Anthropic API call log
    # All FKs nullable: not every call originates from a workflow run.
    # No UPDATE or DELETE permitted — enforced at application layer.
    op.create_table(
        "llm_usage_log",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            primary_key=True,
        ),
        sa.Column(
            "pod_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("pods.id"),
            nullable=True,
        ),
        sa.Column(
            "thesis_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("theses.id"),
            nullable=True,
        ),
        sa.Column(
            "workflow_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("workflow_runs.id"),
            nullable=True,
        ),
        sa.Column("model", sa.String(100), nullable=False),
        sa.Column("task_type", sa.String(100), nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False),
        sa.Column("output_tokens", sa.Integer, nullable=False),
        sa.Column("estimated_cost_usd", sa.Float, nullable=False),
        sa.Column(
            "called_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    # Drop tables in reverse FK dependency order
    op.drop_table("llm_usage_log")
    op.drop_table("audit_log")
    op.drop_table("alerts")
    op.drop_table("news_events")
    op.drop_table("economic_calendar")
    op.drop_table("portfolio_snapshots")
    op.drop_table("trades")
    op.drop_table("positions")
    op.drop_table("further_reading")
    op.drop_table("workflow_runs")
    op.drop_table("workflow_registry")
    op.drop_table("condition_evaluations")
    op.drop_table("falsification_conditions")
    op.drop_table("thesis_instruments")
    op.drop_table("theses")
    op.drop_table("pod_memberships")
    op.drop_table("pod_configs")
    op.drop_table("users")
    op.drop_table("pods")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS close_reason_enum")
    op.execute("DROP TYPE IF EXISTS order_type_enum")
    op.execute("DROP TYPE IF EXISTS workflow_status_enum")
    op.execute("DROP TYPE IF EXISTS condition_result_enum")
    op.execute("DROP TYPE IF EXISTS chain_operator_enum")
    op.execute("DROP TYPE IF EXISTS condition_type_enum")
    op.execute("DROP TYPE IF EXISTS instrument_role_enum")
    op.execute("DROP TYPE IF EXISTS direction_enum")
    op.execute("DROP TYPE IF EXISTS thesis_status_enum")
    op.execute("DROP TYPE IF EXISTS pod_membership_role_enum")
    op.execute("DROP TYPE IF EXISTS kill_authority_enum")
    op.execute("DROP TYPE IF EXISTS trading_mode_enum")
