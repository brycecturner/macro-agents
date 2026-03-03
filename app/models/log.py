from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.workflow import WorkflowRun


class AuditLog(Base):
    """Append-only log of all state changes on tracked entities.

    Every write to a tracked entity (thesis status, kill_authority,
    trading_mode, go/no-go decisions) must also write to this table in
    the same transaction. No UPDATE or DELETE operations are ever permitted.
    """

    __tablename__ = "audit_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pod_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pods.id"), nullable=False
    )
    # UUID of the entity that changed — not a FK since it may reference any table
    entity_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    # e.g. 'thesis', 'pod_configs', 'position'
    entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
    # e.g. 'mode_switch_attempted', 'cash_check_passed', 'thesis_status_changed'
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    # Any is unavoidable: previous/new values vary by entity type
    previous_value: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    new_value: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # User UUID string or agent identifier (e.g. 'kill_authority_agent')
    changed_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class LLMUsageLog(Base):
    """Append-only log of every Anthropic API call with token counts and cost.

    Records are written by AnthropicClient after every API call — no other
    code writes to this table. Never updated or deleted. Designed to support
    a cost dashboard aggregated by model, task_type, thesis, date, and pod in v2.
    """

    __tablename__ = "llm_usage_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Nullable — not all LLM calls originate from a workflow run
    # (e.g. intake conversation, event classifier)
    pod_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pods.id"), nullable=True
    )
    thesis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("theses.id"), nullable=True
    )
    workflow_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workflow_runs.id"), nullable=True
    )
    # Exact model string used, e.g. 'claude-opus-4-6'
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    # Human-readable label, e.g. 'intake', 'macro_context', 'event_classifier'
    task_type: Mapped[str] = mapped_column(String(100), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    # Computed at log time from MODEL_PRICING constants — reflects price paid,
    # not current pricing
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    called_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    workflow_run: Mapped[WorkflowRun | None] = relationship(
        "WorkflowRun", back_populates="llm_usage_logs"
    )
