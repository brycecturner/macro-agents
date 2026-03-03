from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base
from app.models.enums import WorkflowStatus

if TYPE_CHECKING:
    from app.models.log import LLMUsageLog
    from app.models.thesis import Thesis


class WorkflowRegistry(Base):
    """Registry of all available workflow classes.

    Populated by the seed script or application startup. Workflows are
    registered by name — the name must match the Python class name exactly.
    """

    __tablename__ = "workflow_registry"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thesis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("theses.id"), nullable=False
    )
    workflow_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[WorkflowStatus] = mapped_column(
        Enum(WorkflowStatus, name="workflow_status_enum"), nullable=False
    )
    # Typed structured output — schema varies per workflow; nullable on failure
    # Any is unavoidable here: each workflow defines its own output schema
    structured_output: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    # List of Citation objects serialized as JSONB
    citations: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    # List of agent inference strings — explicitly labeled, not data-backed claims
    agent_inferences: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    # Full raw output stored for debugging — not passed between workflow steps
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    thesis: Mapped[Thesis] = relationship("Thesis", back_populates="workflow_runs")
    llm_usage_logs: Mapped[list[LLMUsageLog]] = relationship(
        "LLMUsageLog", back_populates="workflow_run"
    )


class FurtherReading(Base):
    """Curated sources surfaced during WebResearchWorkflow, ranked and annotated.

    3–5 entries per thesis. Cited sources (directly referenced in the brief body)
    are ranked first; additional relevant sources follow. Ties broken by source quality.
    """

    __tablename__ = "further_reading"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thesis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("theses.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    # 'web' | 'FRED series' | 'academic paper'
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # One sentence written by the agent explaining relevance to the thesis
    annotation: Mapped[str] = mapped_column(Text, nullable=False)
    # 1-based rank; lower = displayed higher in the brief
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    # True if directly cited in the brief body; False if additional relevant source
    is_cited: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    thesis: Mapped[Thesis] = relationship("Thesis")
