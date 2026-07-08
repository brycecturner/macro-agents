from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base
from app.models.enums import (
    ChainOperator,
    ConditionResult,
    ConditionType,
    Direction,
    InstrumentRole,
    KillAuthority,
    ThesisStatus,
)

if TYPE_CHECKING:
    from app.models.workflow import WorkflowRun

# Embedding dimension for pgvector semantic search.
# 1536 is compatible with common embedding models.
EMBEDDING_DIM = 1536


class Thesis(Base):
    __tablename__ = "theses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pod_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pods.id"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    time_horizon: Mapped[str] = mapped_column(String(100), nullable=False)
    direction: Mapped[Direction] = mapped_column(
        Enum(Direction, name="direction_enum"), nullable=False
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[ThesisStatus] = mapped_column(
        Enum(ThesisStatus, name="thesis_status_enum"), nullable=False
    )
    kill_authority: Mapped[KillAuthority] = mapped_column(
        Enum(KillAuthority, name="kill_authority_enum"),
        nullable=False,
    )
    # False when the intake timed out and the agent proceeded without user
    # confirmation. Persists until the user explicitly acknowledges it in the UI.
    thesis_confirmed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    # Intake agent fields — populated after the LLM generates the intake message
    intake_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    intake_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    intake_user_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    intake_responded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # pgvector embedding for semantic thesis search (nullable — generated after intake)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )
    # Assembled Tier 1 trade brief (PRD Section 4.4), stored as JSON and
    # regenerated whenever the research pipeline completes. Any is unavoidable:
    # the brief shape is assembled from multiple workflows' typed outputs.
    brief: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    brief_generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    instruments: Mapped[list[ThesisInstrument]] = relationship(
        "ThesisInstrument", back_populates="thesis"
    )
    falsification_conditions: Mapped[list[FalsificationCondition]] = relationship(
        "FalsificationCondition", back_populates="thesis"
    )
    condition_evaluations: Mapped[list[ConditionEvaluation]] = relationship(
        "ConditionEvaluation", back_populates="thesis"
    )
    workflow_runs: Mapped[list[WorkflowRun]] = relationship(
        "WorkflowRun", back_populates="thesis"
    )


class ThesisInstrument(Base):
    __tablename__ = "thesis_instruments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thesis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("theses.id"), nullable=False
    )
    instrument: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[Direction] = mapped_column(
        Enum(Direction, name="direction_enum"), nullable=False
    )
    role: Mapped[InstrumentRole] = mapped_column(
        Enum(InstrumentRole, name="instrument_role_enum"), nullable=False
    )

    thesis: Mapped[Thesis] = relationship("Thesis", back_populates="instruments")


class FalsificationCondition(Base):
    __tablename__ = "falsification_conditions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    thesis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("theses.id"), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    condition_type: Mapped[ConditionType] = mapped_column(
        Enum(ConditionType, name="condition_type_enum"), nullable=False
    )
    # Only populated for event-type conditions — maps to an event detection mechanism
    trigger_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    measurable_proxy: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation_logic: Mapped[str] = mapped_column(Text, nullable=False)
    # Reserved for v2 condition chain logic — nullable in v1
    chain_operator: Mapped[ChainOperator | None] = mapped_column(
        Enum(ChainOperator, name="chain_operator_enum"), nullable=True
    )
    # Reserved for v2 condition chain grouping — nullable in v1
    chain_group: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_evaluated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    thesis: Mapped[Thesis] = relationship(
        "Thesis", back_populates="falsification_conditions"
    )
    evaluations: Mapped[list[ConditionEvaluation]] = relationship(
        "ConditionEvaluation", back_populates="condition"
    )


class ConditionEvaluation(Base):
    __tablename__ = "condition_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    condition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("falsification_conditions.id"), nullable=False
    )
    # Denormalized for query convenience — thesis_id is available via condition,
    # but including it here avoids a join on the hot daily sweep path
    thesis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("theses.id"), nullable=False
    )
    result: Mapped[ConditionResult] = mapped_column(
        Enum(ConditionResult, name="condition_result_enum"), nullable=False
    )
    # The specific data point that was checked (e.g., "10Y yield: 4.52%")
    data_point: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full citation for the data source used in this evaluation
    citation: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    condition: Mapped[FalsificationCondition] = relationship(
        "FalsificationCondition", back_populates="evaluations"
    )
    thesis: Mapped[Thesis] = relationship(
        "Thesis", back_populates="condition_evaluations"
    )
