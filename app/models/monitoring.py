import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.models.base import Base


class EconomicCalendar(Base):
    """Scheduled macro release dates populated from the FRED release calendar.

    Used by the daily monitoring job to determine whether a scheduled event
    (CPI, FOMC, NFP, etc.) has occurred since a condition was last evaluated.

    Supported release types: CPI_RELEASE, FOMC_DECISION, NFP_RELEASE,
    PMI_RELEASE, GDP_RELEASE, PCE_RELEASE
    """

    __tablename__ = "economic_calendar"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    release_type: Mapped[str] = mapped_column(String(50), nullable=False)
    release_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class NewsEvent(Base):
    """Unscheduled events detected by the IBKR news feed + LLM classifier.

    The daily monitoring job polls IBKR news headlines. A lightweight LLM
    classifier determines whether any headline matches an unscheduled trigger
    type on an active condition. Users can override misclassifications.

    Supported trigger types: TARIFF_ANNOUNCEMENT, FED_SPEECH,
    GEOPOLITICAL_EVENT, SURPRISE_RATE_MOVE
    """

    __tablename__ = "news_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    headline: Mapped[str] = mapped_column(Text, nullable=False)
    # The trigger type matched by the classifier — nullable if classifier did not fire
    trigger_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    classifier_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # True when a user has manually overridden the classifier's classification
    is_override: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    event_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Alert(Base):
    """Log of all system alerts with delivery status.

    Email is the primary delivery channel. All actions (acknowledging alerts,
    changing kill authority, closing trades) go through the UI — users never
    reply to alert emails to take action.
    """

    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pod_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pods.id"), nullable=False
    )
    # Nullable — some alerts are pod-level (e.g., mode switch failures)
    thesis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("theses.id"), nullable=True
    )
    # Nullable — not all alerts are tied to a specific condition
    condition_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("falsification_conditions.id"), nullable=True
    )
    # e.g. 'condition_falsified', 'intake_timeout', 'mode_switch_failed',
    # 'order_rejected', 'trade_confirmed'
    alert_type: Mapped[str] = mapped_column(String(100), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    delivered: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    delivery_timestamp: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
