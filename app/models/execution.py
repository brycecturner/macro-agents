import uuid
from datetime import date, datetime

from sqlalchemy import Date, DateTime, Enum, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base
from app.models.enums import CloseReason, Direction, OrderType, TradingMode


class Position(Base):
    __tablename__ = "positions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pod_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pods.id"), nullable=False
    )
    thesis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("theses.id"), nullable=False
    )
    instrument: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[Direction] = mapped_column(
        Enum(Direction, name="direction_enum"), nullable=False
    )
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    # Snapshot of trading_mode at position open time — not a live reference to
    # pod_configs. Records which account the position lives in.
    trading_mode: Mapped[TradingMode] = mapped_column(
        Enum(TradingMode, name="trading_mode_enum"), nullable=False
    )
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    trades: Mapped[list["Trade"]] = relationship("Trade", back_populates="position")


class Trade(Base):
    __tablename__ = "trades"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pod_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pods.id"), nullable=False
    )
    thesis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("theses.id"), nullable=False
    )
    position_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("positions.id"), nullable=False
    )
    instrument: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[Direction] = mapped_column(
        Enum(Direction, name="direction_enum"), nullable=False
    )
    order_type: Mapped[OrderType] = mapped_column(
        Enum(OrderType, name="order_type_enum"), nullable=False
    )
    submitted_price: Mapped[float] = mapped_column(Float, nullable=False)
    fill_price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    slippage: Mapped[float] = mapped_column(Float, nullable=False)
    # Nullable — only populated on closing trades, not opening trades
    close_reason: Mapped[CloseReason | None] = mapped_column(
        Enum(CloseReason, name="close_reason_enum"), nullable=True
    )
    fill_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    position: Mapped[Position] = relationship("Position", back_populates="trades")


class PortfolioSnapshot(Base):
    """Daily pod-level performance snapshot, computed at market close."""

    __tablename__ = "portfolio_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pod_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pods.id"), nullable=False
    )
    nav: Mapped[float] = mapped_column(Float, nullable=False)
    nav_change_daily: Mapped[float] = mapped_column(Float, nullable=False)
    gross_exposure: Mapped[float] = mapped_column(Float, nullable=False)
    net_exposure: Mapped[float] = mapped_column(Float, nullable=False)
    # Rolling Sharpe ratios — nullable until enough history accumulates
    sharpe_30d: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_90d: Mapped[float | None] = mapped_column(Float, nullable=True)
    sharpe_1y: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown_rolling: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown_inception: Mapped[float | None] = mapped_column(Float, nullable=True)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
