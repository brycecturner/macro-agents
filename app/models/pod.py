import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.models.base import Base
from app.models.enums import KillAuthority, PodMembershipRole, TradingMode


class Pod(Base):
    __tablename__ = "pods"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    config: Mapped["PodConfig"] = relationship(
        "PodConfig", back_populates="pod", uselist=False
    )
    memberships: Mapped[list["PodMembership"]] = relationship(
        "PodMembership", back_populates="pod"
    )


class PodConfig(Base):
    __tablename__ = "pod_configs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pod_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pods.id"), nullable=False, unique=True
    )
    trading_mode: Mapped[TradingMode] = mapped_column(
        Enum(TradingMode, name="trading_mode_enum"), nullable=False
    )
    target_vol_per_position: Mapped[float] = mapped_column(Float, nullable=False)
    max_position_pct: Mapped[float] = mapped_column(Float, nullable=False)
    rebalance_threshold_pct: Mapped[float] = mapped_column(Float, nullable=False)
    rebalance_day: Mapped[int] = mapped_column(Integer, nullable=False)
    intake_timeout_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    kill_authority_default: Mapped[KillAuthority] = mapped_column(
        Enum(KillAuthority, name="kill_authority_enum"), nullable=False
    )
    vol_lookback_days: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    pod: Mapped[Pod] = relationship("Pod", back_populates="config")


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    memberships: Mapped[list["PodMembership"]] = relationship(
        "PodMembership", back_populates="user"
    )


class PodMembership(Base):
    __tablename__ = "pod_memberships"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    pod_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pods.id"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    role: Mapped[PodMembershipRole] = mapped_column(
        Enum(PodMembershipRole, name="pod_membership_role_enum"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    pod: Mapped[Pod] = relationship("Pod", back_populates="memberships")
    user: Mapped[User] = relationship("User", back_populates="memberships")
