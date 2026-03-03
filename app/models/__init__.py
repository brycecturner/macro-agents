# Import all models here so that:
# 1. SQLAlchemy's mapper can resolve all forward references in relationships
# 2. Alembic's env.py can import this module and see the full metadata

from app.models.base import Base
from app.models.enums import (
    ChainOperator,
    CloseReason,
    ConditionResult,
    ConditionType,
    Direction,
    InstrumentRole,
    KillAuthority,
    OrderType,
    PodMembershipRole,
    ThesisStatus,
    TradingMode,
    WorkflowStatus,
)
from app.models.execution import PortfolioSnapshot, Position, Trade
from app.models.log import AuditLog, LLMUsageLog
from app.models.monitoring import Alert, EconomicCalendar, NewsEvent
from app.models.pod import Pod, PodConfig, PodMembership, User
from app.models.thesis import (
    ConditionEvaluation,
    FalsificationCondition,
    Thesis,
    ThesisInstrument,
)
from app.models.workflow import FurtherReading, WorkflowRegistry, WorkflowRun

__all__ = [
    "Base",
    # Enums
    "ChainOperator",
    "CloseReason",
    "ConditionResult",
    "ConditionType",
    "Direction",
    "InstrumentRole",
    "KillAuthority",
    "OrderType",
    "PodMembershipRole",
    "ThesisStatus",
    "TradingMode",
    "WorkflowStatus",
    # Models
    "Alert",
    "AuditLog",
    "ConditionEvaluation",
    "EconomicCalendar",
    "FalsificationCondition",
    "FurtherReading",
    "LLMUsageLog",
    "NewsEvent",
    "Pod",
    "PodConfig",
    "PodMembership",
    "PortfolioSnapshot",
    "Position",
    "Thesis",
    "ThesisInstrument",
    "Trade",
    "User",
    "WorkflowRegistry",
    "WorkflowRun",
]
