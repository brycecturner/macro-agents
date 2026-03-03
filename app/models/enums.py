import enum


class TradingMode(enum.Enum):
    paper = "paper"
    real = "real"


class KillAuthority(enum.Enum):
    alert_only = "alert_only"
    auto_close = "auto_close"


class PodMembershipRole(enum.Enum):
    pm = "pm"
    analyst = "analyst"
    readonly = "readonly"


class ThesisStatus(enum.Enum):
    draft = "draft"
    intake_sent = "intake_sent"
    researched = "researched"
    approved = "approved"
    active = "active"
    closed = "closed"
    rejected = "rejected"


class Direction(enum.Enum):
    long = "long"
    short = "short"


class InstrumentRole(enum.Enum):
    primary = "primary"
    hedge = "hedge"
    secondary = "secondary"


class ConditionType(enum.Enum):
    state = "state"
    event = "event"


class ChainOperator(enum.Enum):
    AND = "AND"
    OR = "OR"


class ConditionResult(enum.Enum):
    passing = "passing"
    failing = "failing"
    no_trigger = "no_trigger"


class WorkflowStatus(enum.Enum):
    completed = "completed"
    failed = "failed"
    partial = "partial"


class OrderType(enum.Enum):
    limit = "limit"
    market = "market"


class CloseReason(enum.Enum):
    rebalance = "rebalance"
    kill_condition = "kill_condition"
    auto_close = "auto_close"
    human_manual = "human_manual"
