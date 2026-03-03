"""Unit tests for SQLAlchemy ORM models.

These tests verify model structure, enum correctness, and table naming without
requiring a live database connection. No external calls are made.
"""

from app.models import (
    Alert,
    AuditLog,
    Base,
    ConditionEvaluation,
    EconomicCalendar,
    FalsificationCondition,
    FurtherReading,
    LLMUsageLog,
    NewsEvent,
    Pod,
    PodConfig,
    PodMembership,
    PortfolioSnapshot,
    Position,
    Thesis,
    ThesisInstrument,
    Trade,
    User,
    WorkflowRegistry,
    WorkflowRun,
)
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


class TestEnums:
    def test_trading_mode_values(self):
        assert TradingMode.paper.value == "paper"
        assert TradingMode.real.value == "real"

    def test_kill_authority_values(self):
        assert KillAuthority.alert_only.value == "alert_only"
        assert KillAuthority.auto_close.value == "auto_close"

    def test_pod_membership_role_values(self):
        assert PodMembershipRole.pm.value == "pm"
        assert PodMembershipRole.analyst.value == "analyst"
        assert PodMembershipRole.readonly.value == "readonly"

    def test_thesis_status_values(self):
        expected = {
            "draft",
            "intake_sent",
            "researched",
            "approved",
            "active",
            "closed",
            "rejected",
        }
        assert {s.value for s in ThesisStatus} == expected

    def test_direction_values(self):
        assert Direction.long.value == "long"
        assert Direction.short.value == "short"

    def test_instrument_role_values(self):
        assert InstrumentRole.primary.value == "primary"
        assert InstrumentRole.hedge.value == "hedge"
        assert InstrumentRole.secondary.value == "secondary"

    def test_condition_type_values(self):
        assert ConditionType.state.value == "state"
        assert ConditionType.event.value == "event"

    def test_chain_operator_values(self):
        assert ChainOperator.AND.value == "AND"
        assert ChainOperator.OR.value == "OR"

    def test_condition_result_values(self):
        assert ConditionResult.passing.value == "passing"
        assert ConditionResult.failing.value == "failing"
        assert ConditionResult.no_trigger.value == "no_trigger"

    def test_workflow_status_values(self):
        assert WorkflowStatus.completed.value == "completed"
        assert WorkflowStatus.failed.value == "failed"
        assert WorkflowStatus.partial.value == "partial"

    def test_order_type_values(self):
        assert OrderType.limit.value == "limit"
        assert OrderType.market.value == "market"

    def test_close_reason_values(self):
        assert CloseReason.rebalance.value == "rebalance"
        assert CloseReason.kill_condition.value == "kill_condition"
        assert CloseReason.auto_close.value == "auto_close"
        assert CloseReason.human_manual.value == "human_manual"


class TestTableNames:
    def test_pod_table_name(self):
        assert Pod.__tablename__ == "pods"

    def test_pod_config_table_name(self):
        assert PodConfig.__tablename__ == "pod_configs"

    def test_user_table_name(self):
        assert User.__tablename__ == "users"

    def test_pod_membership_table_name(self):
        assert PodMembership.__tablename__ == "pod_memberships"

    def test_thesis_table_name(self):
        assert Thesis.__tablename__ == "theses"

    def test_thesis_instrument_table_name(self):
        assert ThesisInstrument.__tablename__ == "thesis_instruments"

    def test_falsification_condition_table_name(self):
        assert FalsificationCondition.__tablename__ == "falsification_conditions"

    def test_condition_evaluation_table_name(self):
        assert ConditionEvaluation.__tablename__ == "condition_evaluations"

    def test_workflow_registry_table_name(self):
        assert WorkflowRegistry.__tablename__ == "workflow_registry"

    def test_workflow_run_table_name(self):
        assert WorkflowRun.__tablename__ == "workflow_runs"

    def test_further_reading_table_name(self):
        assert FurtherReading.__tablename__ == "further_reading"

    def test_position_table_name(self):
        assert Position.__tablename__ == "positions"

    def test_trade_table_name(self):
        assert Trade.__tablename__ == "trades"

    def test_portfolio_snapshot_table_name(self):
        assert PortfolioSnapshot.__tablename__ == "portfolio_snapshots"

    def test_economic_calendar_table_name(self):
        assert EconomicCalendar.__tablename__ == "economic_calendar"

    def test_news_event_table_name(self):
        assert NewsEvent.__tablename__ == "news_events"

    def test_alert_table_name(self):
        assert Alert.__tablename__ == "alerts"

    def test_audit_log_table_name(self):
        assert AuditLog.__tablename__ == "audit_log"

    def test_llm_usage_log_table_name(self):
        assert LLMUsageLog.__tablename__ == "llm_usage_log"


class TestMetadataRegistration:
    """Verify all tables are registered in Base.metadata."""

    def test_all_tables_registered(self):
        registered = set(Base.metadata.tables.keys())
        expected = {
            "pods",
            "pod_configs",
            "users",
            "pod_memberships",
            "theses",
            "thesis_instruments",
            "falsification_conditions",
            "condition_evaluations",
            "workflow_registry",
            "workflow_runs",
            "further_reading",
            "positions",
            "trades",
            "portfolio_snapshots",
            "economic_calendar",
            "news_events",
            "alerts",
            "audit_log",
            "llm_usage_log",
        }
        assert expected.issubset(registered)


class TestPodConfigDefaults:
    """Verify pod_configs column definitions match PRD Section 9.2 defaults."""

    def test_pod_config_columns_exist(self):
        cols = {c.name for c in PodConfig.__table__.columns}
        assert "trading_mode" in cols
        assert "target_vol_per_position" in cols
        assert "max_position_pct" in cols
        assert "rebalance_threshold_pct" in cols
        assert "rebalance_day" in cols
        assert "intake_timeout_hours" in cols
        assert "kill_authority_default" in cols
        assert "vol_lookback_days" in cols


class TestFalsificationConditionChainFields:
    """chain_operator and chain_group must exist and be nullable (v2 reserved)."""

    def test_chain_operator_column_is_nullable(self):
        col = FalsificationCondition.__table__.c.chain_operator
        assert col.nullable is True

    def test_chain_group_column_is_nullable(self):
        col = FalsificationCondition.__table__.c.chain_group
        assert col.nullable is True


class TestThesisEmbeddingColumn:
    """theses.embedding must exist for pgvector semantic search."""

    def test_embedding_column_exists(self):
        cols = {c.name for c in Thesis.__table__.columns}
        assert "embedding" in cols

    def test_embedding_column_is_nullable(self):
        col = Thesis.__table__.c.embedding
        assert col.nullable is True


class TestUUIDPrimaryKeys:
    """All models must use UUID primary keys."""

    def _get_pk_col(self, model):
        return next(c for c in model.__table__.columns if c.primary_key)

    def test_pod_pk_is_uuid(self):
        pk = self._get_pk_col(Pod)
        assert pk.name == "id"

    def test_thesis_pk_is_uuid(self):
        pk = self._get_pk_col(Thesis)
        assert pk.name == "id"

    def test_workflow_run_pk_is_uuid(self):
        pk = self._get_pk_col(WorkflowRun)
        assert pk.name == "id"

    def test_audit_log_pk_is_uuid(self):
        pk = self._get_pk_col(AuditLog)
        assert pk.name == "id"

    def test_llm_usage_log_pk_is_uuid(self):
        pk = self._get_pk_col(LLMUsageLog)
        assert pk.name == "id"


class TestSeedDefaults:
    """Verify seed script default values match PRD Section 9.2."""

    def test_seed_defaults_match_prd(self):
        from scripts.seed import POD_CONFIG_DEFAULTS

        assert POD_CONFIG_DEFAULTS["trading_mode"] == TradingMode.paper
        assert POD_CONFIG_DEFAULTS["target_vol_per_position"] == 0.05
        assert POD_CONFIG_DEFAULTS["max_position_pct"] == 0.25
        assert POD_CONFIG_DEFAULTS["rebalance_threshold_pct"] == 0.01
        assert POD_CONFIG_DEFAULTS["rebalance_day"] == 0
        assert POD_CONFIG_DEFAULTS["intake_timeout_hours"] == 24
        assert POD_CONFIG_DEFAULTS["kill_authority_default"] == KillAuthority.alert_only
        assert POD_CONFIG_DEFAULTS["vol_lookback_days"] == 60
