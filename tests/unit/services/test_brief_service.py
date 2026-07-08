"""Tests for BriefService — Tier 1 trade brief assembly and storage."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock

from app.models.enums import ConditionType, Direction
from app.models.thesis import FalsificationCondition
from app.models.workflow import WorkflowRun
from app.services.brief_service import assemble_brief, store_brief

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thesis(**kwargs) -> MagicMock:
    t = MagicMock()
    t.id = kwargs.get("id", uuid.uuid4())
    t.title = kwargs.get("title", "Yield Curve Steepener")
    t.direction = kwargs.get("direction", Direction.long)
    t.time_horizon = kwargs.get("time_horizon", "6 months")
    t.instruments = kwargs.get("instruments", [])
    return t


def _make_instrument(instrument: str, role: str = "primary") -> MagicMock:
    instr = MagicMock()
    instr.instrument = instrument
    instr.role = MagicMock()
    instr.role.value = role
    return instr


def _make_run(
    workflow_name: str,
    structured_output: dict | None = None,
    citations: list[dict] | None = None,
    started_at: datetime | None = None,
) -> MagicMock:
    run = MagicMock(spec=WorkflowRun)
    run.workflow_name = workflow_name
    run.structured_output = structured_output or {}
    run.citations = citations or []
    run.started_at = started_at or datetime.now(UTC)
    return run


def _make_condition(
    description: str = "10Y yield above 4.5%",
    condition_type: ConditionType = ConditionType.state,
    trigger_type: str | None = None,
    measurable_proxy: str = "FRED:DGS10",
    evaluation_logic: str = "> 4.5",
) -> MagicMock:
    c = MagicMock(spec=FalsificationCondition)
    c.id = uuid.uuid4()
    c.description = description
    c.condition_type = condition_type
    c.trigger_type = trigger_type
    c.measurable_proxy = measurable_proxy
    c.evaluation_logic = evaluation_logic
    c.created_at = datetime.now(UTC)
    return c


def _configure_db(
    db: MagicMock,
    *,
    runs: list[MagicMock] | None = None,
    conditions: list[MagicMock] | None = None,
) -> None:
    def _query(model: type) -> MagicMock:
        q = MagicMock()
        if model is WorkflowRun:
            q.filter.return_value.order_by.return_value.all.return_value = runs or []
        elif model is FalsificationCondition:
            q.filter.return_value.order_by.return_value.all.return_value = (
                conditions or []
            )
        return q

    db.query.side_effect = _query


# ---------------------------------------------------------------------------
# assemble_brief
# ---------------------------------------------------------------------------


class TestAssembleBrief:
    def test_includes_thesis_identity_fields(self):
        thesis = _make_thesis(title="Inflation Breakout", time_horizon="1 year")
        db = MagicMock()
        _configure_db(db)

        brief = assemble_brief(thesis, db)

        assert brief["thesis_id"] == str(thesis.id)
        assert brief["title"] == "Inflation Breakout"
        assert brief["direction"] == "long"
        assert brief["time_horizon"] == "1 year"

    def test_summary_from_macro_context_workflow(self):
        thesis = _make_thesis()
        db = MagicMock()
        run = _make_run(
            "MacroContextWorkflow", structured_output={"summary": "Macro backdrop."}
        )
        _configure_db(db, runs=[run])

        brief = assemble_brief(thesis, db)

        assert brief["summary"] == "Macro backdrop."

    def test_summary_empty_when_macro_context_missing(self):
        thesis = _make_thesis()
        db = MagicMock()
        _configure_db(db)

        brief = assemble_brief(thesis, db)

        assert brief["summary"] == ""

    def test_instrument_from_thesis_instruments_when_present(self):
        thesis = _make_thesis(instruments=[_make_instrument("TLT")])
        db = MagicMock()
        _configure_db(db)

        brief = assemble_brief(thesis, db)

        assert brief["instrument"] == "TLT"

    def test_instrument_falls_back_to_backtest_output(self):
        thesis = _make_thesis(instruments=[])
        db = MagicMock()
        run = _make_run("BacktestWorkflow", structured_output={"instrument": "GLD"})
        _configure_db(db, runs=[run])

        brief = assemble_brief(thesis, db)

        assert brief["instrument"] == "GLD"

    def test_instrument_prefers_primary_role(self):
        thesis = _make_thesis(
            instruments=[
                _make_instrument("VIXY", role="hedge"),
                _make_instrument("SPY", role="primary"),
            ]
        )
        db = MagicMock()
        _configure_db(db)

        brief = assemble_brief(thesis, db)

        assert brief["instrument"] == "SPY"

    def test_backtest_stats_populated_from_backtest_workflow(self):
        thesis = _make_thesis()
        db = MagicMock()
        run = _make_run(
            "BacktestWorkflow",
            structured_output={
                "label": "Historical Analog Analysis",
                "instrument": "TLT",
                "aggregate": {
                    "n_periods": 3,
                    "avg_return": 0.05,
                    "worst_return": -0.02,
                    "best_return": 0.12,
                    "win_rate": 0.67,
                    "avg_max_drawdown": -0.04,
                    "statistical_limitation_note": "Small sample.",
                },
                "benchmark_comparison": {"spy": {"avg_return": 0.03}},
                "analysis": "Instrument rallied in prior analogs.",
            },
        )
        _configure_db(db, runs=[run])

        brief = assemble_brief(thesis, db)
        stats = brief["backtest_stats"]

        assert stats["label"] == "Historical Analog Analysis"
        assert stats["n_periods"] == 3
        assert stats["avg_return"] == 0.05
        assert stats["worst_return"] == -0.02
        assert stats["best_return"] == 0.12
        assert stats["win_rate"] == 0.67
        assert stats["avg_max_drawdown"] == -0.04
        assert stats["statistical_limitation_note"] == "Small sample."
        assert stats["benchmark_comparison"] == {"spy": {"avg_return": 0.03}}
        assert stats["analysis"] == "Instrument rallied in prior analogs."

    def test_backtest_stats_defaults_when_workflow_missing(self):
        thesis = _make_thesis()
        db = MagicMock()
        _configure_db(db)

        brief = assemble_brief(thesis, db)
        stats = brief["backtest_stats"]

        assert stats["label"] == "Historical Analog Analysis"
        assert stats["n_periods"] == 0
        assert stats["avg_return"] is None

    def test_assumptions_from_recommendation_workflow(self):
        thesis = _make_thesis()
        db = MagicMock()
        run = _make_run(
            "RecommendationWorkflow",
            structured_output={"key_assumptions": ["Fed cuts continue"]},
        )
        _configure_db(db, runs=[run])

        brief = assemble_brief(thesis, db)

        assert brief["assumptions"] == ["Fed cuts continue"]

    def test_recommendation_fields_populated(self):
        thesis = _make_thesis()
        db = MagicMock()
        run = _make_run(
            "RecommendationWorkflow",
            structured_output={
                "recommendation": "go",
                "rationale": "Evidence is supportive.",
                "confidence_level": "high",
            },
        )
        _configure_db(db, runs=[run])

        brief = assemble_brief(thesis, db)
        rec = brief["recommendation"]

        assert rec["recommendation"] == "go"
        assert rec["rationale"] == "Evidence is supportive."
        assert rec["confidence_level"] == "high"

    def test_recommendation_defaults_when_workflow_missing(self):
        thesis = _make_thesis()
        db = MagicMock()
        _configure_db(db)

        brief = assemble_brief(thesis, db)
        rec = brief["recommendation"]

        assert rec["recommendation"] is None
        assert rec["rationale"] == ""
        assert rec["confidence_level"] is None

    def test_falsification_conditions_included(self):
        thesis = _make_thesis()
        db = MagicMock()
        condition = _make_condition(
            description="10Y yield above 4.5%",
            condition_type=ConditionType.state,
            trigger_type=None,
        )
        _configure_db(db, conditions=[condition])

        brief = assemble_brief(thesis, db)

        assert len(brief["falsification_conditions"]) == 1
        entry = brief["falsification_conditions"][0]
        assert entry["id"] == str(condition.id)
        assert entry["description"] == "10Y yield above 4.5%"
        assert entry["condition_type"] == "state"
        assert entry["trigger_type"] is None

    def test_event_condition_includes_trigger_type(self):
        thesis = _make_thesis()
        db = MagicMock()
        condition = _make_condition(
            condition_type=ConditionType.event, trigger_type="CPI_RELEASE"
        )
        _configure_db(db, conditions=[condition])

        brief = assemble_brief(thesis, db)

        entry = brief["falsification_conditions"][0]
        assert entry["condition_type"] == "event"
        assert entry["trigger_type"] == "CPI_RELEASE"

    def test_no_conditions_returns_empty_list(self):
        thesis = _make_thesis()
        db = MagicMock()
        _configure_db(db)

        brief = assemble_brief(thesis, db)

        assert brief["falsification_conditions"] == []

    def test_source_index_dedupes_citations_across_runs(self):
        thesis = _make_thesis()
        db = MagicMock()
        shared_citation = {
            "source_type": "FRED",
            "label": "FRED:T10Y2Y, retrieved 2026-07-01",
            "url": None,
            "retrieval_date": "2026-07-01",
        }
        run_a = _make_run(
            "MacroContextWorkflow",
            citations=[shared_citation],
            started_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        run_b = _make_run(
            "HistoricalAnalogWorkflow",
            citations=[shared_citation],
            started_at=datetime(2026, 7, 1, 0, 1, tzinfo=UTC),
        )
        _configure_db(db, runs=[run_a, run_b])

        brief = assemble_brief(thesis, db)

        assert brief["source_index"] == [shared_citation]

    def test_source_index_includes_distinct_citations_from_all_runs(self):
        thesis = _make_thesis()
        db = MagicMock()
        citation_a = {
            "source_type": "FRED",
            "label": "FRED:T10Y2Y, retrieved 2026-07-01",
            "url": None,
            "retrieval_date": "2026-07-01",
        }
        citation_b = {
            "source_type": "IBKR",
            "label": "IBKR:TLT price_history, 2026-07-01T00:00:00",
            "url": None,
            "retrieval_date": "2026-07-01",
        }
        run_a = _make_run("MacroContextWorkflow", citations=[citation_a])
        run_b = _make_run("InstrumentAnalysisWorkflow", citations=[citation_b])
        _configure_db(db, runs=[run_a, run_b])

        brief = assemble_brief(thesis, db)

        assert brief["source_index"] == [citation_a, citation_b]

    def test_source_index_empty_when_no_runs(self):
        thesis = _make_thesis()
        db = MagicMock()
        _configure_db(db)

        brief = assemble_brief(thesis, db)

        assert brief["source_index"] == []

    def test_short_direction_serialized(self):
        thesis = _make_thesis(direction=Direction.short)
        db = MagicMock()
        _configure_db(db)

        brief = assemble_brief(thesis, db)

        assert brief["direction"] == "short"


# ---------------------------------------------------------------------------
# store_brief
# ---------------------------------------------------------------------------


class TestStoreBrief:
    def test_attaches_brief_to_thesis(self):
        thesis = _make_thesis()
        db = MagicMock()
        _configure_db(db)

        result = store_brief(thesis, db)

        assert thesis.brief == result
        assert thesis.brief["thesis_id"] == str(thesis.id)

    def test_sets_brief_generated_at(self):
        thesis = _make_thesis()
        db = MagicMock()
        _configure_db(db)

        store_brief(thesis, db)

        assert thesis.brief_generated_at is not None

    def test_does_not_commit(self):
        thesis = _make_thesis()
        db = MagicMock()
        _configure_db(db)

        store_brief(thesis, db)

        db.commit.assert_not_called()
