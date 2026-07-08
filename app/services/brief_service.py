"""BriefService — assembles the Tier 1 trade brief from workflow_runs outputs.

Per PRD Section 4.4, the Tier 1 brief is a fixed structure covering the thesis
summary, instrument/direction, time horizon, historical analog analysis
("backtest stats"), key assumptions, falsification conditions, the agent
recommendation, and a deduplicated source index of every citation used.

`assemble_brief` is a pure read — it derives the brief from workflow_runs and
falsification_conditions rows and does not write anything. `store_brief`
attaches the assembled brief to the thesis (caller commits, so it can be
combined with other writes in the same transaction).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.models.thesis import FalsificationCondition, Thesis
from app.models.workflow import WorkflowRun

_MACRO_CONTEXT = "MacroContextWorkflow"
_BACKTEST = "BacktestWorkflow"
_RECOMMENDATION = "RecommendationWorkflow"


def _enum_value(value: object) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _resolve_primary_instrument(thesis: Thesis, backtest_output: dict) -> str | None:
    """Prefer thesis_instruments; fall back to the instrument BacktestWorkflow used."""
    instruments = list(getattr(thesis, "instruments", []) or [])
    for instr in instruments:
        if _enum_value(instr.role) == "primary":
            return instr.instrument
    if instruments:
        return instruments[0].instrument
    return backtest_output.get("instrument")


def _build_source_index(runs: list[WorkflowRun]) -> list[dict]:
    """Dedupe citations across all workflow runs, preserving first-seen order."""
    seen: set[tuple] = set()
    source_index: list[dict] = []
    for run in runs:
        for citation in run.citations or []:
            key = (
                citation.get("source_type"),
                citation.get("label"),
                citation.get("url"),
            )
            if key in seen:
                continue
            seen.add(key)
            source_index.append(citation)
    return source_index


def _build_backtest_stats(backtest_output: dict) -> dict:
    aggregate = backtest_output.get("aggregate", {}) or {}
    return {
        "label": backtest_output.get("label", "Historical Analog Analysis"),
        "n_periods": aggregate.get("n_periods", 0),
        "avg_return": aggregate.get("avg_return"),
        "worst_return": aggregate.get("worst_return"),
        "best_return": aggregate.get("best_return"),
        "win_rate": aggregate.get("win_rate"),
        "avg_max_drawdown": aggregate.get("avg_max_drawdown"),
        "statistical_limitation_note": aggregate.get("statistical_limitation_note"),
        "benchmark_comparison": backtest_output.get("benchmark_comparison", {}),
        "analysis": backtest_output.get("analysis", ""),
    }


def assemble_brief(thesis: Thesis, db: Session) -> dict:
    """Assemble the Tier 1 trade brief from persisted workflow_runs and
    falsification_conditions rows. Does not write anything.
    """
    runs = (
        db.query(WorkflowRun)
        .filter(WorkflowRun.thesis_id == thesis.id)
        .order_by(WorkflowRun.started_at)
        .all()
    )
    runs_by_name = {run.workflow_name: run for run in runs}

    macro_output = (
        (runs_by_name[_MACRO_CONTEXT].structured_output or {})
        if (_MACRO_CONTEXT in runs_by_name)
        else {}
    )
    backtest_output = (
        (runs_by_name[_BACKTEST].structured_output or {})
        if (_BACKTEST in runs_by_name)
        else {}
    )
    recommendation_output = (
        (runs_by_name[_RECOMMENDATION].structured_output or {})
        if _RECOMMENDATION in runs_by_name
        else {}
    )

    conditions = (
        db.query(FalsificationCondition)
        .filter(FalsificationCondition.thesis_id == thesis.id)
        .order_by(FalsificationCondition.created_at)
        .all()
    )

    return {
        "thesis_id": str(thesis.id),
        "title": thesis.title,
        "summary": macro_output.get("summary", ""),
        "instrument": _resolve_primary_instrument(thesis, backtest_output),
        "direction": _enum_value(thesis.direction),
        "time_horizon": thesis.time_horizon,
        "backtest_stats": _build_backtest_stats(backtest_output),
        "assumptions": recommendation_output.get("key_assumptions", []),
        "falsification_conditions": [
            {
                "id": str(condition.id),
                "description": condition.description,
                "condition_type": _enum_value(condition.condition_type),
                "trigger_type": condition.trigger_type,
                "measurable_proxy": condition.measurable_proxy,
                "evaluation_logic": condition.evaluation_logic,
            }
            for condition in conditions
        ],
        "recommendation": {
            "recommendation": recommendation_output.get("recommendation"),
            "rationale": recommendation_output.get("rationale", ""),
            "confidence_level": recommendation_output.get("confidence_level"),
        },
        "source_index": _build_source_index(runs),
    }


def store_brief(thesis: Thesis, db: Session) -> dict:
    """Assemble the brief and attach it to the thesis. Caller is responsible
    for committing — this lets callers combine the write with other state
    changes (e.g. status transitions) in a single transaction.
    """
    brief = assemble_brief(thesis, db)
    thesis.brief = brief
    thesis.brief_generated_at = datetime.now(UTC)
    return brief
