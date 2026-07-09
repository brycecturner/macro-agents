"""Tests for HistoricalAnalogDetailWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np

from app.integrations.anthropic_client import AnthropicResponse
from app.integrations.ibkr_client import IBKRBar, IBKRClientError, IBKRPriceHistory
from app.workflows.base import WorkflowContext, WorkflowResult, WorkflowStatus
from app.workflows.historical_analog_detail import (
    HistoricalAnalogDetailWorkflow,
    _duration_months,
)

# ---------------------------------------------------------------------------
# _duration_months
# ---------------------------------------------------------------------------


class TestDurationMonths:
    def test_same_month_is_one(self) -> None:
        assert _duration_months("2020-01", "2020-01") == 1

    def test_across_months_same_year(self) -> None:
        assert _duration_months("2018-10", "2019-01") == 4

    def test_across_years(self) -> None:
        assert _duration_months("2021-11", "2022-02") == 4

    def test_full_year(self) -> None:
        assert _duration_months("2020-01", "2020-12") == 12


# ---------------------------------------------------------------------------
# Workflow execute() — helpers
# ---------------------------------------------------------------------------


def _make_daily_bars(
    n: int = 3650,
    start: datetime | None = None,
    start_price: float = 100.0,
    vol: float = 0.01,
    seed: int = 42,
) -> list[IBKRBar]:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, vol, n)
    prices = start_price * np.exp(np.cumsum(log_returns))
    base = start or datetime(2015, 1, 2, tzinfo=UTC)
    bars = []
    for i, price in enumerate(prices):
        ts = base + timedelta(days=i)
        bars.append(
            IBKRBar(
                timestamp=ts,
                open=float(price * 0.999),
                high=float(price * 1.005),
                low=float(price * 0.995),
                close=float(price),
                volume=1_000_000.0,
            )
        )
    return bars


def _make_price_history(
    symbol: str = "TLT", n: int = 3650, **kwargs
) -> IBKRPriceHistory:
    return IBKRPriceHistory(
        symbol=symbol,
        period="10y",
        bar_size="1d",
        bars=_make_daily_bars(n=n, **kwargs),
        retrieved_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
    )


_ANALOG_PERIODS = [
    {
        "start_date": "2018-10",
        "end_date": "2019-01",
        "label": "2018 rate-hike peak",
        "macro_conditions": {"fedfunds": 2.25},
        "similarity_rationale": "[Agent inference] Similar Fed tightening.",
        "outcome_summary": "[Agent inference] Rates peaked and reversed.",
    },
    {
        "start_date": "2022-01",
        "end_date": "2022-09",
        "label": "2022 rate-hike cycle",
        "macro_conditions": {"fedfunds": 4.0},
        "similarity_rationale": "[Agent inference] Aggressive rate hikes.",
        "outcome_summary": "[Agent inference] Bonds sold off sharply.",
    },
]


def _make_thesis(title: str = "Yield Curve Steepener") -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.pod_id = uuid.uuid4()
    thesis.title = title
    thesis.instruments = []
    return thesis


def _make_analog_result(analogs: list[dict] | None = None) -> WorkflowResult:
    return WorkflowResult(
        workflow_name="HistoricalAnalogWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={
            "analogs": analogs if analogs is not None else _ANALOG_PERIODS
        },
        citations=[],
        agent_inferences=[],
        raw_output="{}",
    )


def _make_instrument_result(
    symbol: str = "TLT", direction: str = "long", role: str = "primary"
) -> WorkflowResult:
    return WorkflowResult(
        workflow_name="InstrumentAnalysisWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={
            "instruments": {symbol: {"direction": direction, "role": role}},
            "correlations": {},
            "analysis": "",
        },
        citations=[],
        agent_inferences=[],
        raw_output="{}",
    )


def _make_context(
    with_analog_result: bool = True,
    with_instrument_result: bool = True,
    analogs: list[dict] | None = None,
) -> WorkflowContext:
    ctx = WorkflowContext(thesis=_make_thesis(), db=MagicMock())
    if with_analog_result:
        ctx.prior_results.append(_make_analog_result(analogs=analogs))
    if with_instrument_result:
        ctx.prior_results.append(_make_instrument_result())
    return ctx


def _make_ibkr_client(history: IBKRPriceHistory | None = None) -> MagicMock:
    mock = MagicMock()
    mock.get_price_history.return_value = history or _make_price_history("TLT")
    return mock


def _make_anthropic_response(
    catalysts_by_label: dict[str, list[str]] | None = None,
    agent_inferences: list[str] | None = None,
) -> AnthropicResponse:
    catalysts_by_label = catalysts_by_label or {
        "2018 rate-hike peak": ["[Agent inference] Fed pause signal in Dec 2018."],
        "2022 rate-hike cycle": ["[Agent inference] Aggressive inflation fight."],
    }
    periods = [
        {"label": label, "likely_catalysts": catalysts}
        for label, catalysts in catalysts_by_label.items()
    ]
    content = json.dumps(
        {"periods": periods, "agent_inferences": agent_inferences or []}
    )
    return AnthropicResponse(
        content=content,
        model="claude-sonnet-4-6",
        input_tokens=400,
        output_tokens=200,
        stop_reason="end_turn",
    )


def _make_anthropic_client(response: AnthropicResponse | None = None) -> MagicMock:
    mock = MagicMock()
    mock.complete.return_value = response or _make_anthropic_response()
    return mock


class TestHistoricalAnalogDetailWorkflow:
    def test_returns_completed_status_with_valid_data(self) -> None:
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.COMPLETED

    def test_one_period_entry_per_analog(self) -> None:
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert len(result.structured_output["periods"]) == 2

    def test_duration_months_computed_per_period(self) -> None:
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        first = result.structured_output["periods"][0]
        assert first["duration_months"] == 4

    def test_quantitative_stats_populated_when_price_data_available(self) -> None:
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        first = result.structured_output["periods"][0]
        assert first["total_return"] is not None
        assert first["max_drawdown"] is not None
        assert first["n_trading_days"] is not None

    def test_qualitative_fields_carried_over(self) -> None:
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        first = result.structured_output["periods"][0]
        assert first["macro_conditions"] == {"fedfunds": 2.25}
        assert "Similar Fed tightening" in first["similarity_rationale"]
        assert "Rates peaked" in first["outcome_summary"]

    def test_likely_catalysts_merged_by_label(self) -> None:
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        first = result.structured_output["periods"][0]
        assert first["likely_catalysts"] == [
            "[Agent inference] Fed pause signal in Dec 2018."
        ]

    def test_partial_when_no_analogs(self) -> None:
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context(with_analog_result=False)
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL
        assert result.structured_output["periods"] == []

    def test_completes_without_price_stats_when_ibkr_fails(self) -> None:
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("unreachable")
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=ibkr, anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.COMPLETED
        first = result.structured_output["periods"][0]
        assert first["total_return"] is None
        assert any("IBKR" in inf for inf in result.agent_inferences)

    def test_completes_without_price_stats_when_no_instrument_data(self) -> None:
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context(with_instrument_result=False)
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.COMPLETED
        assert result.structured_output["instrument"] is None
        assert result.structured_output["periods"][0]["total_return"] is None

    def test_citations_include_ibkr_price_history(self) -> None:
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert any("TLT" in c.label for c in result.citations)

    def test_malformed_llm_json_leaves_catalysts_empty(self) -> None:
        response = AnthropicResponse(
            content="not json",
            model="claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )
        anthropic = _make_anthropic_client(response)
        workflow = HistoricalAnalogDetailWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=anthropic
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.COMPLETED
        assert result.structured_output["periods"][0]["likely_catalysts"] == []

    def test_model_attribute_is_sonnet(self) -> None:
        assert HistoricalAnalogDetailWorkflow.model == "claude-sonnet-4-6"

    def test_registered_with_name_and_description(self) -> None:
        assert HistoricalAnalogDetailWorkflow.name == "HistoricalAnalogDetailWorkflow"
        assert HistoricalAnalogDetailWorkflow.description
