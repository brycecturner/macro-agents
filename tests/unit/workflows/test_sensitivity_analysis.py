"""Tests for SensitivityAnalysisWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np

from app.integrations.anthropic_client import AnthropicResponse
from app.integrations.ibkr_client import IBKRBar, IBKRClientError, IBKRPriceHistory
from app.workflows.base import WorkflowContext, WorkflowResult, WorkflowStatus
from app.workflows.sensitivity_analysis import SensitivityAnalysisWorkflow

# ---------------------------------------------------------------------------
# Helpers
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
        "macro_conditions": {},
        "similarity_rationale": "[Agent inference] Similar Fed tightening.",
        "outcome_summary": "[Agent inference] Rates peaked and reversed.",
    },
    {
        "start_date": "2022-01",
        "end_date": "2022-09",
        "label": "2022 rate-hike cycle",
        "macro_conditions": {},
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
    analysis: str = "The edge is fairly robust to entry timing.",
    agent_inferences: list[str] | None = None,
) -> AnthropicResponse:
    inferences = agent_inferences or ["[Agent inference] Sample size is small."]
    content = json.dumps({"analysis": analysis, "agent_inferences": inferences})
    return AnthropicResponse(
        content=content,
        model="claude-sonnet-4-6",
        input_tokens=300,
        output_tokens=150,
        stop_reason="end_turn",
    )


def _make_anthropic_client(response: AnthropicResponse | None = None) -> MagicMock:
    mock = MagicMock()
    mock.complete.return_value = response or _make_anthropic_response()
    return mock


class TestSensitivityAnalysisWorkflow:
    def test_returns_completed_status_with_valid_data(self) -> None:
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.COMPLETED

    def test_offsets_include_all_seven_values(self) -> None:
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        offsets = [
            entry["offset_months"] for entry in result.structured_output["offsets"]
        ]
        assert offsets == [-3, -2, -1, 0, 1, 2, 3]

    def test_baseline_offset_has_aggregate_data(self) -> None:
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        baseline = next(
            e for e in result.structured_output["offsets"] if e["offset_months"] == 0
        )
        assert baseline["aggregate"]
        assert baseline["aggregate"]["n_periods"] == 2

    def test_instrument_and_direction_in_output(self) -> None:
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["instrument"] == "TLT"
        assert result.structured_output["direction"] == "long"

    def test_partial_when_no_analog_result(self) -> None:
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context(with_analog_result=False)
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL
        assert any("HistoricalAnalogWorkflow" in inf for inf in result.agent_inferences)

    def test_partial_when_no_instrument_data(self) -> None:
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context(with_instrument_result=False)
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL

    def test_partial_when_ibkr_fails(self) -> None:
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("unreachable")
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=ibkr, anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL
        assert any("IBKR" in inf for inf in result.agent_inferences)

    def test_citations_include_ibkr_price_history(self) -> None:
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert any("TLT" in c.label for c in result.citations)

    def test_llm_agent_inferences_included(self) -> None:
        anthropic = _make_anthropic_client(
            _make_anthropic_response(
                agent_inferences=["[Agent inference] Custom note."]
            )
        )
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=anthropic
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert "[Agent inference] Custom note." in result.agent_inferences

    def test_analysis_narrative_stored(self) -> None:
        anthropic = _make_anthropic_client(
            _make_anthropic_response(analysis="Robust across offsets.")
        )
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=anthropic
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["analysis"] == "Robust across offsets."

    def test_falls_back_to_thesis_instruments_when_no_instrument_result(self) -> None:
        instr = MagicMock()
        instr.instrument = "TLT"
        instr.direction = MagicMock()
        instr.direction.value = "long"
        instr.role = MagicMock()
        instr.role.value = "primary"

        context = _make_context(with_instrument_result=False)
        context.thesis.instruments = [instr]

        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["instrument"] == "TLT"
        assert result.status == WorkflowStatus.COMPLETED

    def test_malformed_llm_json_falls_back_to_raw_content(self) -> None:
        response = AnthropicResponse(
            content="not json",
            model="claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )
        anthropic = _make_anthropic_client(response)
        workflow = SensitivityAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=anthropic
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["analysis"] == "not json"

    def test_model_attribute_is_sonnet(self) -> None:
        assert SensitivityAnalysisWorkflow.model == "claude-sonnet-4-6"

    def test_registered_with_name_and_description(self) -> None:
        assert SensitivityAnalysisWorkflow.name == "SensitivityAnalysisWorkflow"
        assert SensitivityAnalysisWorkflow.description
