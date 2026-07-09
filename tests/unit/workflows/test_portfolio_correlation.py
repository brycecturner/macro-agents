"""Tests for PortfolioCorrelationWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np

from app.integrations.anthropic_client import AnthropicResponse
from app.integrations.ibkr_client import IBKRBar, IBKRClientError, IBKRPriceHistory
from app.models.execution import Position
from app.workflows.base import WorkflowContext, WorkflowResult, WorkflowStatus
from app.workflows.portfolio_correlation import PortfolioCorrelationWorkflow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_daily_bars(
    n: int = 260,
    start: datetime | None = None,
    start_price: float = 100.0,
    vol: float = 0.01,
    seed: int = 42,
) -> list[IBKRBar]:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, vol, n)
    prices = start_price * np.exp(np.cumsum(log_returns))
    base = start or datetime(2024, 1, 2, tzinfo=UTC)
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
    symbol: str = "TLT", n: int = 260, **kwargs
) -> IBKRPriceHistory:
    return IBKRPriceHistory(
        symbol=symbol,
        period="1y",
        bar_size="1d",
        bars=_make_daily_bars(n=n, **kwargs),
        retrieved_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
    )


def _make_thesis(title: str = "Yield Curve Steepener") -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.pod_id = uuid.uuid4()
    thesis.title = title
    thesis.instruments = []
    return thesis


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


def _make_position(instrument: str) -> MagicMock:
    p = MagicMock(spec=Position)
    p.instrument = instrument
    return p


def _make_db(positions: list | None = None) -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = positions or []
    return db


def _make_context(
    with_instrument_result: bool = True,
    positions: list | None = None,
) -> WorkflowContext:
    ctx = WorkflowContext(thesis=_make_thesis(), db=_make_db(positions))
    if with_instrument_result:
        ctx.prior_results.append(_make_instrument_result())
    return ctx


def _make_ibkr_client(
    histories: dict[str, IBKRPriceHistory] | None = None,
) -> MagicMock:
    mapping = histories or {}
    mock = MagicMock()

    def _get(symbol: str, **_kwargs) -> IBKRPriceHistory:
        return mapping.get(
            symbol, _make_price_history(symbol, seed=hash(symbol) % 1000)
        )

    mock.get_price_history.side_effect = _get
    return mock


def _make_anthropic_response(
    analysis: str = "GLD shows moderate correlation.",
    agent_inferences: list[str] | None = None,
) -> AnthropicResponse:
    inferences = agent_inferences or ["[Agent inference] No major overlap found."]
    content = json.dumps({"analysis": analysis, "agent_inferences": inferences})
    return AnthropicResponse(
        content=content,
        model="claude-sonnet-4-6",
        input_tokens=200,
        output_tokens=100,
        stop_reason="end_turn",
    )


def _make_anthropic_client(response: AnthropicResponse | None = None) -> MagicMock:
    mock = MagicMock()
    mock.complete.return_value = response or _make_anthropic_response()
    return mock


class TestPortfolioCorrelationWorkflow:
    def test_returns_completed_when_other_position_exists(self) -> None:
        context = _make_context(positions=[_make_position("GLD")])
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.COMPLETED

    def test_correlations_keyed_by_other_symbol(self) -> None:
        context = _make_context(
            positions=[_make_position("GLD"), _make_position("EEM")]
        )
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert set(result.structured_output["correlations"].keys()) == {"GLD", "EEM"}

    def test_correlation_value_between_negative_one_and_one(self) -> None:
        context = _make_context(positions=[_make_position("GLD")])
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        corr = result.structured_output["correlations"]["GLD"]["correlation"]
        assert -1.0 <= corr <= 1.0

    def test_excludes_position_with_same_instrument_as_thesis(self) -> None:
        context = _make_context(positions=[_make_position("TLT")])
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL

    def test_deduplicates_repeated_instrument_positions(self) -> None:
        context = _make_context(
            positions=[_make_position("GLD"), _make_position("GLD")]
        )
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert list(result.structured_output["correlations"].keys()) == ["GLD"]

    def test_partial_when_no_other_positions(self) -> None:
        context = _make_context(positions=[])
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL
        assert any(
            "No other active positions" in inf for inf in result.agent_inferences
        )

    def test_partial_when_no_instrument_data(self) -> None:
        context = _make_context(
            with_instrument_result=False, positions=[_make_position("GLD")]
        )
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL

    def test_partial_when_primary_ibkr_fetch_fails(self) -> None:
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("unreachable")
        context = _make_context(positions=[_make_position("GLD")])
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=ibkr, anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL

    def test_other_symbol_excluded_when_its_ibkr_fetch_fails(self) -> None:
        def _get(symbol: str, **_kwargs):
            if symbol == "GLD":
                raise IBKRClientError("unreachable")
            return _make_price_history(symbol)

        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = _get
        context = _make_context(positions=[_make_position("GLD")])
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=ibkr, anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL
        assert "GLD" not in result.structured_output["correlations"]

    def test_citations_include_all_fetched_instruments(self) -> None:
        context = _make_context(positions=[_make_position("GLD")])
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        labels = [c.label for c in result.citations]
        assert any("TLT" in label for label in labels)
        assert any("GLD" in label for label in labels)

    def test_analysis_narrative_stored(self) -> None:
        anthropic = _make_anthropic_client(
            _make_anthropic_response(analysis="High overlap with GLD.")
        )
        context = _make_context(positions=[_make_position("GLD")])
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=anthropic
        )
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["analysis"] == "High overlap with GLD."

    def test_no_db_treated_as_no_other_positions(self) -> None:
        ctx = WorkflowContext(thesis=_make_thesis(), db=None)
        ctx.prior_results.append(_make_instrument_result())
        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(ctx.thesis, ctx)
        assert result.status == WorkflowStatus.PARTIAL

    def test_falls_back_to_thesis_instruments_when_no_instrument_result(self) -> None:
        instr = MagicMock()
        instr.instrument = "TLT"
        instr.role = MagicMock()
        instr.role.value = "primary"

        context = _make_context(
            with_instrument_result=False, positions=[_make_position("GLD")]
        )
        context.thesis.instruments = [instr]

        workflow = PortfolioCorrelationWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["instrument"] == "TLT"
        assert result.status == WorkflowStatus.COMPLETED

    def test_model_attribute_is_sonnet(self) -> None:
        assert PortfolioCorrelationWorkflow.model == "claude-sonnet-4-6"

    def test_registered_with_name_and_description(self) -> None:
        assert PortfolioCorrelationWorkflow.name == "PortfolioCorrelationWorkflow"
        assert PortfolioCorrelationWorkflow.description
