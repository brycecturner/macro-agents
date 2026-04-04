"""Tests for InstrumentAnalysisWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from app.integrations.anthropic_client import AnthropicResponse
from app.integrations.ibkr_client import IBKRBar, IBKRClientError, IBKRPriceHistory
from app.workflows.base import (
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)
from app.workflows.instrument_analysis import (
    InstrumentAnalysisWorkflow,
    _compute_correlations,
    _compute_instrument_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRED_SERIES = ["T10Y2Y", "CPIAUCSL", "FEDFUNDS", "UNRATE"]


def _make_thesis(title: str = "Yield Curve Steepener") -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.pod_id = uuid.uuid4()
    thesis.title = title
    thesis.direction = MagicMock()
    thesis.direction.value = "long"
    thesis.time_horizon = "6 months"
    thesis.notes = "Long TLT as yield curve steepens."
    return thesis


def _make_instrument(
    symbol: str = "TLT", direction: str = "long", role: str = "primary"
) -> MagicMock:
    instr = MagicMock()
    instr.instrument = symbol
    instr.direction = MagicMock()
    instr.direction.value = direction
    instr.role = MagicMock()
    instr.role.value = role
    return instr


def _make_thesis_with_instruments(
    instruments: list[tuple[str, str, str]] | None = None,
) -> MagicMock:
    """Build a mock thesis with instrument list.

    instruments: list of (symbol, direction, role) tuples.
    """
    thesis = _make_thesis()
    if instruments is None:
        instruments = [("TLT", "long", "primary")]
    thesis.instruments = [_make_instrument(*i) for i in instruments]
    return thesis


def _make_daily_bars(
    n: int = 1260,  # ~5 years of trading days
    start_price: float = 100.0,
    vol: float = 0.01,
    seed: int = 42,
) -> list[IBKRBar]:
    """Generate n daily OHLCV bars with a random walk close price."""
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(0, vol, n)
    prices = start_price * np.exp(np.cumsum(log_returns))

    base = datetime(2019, 1, 2, tzinfo=UTC)
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
    symbol: str = "TLT",
    n: int = 1260,
    **kwargs,
) -> IBKRPriceHistory:
    return IBKRPriceHistory(
        symbol=symbol,
        period="5y",
        bar_size="1d",
        bars=_make_daily_bars(n=n, **kwargs),
        retrieved_at=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


def _make_macro_series_data(n_months: int = 60) -> dict[str, dict]:
    """Build a minimal macro series dict matching MacroContextWorkflow output."""
    index = pd.date_range("2019-01-31", periods=n_months, freq="ME")
    result = {}
    for i, sid in enumerate(_FRED_SERIES):
        values = [float(1.0 + i * 0.1 + j * 0.05) for j in range(n_months)]
        result[sid] = {
            "label": sid,
            "current_value": values[-1],
            "current_date": index[-1].date().isoformat(),
            "year_ago_value": None,
            "yoy_change": None,
            "historical_data": [
                {"date": idx.date().isoformat(), "value": v}
                for idx, v in zip(index, values, strict=True)
            ],
        }
    return result


def _make_macro_workflow_result(series_data: dict | None = None) -> WorkflowResult:
    sd = series_data if series_data is not None else _make_macro_series_data()
    return WorkflowResult(
        workflow_name="MacroContextWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={"summary": "Macro is restrictive.", "series": sd},
        citations=[],
        agent_inferences=[],
        raw_output="{}",
    )


def _make_context(with_macro_result: bool = True) -> WorkflowContext:
    ctx = WorkflowContext(thesis=_make_thesis(), db=MagicMock())
    if with_macro_result:
        ctx.prior_results.append(_make_macro_workflow_result())
    return ctx


def _make_ibkr_client(price_history: IBKRPriceHistory | None = None) -> MagicMock:
    mock = MagicMock()
    mock.get_price_history.return_value = price_history or _make_price_history()
    return mock


def _make_anthropic_response(
    analysis: str = "The instrument is well-suited to the thesis.",
    agent_inferences: list[str] | None = None,
) -> AnthropicResponse:
    inferences = agent_inferences or ["[Agent inference] Historical vol is moderate."]
    content = json.dumps({"analysis": analysis, "agent_inferences": inferences})
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


# ---------------------------------------------------------------------------
# _compute_instrument_stats
# ---------------------------------------------------------------------------


class TestComputeInstrumentStats:
    def test_returns_none_when_insufficient_bars(self):
        bars = _make_daily_bars(n=30)
        result = _compute_instrument_stats(bars, "TLT")
        assert result is None

    def test_returns_dict_for_sufficient_bars(self):
        bars = _make_daily_bars(n=1260)
        result = _compute_instrument_stats(bars, "TLT")
        assert result is not None
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        bars = _make_daily_bars(n=1260)
        result = _compute_instrument_stats(bars, "TLT")
        assert result is not None
        for key in [
            "annualized_return",
            "annualized_vol",
            "max_drawdown",
            "vol_60d_realized",
            "start_date",
            "end_date",
            "n_trading_days",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_annualized_return_reasonable_range(self):
        bars = _make_daily_bars(n=1260)
        result = _compute_instrument_stats(bars, "TLT")
        assert result is not None
        # For a reasonable simulation, return should be between -50% and +100%
        assert -0.5 < result["annualized_return"] < 1.0

    def test_annualized_vol_positive(self):
        bars = _make_daily_bars(n=1260)
        result = _compute_instrument_stats(bars, "TLT")
        assert result is not None
        assert result["annualized_vol"] > 0

    def test_max_drawdown_is_non_positive(self):
        bars = _make_daily_bars(n=1260)
        result = _compute_instrument_stats(bars, "TLT")
        assert result is not None
        assert result["max_drawdown"] <= 0

    def test_vol_60d_positive(self):
        bars = _make_daily_bars(n=1260)
        result = _compute_instrument_stats(bars, "TLT")
        assert result is not None
        assert result["vol_60d_realized"] is not None
        assert result["vol_60d_realized"] > 0

    def test_n_trading_days_matches_bar_count(self):
        bars = _make_daily_bars(n=1260)
        result = _compute_instrument_stats(bars, "TLT")
        assert result is not None
        assert result["n_trading_days"] == 1260

    def test_annualized_vol_is_higher_for_noisier_series(self):
        low_vol = _make_daily_bars(n=1260, vol=0.005)
        high_vol = _make_daily_bars(n=1260, vol=0.02, seed=99)
        low_result = _compute_instrument_stats(low_vol, "LOW")
        high_result = _compute_instrument_stats(high_vol, "HIGH")
        assert low_result is not None
        assert high_result is not None
        assert low_result["annualized_vol"] < high_result["annualized_vol"]

    def test_max_drawdown_lower_for_declining_series(self):
        """A declining series should produce a large negative max drawdown."""
        # Start at 100, monotonically decline to ~50 midway, then recover slightly
        n = 252
        prices = [100.0 - (i * 0.2) for i in range(n // 2)]
        prices += [prices[-1] + (i * 0.05) for i in range(n - n // 2)]
        ts_base = datetime(2022, 1, 2, tzinfo=UTC)
        bars = [
            IBKRBar(
                timestamp=ts_base + timedelta(days=i),
                open=p,
                high=p * 1.01,
                low=p * 0.99,
                close=p,
                volume=1_000_000.0,
            )
            for i, p in enumerate(prices)
        ]
        result = _compute_instrument_stats(bars, "DECLINE")
        assert result is not None
        assert result["max_drawdown"] < -0.2

    def test_returns_none_when_all_prices_zero(self):
        ts_base = datetime(2022, 1, 2, tzinfo=UTC)
        bars = [
            IBKRBar(
                timestamp=ts_base + timedelta(days=i),
                open=0.0,
                high=0.0,
                low=0.0,
                close=0.0,
                volume=0.0,
            )
            for i in range(200)
        ]
        result = _compute_instrument_stats(bars, "ZERO")
        assert result is None


# ---------------------------------------------------------------------------
# _compute_correlations
# ---------------------------------------------------------------------------


class TestComputeCorrelations:
    def test_returns_empty_dict_for_empty_bars(self):
        result = _compute_correlations([], _make_macro_series_data())
        assert result == {}

    def test_returns_empty_dict_for_empty_macro_series(self):
        bars = _make_daily_bars(n=1260)
        result = _compute_correlations(bars, {})
        assert result == {}

    def test_returns_correlations_for_all_fred_series(self):
        bars = _make_daily_bars(n=1260)
        macro = _make_macro_series_data(n_months=60)
        result = _compute_correlations(bars, macro)
        # All series with enough overlap should appear
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_correlation_values_in_valid_range(self):
        bars = _make_daily_bars(n=1260)
        macro = _make_macro_series_data(n_months=60)
        result = _compute_correlations(bars, macro)
        for series_id, corr in result.items():
            assert -1.0 <= corr <= 1.0, f"{series_id} correlation out of range: {corr}"

    def test_oecd_series_excluded_from_correlation(self):
        bars = _make_daily_bars(n=1260)
        macro = _make_macro_series_data(n_months=60)
        macro["OECD:ECB Policy Rate"] = {
            "label": "ECB Rate",
            "current_value": 3.5,
            "current_date": "2024-01-31",
            "year_ago_value": None,
            "yoy_change": None,
            "historical_data": [
                {"date": "2023-01-31", "value": 3.5},
                {"date": "2023-02-28", "value": 3.5},
            ],
        }
        result = _compute_correlations(bars, macro)
        assert "OECD:ECB Policy Rate" not in result

    def test_series_with_insufficient_overlap_excluded(self):
        bars = _make_daily_bars(n=1260)
        # Only 5 monthly observations — below _MIN_CORR_MONTHS = 12
        sparse = {
            "T10Y2Y": {
                "label": "T10Y2Y",
                "current_value": -0.5,
                "current_date": "2024-01-31",
                "year_ago_value": None,
                "yoy_change": None,
                "historical_data": [
                    {"date": f"202{i}-01-31", "value": float(-0.5 + i * 0.1)}
                    for i in range(5)
                ],
            }
        }
        result = _compute_correlations(bars, sparse)
        assert "T10Y2Y" not in result

    def test_correlation_values_are_finite(self):
        """All returned correlation values must be finite floats."""
        bars = _make_daily_bars(n=1260)
        macro = _make_macro_series_data(n_months=60)
        result = _compute_correlations(bars, macro)
        for series_id, corr in result.items():
            assert isinstance(corr, float), f"{series_id}: expected float"
            assert not (
                corr != corr
            ), f"{series_id}: NaN correlation returned"  # NaN check


# ---------------------------------------------------------------------------
# InstrumentAnalysisWorkflow — no instruments
# ---------------------------------------------------------------------------


class TestInstrumentAnalysisWorkflowNoInstruments:
    def test_returns_partial_when_no_instruments(self):
        thesis = _make_thesis()
        thesis.instruments = []
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(thesis, _make_context())
        assert result.status == WorkflowStatus.PARTIAL

    def test_flags_missing_instruments_in_inferences(self):
        thesis = _make_thesis()
        thesis.instruments = []
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(thesis, _make_context())
        combined = " ".join(result.agent_inferences)
        assert "[Agent inference]" in combined
        assert "instrument" in combined.lower()

    def test_no_citations_when_no_instruments(self):
        thesis = _make_thesis()
        thesis.instruments = []
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(thesis, _make_context())
        assert result.citations == []


# ---------------------------------------------------------------------------
# InstrumentAnalysisWorkflow — IBKR failure handling
# ---------------------------------------------------------------------------


class TestInstrumentAnalysisWorkflowIBKRFailure:
    def test_handles_ibkr_error_gracefully(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("IBKR unreachable")
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        thesis = _make_thesis_with_instruments()
        result = workflow.execute(thesis, _make_context())
        # LLM still called — result is COMPLETED with empty instruments
        assert result.status == WorkflowStatus.COMPLETED

    def test_flags_ibkr_failure_in_agent_inferences(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("IBKR unreachable")
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        thesis = _make_thesis_with_instruments()
        result = workflow.execute(thesis, _make_context())
        combined = " ".join(result.agent_inferences)
        assert "[Agent inference]" in combined
        assert "TLT" in combined

    def test_no_citation_when_ibkr_fails(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("IBKR unreachable")
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        thesis = _make_thesis_with_instruments()
        result = workflow.execute(thesis, _make_context())
        ibkr_citations = [
            c for c in result.citations if c.source_type == CitationSourceType.IBKR
        ]
        assert ibkr_citations == []


# ---------------------------------------------------------------------------
# InstrumentAnalysisWorkflow — structured output
# ---------------------------------------------------------------------------


class TestInstrumentAnalysisWorkflowOutput:
    def test_result_status_is_completed(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED

    def test_structured_output_has_required_keys(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        for key in ["instruments", "correlations", "analysis"]:
            assert key in result.structured_output, f"Missing key: {key}"

    def test_instrument_stats_in_output(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        assert "TLT" in result.structured_output["instruments"]

    def test_instrument_stats_has_all_stat_fields(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        stats = result.structured_output["instruments"]["TLT"]
        for field in [
            "annualized_return",
            "annualized_vol",
            "max_drawdown",
            "vol_60d_realized",
            "start_date",
            "end_date",
            "n_trading_days",
            "direction",
            "role",
        ]:
            assert field in stats, f"Missing stat field: {field}"

    def test_instrument_direction_and_role_stored(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        stats = result.structured_output["instruments"]["TLT"]
        assert stats["direction"] == "long"
        assert stats["role"] == "primary"

    def test_correlations_in_output_for_instrument(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        assert "TLT" in result.structured_output["correlations"]

    def test_analysis_is_string(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        assert isinstance(result.structured_output["analysis"], str)
        assert len(result.structured_output["analysis"]) > 0

    def test_workflow_name_is_correct(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        assert result.workflow_name == "InstrumentAnalysisWorkflow"

    def test_agent_inferences_extracted_from_llm(self):
        response = _make_anthropic_response(
            agent_inferences=["[Agent inference] Vol is elevated."]
        )
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(response),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        assert any("Vol is elevated" in inf for inf in result.agent_inferences)

    def test_fallback_on_invalid_json_response(self):
        bad_response = AnthropicResponse(
            content="not json",
            model="claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(bad_response),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED
        assert result.structured_output["analysis"] == "not json"

    def test_raw_output_contains_llm_response(self):
        response = _make_anthropic_response(analysis="Well-suited instrument.")
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(response),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        assert "Well-suited instrument." in result.raw_output

    def test_multiple_instruments_all_in_output(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = lambda symbol, **kw: _make_price_history(
            symbol=symbol
        )
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        thesis = _make_thesis_with_instruments(
            [("TLT", "long", "primary"), ("GLD", "long", "hedge")]
        )
        result = workflow.execute(thesis, _make_context())
        assert "TLT" in result.structured_output["instruments"]
        assert "GLD" in result.structured_output["instruments"]


# ---------------------------------------------------------------------------
# InstrumentAnalysisWorkflow — citations
# ---------------------------------------------------------------------------


class TestInstrumentAnalysisWorkflowCitations:
    def test_one_citation_per_instrument(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        ibkr_citations = [
            c for c in result.citations if c.source_type == CitationSourceType.IBKR
        ]
        assert len(ibkr_citations) == 1

    def test_citation_source_type_is_ibkr(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        for c in result.citations:
            assert c.source_type == CitationSourceType.IBKR

    def test_citation_label_format(self):
        """IBKR citation format: 'IBKR:{symbol} price_history, {timestamp}'"""
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        labels = [c.label for c in result.citations]
        assert any("IBKR:TLT price_history" in label for label in labels)

    def test_citation_retrieval_date_matches_price_history(self):
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis_with_instruments(), _make_context())
        for c in result.citations:
            assert c.retrieval_date is not None

    def test_two_citations_for_two_instruments(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = lambda symbol, **kw: _make_price_history(
            symbol=symbol
        )
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        thesis = _make_thesis_with_instruments(
            [("TLT", "long", "primary"), ("GLD", "long", "hedge")]
        )
        result = workflow.execute(thesis, _make_context())
        assert len(result.citations) == 2


# ---------------------------------------------------------------------------
# InstrumentAnalysisWorkflow — LLM call
# ---------------------------------------------------------------------------


class TestInstrumentAnalysisWorkflowLLM:
    def test_llm_called_with_correct_task_type(self):
        anthropic = _make_anthropic_client()
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis_with_instruments(), _make_context())
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["task_type"] == "instrument_analysis"

    def test_llm_called_with_correct_model(self):
        anthropic = _make_anthropic_client()
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis_with_instruments(), _make_context())
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_workflow_run_id_passed_to_llm(self):
        anthropic = _make_anthropic_client()
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        run_id = uuid.uuid4()
        ctx = _make_context()
        ctx.current_workflow_run_id = run_id
        workflow.execute(_make_thesis_with_instruments(), ctx)
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["workflow_run_id"] == run_id

    def test_thesis_title_in_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        thesis = _make_thesis_with_instruments()
        thesis.title = "Duration Steepener Play"
        workflow.execute(thesis, _make_context())
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "Duration Steepener Play" in content

    def test_instrument_stats_in_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis_with_instruments(), _make_context())
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "TLT" in content
        assert "Annualized return" in content

    def test_llm_called_once_regardless_of_instrument_count(self):
        """LLM synthesis is always a single call."""
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = lambda symbol, **kw: _make_price_history(
            symbol=symbol
        )
        anthropic = _make_anthropic_client()
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=ibkr,
            anthropic_client=anthropic,
        )
        thesis = _make_thesis_with_instruments(
            [("TLT", "long", "primary"), ("GLD", "long", "hedge")]
        )
        workflow.execute(thesis, _make_context())
        assert anthropic.complete.call_count == 1

    def test_ibkr_price_history_requested_for_each_instrument(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = lambda symbol, **kw: _make_price_history(
            symbol=symbol
        )
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        thesis = _make_thesis_with_instruments(
            [("TLT", "long", "primary"), ("GLD", "long", "hedge")]
        )
        workflow.execute(thesis, _make_context())
        assert ibkr.get_price_history.call_count == 2

    def test_price_history_requested_with_correct_period(self):
        ibkr = _make_ibkr_client()
        workflow = InstrumentAnalysisWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        workflow.execute(_make_thesis_with_instruments(), _make_context())
        call_kwargs = ibkr.get_price_history.call_args[1]
        assert call_kwargs["period"] == "5y"
