"""Tests for BacktestWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.integrations.anthropic_client import AnthropicResponse
from app.integrations.ibkr_client import IBKRBar, IBKRClientError, IBKRPriceHistory
from app.workflows.backtest import (
    BacktestWorkflow,
    _bars_to_closes,
    _compute_aggregate_stats,
    _compute_benchmark_aggregate,
    _compute_period_stats,
    _parse_period_end,
    _parse_period_start,
)
from app.workflows.base import (
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

# ---------------------------------------------------------------------------
# Helpers — price bar generation
# ---------------------------------------------------------------------------


def _make_daily_bars(
    n: int = 3650,
    start: datetime | None = None,
    start_price: float = 100.0,
    vol: float = 0.01,
    seed: int = 42,
) -> list[IBKRBar]:
    """Generate n daily bars with a random-walk close price.

    Default start date 2015-01-02, 3650 bars ≈ 10 calendar years,
    matching _BACKTEST_PRICE_PERIOD and covering all analog period dates
    used throughout these tests (up to ~2025-01-01).
    """
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
    symbol: str = "TLT",
    n: int = 3650,
    **kwargs,
) -> IBKRPriceHistory:
    return IBKRPriceHistory(
        symbol=symbol,
        period="10y",
        bar_size="1d",
        bars=_make_daily_bars(n=n, **kwargs),
        retrieved_at=datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Helpers — workflow context
# ---------------------------------------------------------------------------

# Analog periods that fall within the 2015-01-02 .. ~2025-01-01 bar range
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
    {
        "start_date": "2016-06",
        "end_date": "2016-09",
        "label": "2016 Brexit aftermath",
        "macro_conditions": {},
        "similarity_rationale": "[Agent inference] Risk-off rally.",
        "outcome_summary": "[Agent inference] Bonds rallied on safe-haven demand.",
    },
]


def _make_thesis(title: str = "Yield Curve Steepener") -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.pod_id = uuid.uuid4()
    thesis.title = title
    thesis.direction = MagicMock()
    thesis.direction.value = "long"
    thesis.time_horizon = "6 months"
    thesis.notes = "Long TLT as yield curve steepens."
    thesis.instruments = []
    return thesis


def _make_analog_result(
    analogs: list[dict] | None = None,
) -> WorkflowResult:
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
    symbol: str = "TLT",
    direction: str = "long",
    role: str = "primary",
) -> WorkflowResult:
    return WorkflowResult(
        workflow_name="InstrumentAnalysisWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={
            "instruments": {
                symbol: {
                    "direction": direction,
                    "role": role,
                    "annualized_return": 0.05,
                    "annualized_vol": 0.15,
                    "max_drawdown": -0.20,
                    "vol_60d_realized": 0.14,
                    "start_date": "2020-01-02",
                    "end_date": "2025-01-02",
                    "n_trading_days": 1260,
                }
            },
            "correlations": {},
            "analysis": "TLT is well-suited.",
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


def _make_ibkr_client(
    tlt_history: IBKRPriceHistory | None = None,
    spy_history: IBKRPriceHistory | None = None,
    agg_history: IBKRPriceHistory | None = None,
) -> MagicMock:
    """Return a mock IBKRClient that serves distinct histories per symbol."""
    tlt = tlt_history or _make_price_history("TLT")
    spy = spy_history or _make_price_history("SPY", seed=99)
    agg = agg_history or _make_price_history("AGG", seed=77)

    histories = {"TLT": tlt, "SPY": spy, "AGG": agg}

    mock = MagicMock()
    mock.get_price_history.side_effect = lambda symbol, **_: histories.get(
        symbol, _make_price_history(symbol)
    )
    return mock


def _make_anthropic_response(
    analysis: str = "This is a Historical Analog Analysis.",
    agent_inferences: list[str] | None = None,
) -> AnthropicResponse:
    inferences = agent_inferences or ["[Agent inference] Sample size is small."]
    content = json.dumps({"analysis": analysis, "agent_inferences": inferences})
    return AnthropicResponse(
        content=content,
        model="claude-sonnet-4-6",
        input_tokens=500,
        output_tokens=250,
        stop_reason="end_turn",
    )


def _make_anthropic_client(
    response: AnthropicResponse | None = None,
) -> MagicMock:
    mock = MagicMock()
    mock.complete.return_value = response or _make_anthropic_response()
    return mock


# ---------------------------------------------------------------------------
# _parse_period_start / _parse_period_end
# ---------------------------------------------------------------------------


class TestParsePeriodDates:
    def test_start_is_first_day_of_month(self):
        from datetime import date

        assert _parse_period_start("2022-06") == date(2022, 6, 1)

    def test_end_is_last_day_of_month(self):
        from datetime import date

        assert _parse_period_end("2022-02") == date(2022, 2, 28)

    def test_end_leap_year_february(self):
        from datetime import date

        assert _parse_period_end("2024-02") == date(2024, 2, 29)

    def test_end_31_day_month(self):
        from datetime import date

        assert _parse_period_end("2022-01") == date(2022, 1, 31)


# ---------------------------------------------------------------------------
# _bars_to_closes
# ---------------------------------------------------------------------------


class TestBarsToCLoses:
    def test_returns_series_for_valid_bars(self):
        bars = _make_daily_bars(n=100)
        closes = _bars_to_closes(bars)
        assert len(closes) == 100

    def test_returns_empty_series_for_no_bars(self):
        closes = _bars_to_closes([])
        assert len(closes) == 0

    def test_index_is_tz_naive(self):
        bars = _make_daily_bars(n=100)
        closes = _bars_to_closes(bars)
        assert closes.index.tz is None

    def test_zero_price_bars_excluded(self):
        bars = _make_daily_bars(n=100)
        # Inject a zero-price bar
        bars[10] = IBKRBar(
            timestamp=bars[10].timestamp,
            open=0.0,
            high=0.0,
            low=0.0,
            close=0.0,
            volume=0.0,
        )
        closes = _bars_to_closes(bars)
        assert (closes > 0).all()


# ---------------------------------------------------------------------------
# _compute_period_stats
# ---------------------------------------------------------------------------


class TestComputePeriodStats:
    def _closes_from_bars(self, n: int = 2520, **kwargs):
        return _bars_to_closes(_make_daily_bars(n=n, **kwargs))

    def test_returns_none_when_fewer_than_2_days_in_period(self):
        from datetime import date

        closes = self._closes_from_bars()
        # A period entirely before the bar data
        result = _compute_period_stats(
            closes, date(2000, 1, 1), date(2000, 1, 31), "long"
        )
        assert result is None

    def test_returns_dict_for_valid_period(self):
        from datetime import date

        closes = self._closes_from_bars()
        result = _compute_period_stats(
            closes, date(2018, 10, 1), date(2019, 1, 31), "long"
        )
        assert result is not None
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        from datetime import date

        closes = self._closes_from_bars()
        result = _compute_period_stats(
            closes, date(2018, 10, 1), date(2019, 1, 31), "long"
        )
        assert result is not None
        for key in [
            "total_return",
            "annualized_return",
            "max_drawdown",
            "volatility",
            "directionally_correct",
            "n_trading_days",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_total_return_is_float(self):
        from datetime import date

        closes = self._closes_from_bars()
        result = _compute_period_stats(
            closes, date(2018, 10, 1), date(2019, 1, 31), "long"
        )
        assert result is not None
        assert isinstance(result["total_return"], float)

    def test_max_drawdown_is_non_positive(self):
        from datetime import date

        closes = self._closes_from_bars()
        result = _compute_period_stats(
            closes, date(2018, 10, 1), date(2019, 1, 31), "long"
        )
        assert result is not None
        assert result["max_drawdown"] <= 0

    def test_volatility_is_non_negative(self):
        from datetime import date

        closes = self._closes_from_bars()
        result = _compute_period_stats(
            closes, date(2018, 10, 1), date(2019, 1, 31), "long"
        )
        assert result is not None
        assert result["volatility"] >= 0

    def test_directionally_correct_long_positive_return(self):
        from datetime import date

        # Monotonically rising series — long should be correct
        bars = []
        base = datetime(2015, 1, 2, tzinfo=UTC)
        for i in range(2520):
            price = 100.0 + i * 0.01
            bars.append(
                IBKRBar(
                    timestamp=base + timedelta(days=i),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=1_000_000.0,
                )
            )
        closes = _bars_to_closes(bars)
        result = _compute_period_stats(
            closes, date(2018, 10, 1), date(2019, 1, 31), "long"
        )
        assert result is not None
        assert result["total_return"] > 0
        assert result["directionally_correct"] is True

    def test_directionally_correct_long_negative_return(self):
        from datetime import date

        # Monotonically declining series — long should be incorrect
        bars = []
        base = datetime(2015, 1, 2, tzinfo=UTC)
        for i in range(2520):
            price = max(100.0 - i * 0.01, 0.01)
            bars.append(
                IBKRBar(
                    timestamp=base + timedelta(days=i),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=1_000_000.0,
                )
            )
        closes = _bars_to_closes(bars)
        result = _compute_period_stats(
            closes, date(2018, 10, 1), date(2019, 1, 31), "long"
        )
        assert result is not None
        assert result["total_return"] < 0
        assert result["directionally_correct"] is False

    def test_directionally_correct_short_negative_return(self):
        from datetime import date

        # Declining series — short should be correct
        bars = []
        base = datetime(2015, 1, 2, tzinfo=UTC)
        for i in range(2520):
            price = max(100.0 - i * 0.01, 0.01)
            bars.append(
                IBKRBar(
                    timestamp=base + timedelta(days=i),
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=1_000_000.0,
                )
            )
        closes = _bars_to_closes(bars)
        result = _compute_period_stats(
            closes, date(2018, 10, 1), date(2019, 1, 31), "short"
        )
        assert result is not None
        assert result["directionally_correct"] is True

    def test_n_trading_days_reflects_data_in_period(self):
        from datetime import date

        closes = self._closes_from_bars()
        result = _compute_period_stats(
            closes, date(2018, 10, 1), date(2019, 1, 31), "long"
        )
        assert result is not None
        assert result["n_trading_days"] > 0

    def test_annualized_return_longer_than_1_year(self):
        from datetime import date

        # For a period > 1 year, annualized and total return should differ
        closes = self._closes_from_bars()
        result = _compute_period_stats(
            closes, date(2016, 1, 1), date(2018, 12, 31), "long"
        )
        assert result is not None
        # They won't be identical once annualisation is applied
        assert result["annualized_return"] != result["total_return"]


# ---------------------------------------------------------------------------
# _compute_aggregate_stats
# ---------------------------------------------------------------------------


class TestComputeAggregateStats:
    def test_returns_empty_dict_for_empty_input(self):
        assert _compute_aggregate_stats([]) == {}

    def test_n_periods_correct(self):
        stats = [
            {"total_return": 0.1, "max_drawdown": -0.05, "directionally_correct": True},
            {
                "total_return": -0.05,
                "max_drawdown": -0.1,
                "directionally_correct": False,
            },
            {
                "total_return": 0.08,
                "max_drawdown": -0.03,
                "directionally_correct": True,
            },
        ]
        result = _compute_aggregate_stats(stats)
        assert result["n_periods"] == 3

    def test_avg_return_correct(self):
        stats = [
            {"total_return": 0.1, "max_drawdown": 0.0, "directionally_correct": True},
            {"total_return": 0.2, "max_drawdown": 0.0, "directionally_correct": True},
        ]
        result = _compute_aggregate_stats(stats)
        assert abs(result["avg_return"] - 0.15) < 1e-6

    def test_worst_and_best_return(self):
        stats = [
            {"total_return": 0.1, "max_drawdown": 0.0, "directionally_correct": True},
            {
                "total_return": -0.2,
                "max_drawdown": -0.2,
                "directionally_correct": False,
            },
            {"total_return": 0.3, "max_drawdown": -0.05, "directionally_correct": True},
        ]
        result = _compute_aggregate_stats(stats)
        assert result["worst_return"] == pytest.approx(-0.2)
        assert result["best_return"] == pytest.approx(0.3)

    def test_win_rate_all_correct(self):
        stats = [
            {"total_return": 0.1, "max_drawdown": 0.0, "directionally_correct": True},
            {"total_return": 0.2, "max_drawdown": 0.0, "directionally_correct": True},
        ]
        result = _compute_aggregate_stats(stats)
        assert result["win_rate"] == pytest.approx(1.0)

    def test_win_rate_none_correct(self):
        stats = [
            {
                "total_return": -0.1,
                "max_drawdown": -0.1,
                "directionally_correct": False,
            },
            {
                "total_return": -0.2,
                "max_drawdown": -0.2,
                "directionally_correct": False,
            },
        ]
        result = _compute_aggregate_stats(stats)
        assert result["win_rate"] == pytest.approx(0.0)

    def test_win_rate_partial(self):
        stats = [
            {"total_return": 0.1, "max_drawdown": 0.0, "directionally_correct": True},
            {
                "total_return": -0.1,
                "max_drawdown": -0.1,
                "directionally_correct": False,
            },
            {"total_return": 0.05, "max_drawdown": 0.0, "directionally_correct": True},
            {
                "total_return": -0.05,
                "max_drawdown": -0.05,
                "directionally_correct": False,
            },
        ]
        result = _compute_aggregate_stats(stats)
        assert result["win_rate"] == pytest.approx(0.5)

    def test_avg_max_drawdown_correct(self):
        stats = [
            {"total_return": 0.1, "max_drawdown": -0.1, "directionally_correct": True},
            {"total_return": 0.2, "max_drawdown": -0.3, "directionally_correct": True},
        ]
        result = _compute_aggregate_stats(stats)
        assert abs(result["avg_max_drawdown"] - (-0.2)) < 1e-6


# ---------------------------------------------------------------------------
# _compute_benchmark_aggregate
# ---------------------------------------------------------------------------


class TestComputeBenchmarkAggregate:
    def test_returns_empty_dict_for_empty_input(self):
        assert _compute_benchmark_aggregate([]) == {}

    def test_avg_return_correct(self):
        result = _compute_benchmark_aggregate([0.1, 0.2, 0.3])
        assert abs(result["avg_return"] - 0.2) < 1e-6

    def test_worst_and_best_correct(self):
        result = _compute_benchmark_aggregate([-0.1, 0.05, 0.2])
        assert result["worst_return"] == pytest.approx(-0.1)
        assert result["best_return"] == pytest.approx(0.2)

    def test_win_rate_counts_positive_returns(self):
        result = _compute_benchmark_aggregate([0.1, -0.05, 0.2, -0.1])
        assert result["win_rate"] == pytest.approx(0.5)

    def test_n_periods_matches_input_length(self):
        result = _compute_benchmark_aggregate([0.1, 0.2, -0.05])
        assert result["n_periods"] == 3


# ---------------------------------------------------------------------------
# BacktestWorkflow — missing dependencies
# ---------------------------------------------------------------------------


class TestBacktestWorkflowMissingDeps:
    def test_partial_when_no_analog_result(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        ctx = _make_context(with_analog_result=False)
        result = workflow.execute(_make_thesis(), ctx)
        assert result.status == WorkflowStatus.PARTIAL

    def test_partial_when_no_instrument_result_and_no_thesis_instruments(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        ctx = _make_context(with_instrument_result=False)
        thesis = _make_thesis()
        thesis.instruments = []
        result = workflow.execute(thesis, ctx)
        assert result.status == WorkflowStatus.PARTIAL

    def test_partial_when_analog_list_empty(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        ctx = _make_context(analogs=[])
        result = workflow.execute(_make_thesis(), ctx)
        assert result.status == WorkflowStatus.PARTIAL

    def test_partial_when_ibkr_fails_for_primary_instrument(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("IBKR unreachable")
        workflow = BacktestWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.PARTIAL

    def test_flags_missing_analogs_in_inferences(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        ctx = _make_context(with_analog_result=False)
        result = workflow.execute(_make_thesis(), ctx)
        combined = " ".join(result.agent_inferences)
        assert "[Agent inference]" in combined
        assert "analog" in combined.lower()

    def test_partial_result_has_required_output_keys(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        ctx = _make_context(with_analog_result=False)
        result = workflow.execute(_make_thesis(), ctx)
        for key in [
            "label",
            "analog_periods",
            "aggregate",
            "benchmark_comparison",
            "analysis",
        ]:
            assert key in result.structured_output, f"Missing key: {key}"

    def test_ibkr_failure_flagged_in_inferences(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("IBKR unreachable")
        workflow = BacktestWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        combined = " ".join(result.agent_inferences)
        assert "[Agent inference]" in combined
        assert "TLT" in combined

    def test_benchmark_ibkr_failure_does_not_prevent_completion(self):
        # SPY and AGG fail, but TLT succeeds — should still complete
        ibkr = MagicMock()
        tlt_history = _make_price_history("TLT")

        def _side_effect(symbol, **_):
            if symbol == "TLT":
                return tlt_history
            raise IBKRClientError(f"{symbol} unavailable")

        ibkr.get_price_history.side_effect = _side_effect
        workflow = BacktestWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED

    def test_falls_back_to_thesis_instruments_when_no_instrument_result(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = lambda symbol, **_: _make_price_history(
            symbol
        )
        workflow = BacktestWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        thesis = _make_thesis()
        instr = MagicMock()
        instr.instrument = "TLT"
        instr.direction = MagicMock()
        instr.direction.value = "long"
        instr.role = MagicMock()
        instr.role.value = "primary"
        thesis.instruments = [instr]

        ctx = _make_context(with_instrument_result=False)
        result = workflow.execute(thesis, ctx)
        assert result.status == WorkflowStatus.COMPLETED
        assert result.structured_output["instrument"] == "TLT"


# ---------------------------------------------------------------------------
# BacktestWorkflow — structured output
# ---------------------------------------------------------------------------


class TestBacktestWorkflowOutput:
    def test_status_is_completed(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED

    def test_output_has_all_required_keys(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        for key in [
            "label",
            "instrument",
            "direction",
            "analog_periods",
            "aggregate",
            "benchmark_comparison",
            "analysis",
        ]:
            assert key in result.structured_output, f"Missing key: {key}"

    def test_label_is_historical_analog_analysis(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.structured_output["label"] == "Historical Analog Analysis"

    def test_instrument_is_primary_symbol(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.structured_output["instrument"] == "TLT"

    def test_analog_periods_is_list(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert isinstance(result.structured_output["analog_periods"], list)

    def test_each_period_has_required_stat_keys(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        for period in result.structured_output["analog_periods"]:
            for key in [
                "label",
                "start_date",
                "end_date",
                "total_return",
                "annualized_return",
                "max_drawdown",
                "volatility",
                "directionally_correct",
                "n_trading_days",
            ]:
                assert key in period, f"Period missing key: {key}"

    def test_aggregate_has_required_keys(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        agg = result.structured_output["aggregate"]
        for key in [
            "n_periods",
            "avg_return",
            "worst_return",
            "best_return",
            "win_rate",
            "avg_max_drawdown",
        ]:
            assert key in agg, f"Aggregate missing key: {key}"

    def test_aggregate_n_periods_matches_analog_count(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        n_with_data = len(result.structured_output["analog_periods"])
        assert result.structured_output["aggregate"]["n_periods"] == n_with_data

    def test_benchmark_comparison_has_spy_and_60_40(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        bc = result.structured_output["benchmark_comparison"]
        assert "spy" in bc
        assert "60_40" in bc

    def test_spy_benchmark_has_required_keys(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        spy = result.structured_output["benchmark_comparison"]["spy"]
        for key in [
            "avg_return",
            "worst_return",
            "best_return",
            "win_rate",
            "n_periods",
        ]:
            assert key in spy, f"SPY benchmark missing key: {key}"

    def test_statistical_limitation_note_present_when_fewer_than_5_periods(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        # _ANALOG_PERIODS has 3 periods — below threshold of 5
        result = workflow.execute(_make_thesis(), _make_context())
        assert "statistical_limitation_note" in result.structured_output["aggregate"]

    def test_statistical_limitation_note_absent_when_5_or_more_periods(self):
        five_analogs = _ANALOG_PERIODS + [
            {
                "start_date": "2019-06",
                "end_date": "2019-09",
                "label": "2019 Fed pivot",
                "macro_conditions": {},
                "similarity_rationale": "[Agent inference] Fed paused.",
                "outcome_summary": "[Agent inference] Bonds rallied.",
            },
            {
                "start_date": "2020-03",
                "end_date": "2020-05",
                "label": "2020 COVID shock",
                "macro_conditions": {},
                "similarity_rationale": "[Agent inference] Crisis conditions.",
                "outcome_summary": "[Agent inference] Bonds spiked.",
            },
        ]
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        ctx = _make_context(analogs=five_analogs)
        result = workflow.execute(_make_thesis(), ctx)
        assert (
            "statistical_limitation_note" not in result.structured_output["aggregate"]
        )

    def test_periods_with_no_price_data_excluded_with_inference(self):
        analogs_with_old_period = _ANALOG_PERIODS + [
            {
                "start_date": "1995-01",
                "end_date": "1995-06",
                "label": "1995 bond rally",
                "macro_conditions": {},
                "similarity_rationale": "[Agent inference] Rate cuts.",
                "outcome_summary": "[Agent inference] Bonds rallied.",
            }
        ]
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        ctx = _make_context(analogs=analogs_with_old_period)
        result = workflow.execute(_make_thesis(), ctx)
        # The 1995 period should be excluded (no data in 2015-2025 range)
        period_labels = [p["label"] for p in result.structured_output["analog_periods"]]
        assert "1995 bond rally" not in period_labels
        # And an inference should flag it
        combined = " ".join(result.agent_inferences)
        assert "1995 bond rally" in combined

    def test_analysis_is_non_empty_string(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert isinstance(result.structured_output["analysis"], str)
        assert len(result.structured_output["analysis"]) > 0

    def test_workflow_name_is_correct(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.workflow_name == "BacktestWorkflow"

    def test_raw_output_contains_llm_response(self):
        response = _make_anthropic_response(
            analysis="Historical analog suggests upside."
        )
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(response),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert "Historical analog suggests upside." in result.raw_output

    def test_fallback_on_invalid_json_response(self):
        bad_response = AnthropicResponse(
            content="not json",
            model="claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            stop_reason="end_turn",
        )
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(bad_response),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED
        assert result.structured_output["analysis"] == "not json"


# ---------------------------------------------------------------------------
# BacktestWorkflow — citations
# ---------------------------------------------------------------------------


class TestBacktestWorkflowCitations:
    def test_citations_include_primary_instrument(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        labels = [c.label for c in result.citations]
        assert any("IBKR:TLT" in label for label in labels)

    def test_citations_include_spy_benchmark(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        labels = [c.label for c in result.citations]
        assert any("IBKR:SPY" in label for label in labels)

    def test_citations_include_agg_benchmark(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        labels = [c.label for c in result.citations]
        assert any("IBKR:AGG" in label for label in labels)

    def test_all_citations_are_ibkr_type(self):
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        for c in result.citations:
            assert c.source_type == CitationSourceType.IBKR

    def test_no_citation_for_failed_ibkr_fetch(self):
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("unavailable")
        workflow = BacktestWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.citations == []


# ---------------------------------------------------------------------------
# BacktestWorkflow — LLM call
# ---------------------------------------------------------------------------


class TestBacktestWorkflowLLM:
    def test_llm_called_with_correct_task_type(self):
        anthropic = _make_anthropic_client()
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["task_type"] == "backtest"

    def test_llm_called_with_correct_model(self):
        anthropic = _make_anthropic_client()
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"

    def test_workflow_run_id_passed_to_llm(self):
        anthropic = _make_anthropic_client()
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        run_id = uuid.uuid4()
        ctx = _make_context()
        ctx.current_workflow_run_id = run_id
        workflow.execute(_make_thesis(), ctx)
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["workflow_run_id"] == run_id

    def test_thesis_title_in_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        thesis = _make_thesis(title="Duration Steepener Play")
        workflow.execute(thesis, _make_context())
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "Duration Steepener Play" in content

    def test_aggregate_stats_in_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "Average return" in content

    def test_agent_inferences_extracted_from_llm(self):
        response = _make_anthropic_response(
            agent_inferences=["[Agent inference] Very limited sample."]
        )
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=_make_anthropic_client(response),
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert any("Very limited sample" in inf for inf in result.agent_inferences)

    def test_llm_called_exactly_once(self):
        anthropic = _make_anthropic_client()
        workflow = BacktestWorkflow(
            ibkr_client=_make_ibkr_client(),
            anthropic_client=anthropic,
        )
        workflow.execute(_make_thesis(), _make_context())
        assert anthropic.complete.call_count == 1

    def test_ibkr_fetched_for_primary_spy_and_agg(self):
        ibkr = _make_ibkr_client()
        workflow = BacktestWorkflow(
            ibkr_client=ibkr,
            anthropic_client=_make_anthropic_client(),
        )
        workflow.execute(_make_thesis(), _make_context())
        fetched_symbols = {
            call[1]["symbol"] for call in ibkr.get_price_history.call_args_list
        }
        assert "TLT" in fetched_symbols
        assert "SPY" in fetched_symbols
        assert "AGG" in fetched_symbols
