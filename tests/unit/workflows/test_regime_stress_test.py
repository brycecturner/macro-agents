"""Tests for RegimeStressTestWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.integrations.anthropic_client import AnthropicResponse
from app.integrations.ibkr_client import IBKRBar, IBKRClientError, IBKRPriceHistory
from app.workflows.base import WorkflowContext, WorkflowResult, WorkflowStatus
from app.workflows.regime_stress_test import (
    REGIME_CUTTING,
    REGIME_HIGH_INFLATION,
    REGIME_HIKING,
    REGIME_LOW_INFLATION,
    RegimeStressTestWorkflow,
    classify_regimes,
    find_runs,
)

# ---------------------------------------------------------------------------
# classify_regimes
# ---------------------------------------------------------------------------


def _monthly_history(values: list[float], start: str = "2020-01-31") -> list[dict]:
    dates = pd.date_range(start=start, periods=len(values), freq="ME")
    return [
        {"date": d.date().isoformat(), "value": v}
        for d, v in zip(dates, values, strict=True)
    ]


class TestClassifyRegimes:
    def test_empty_when_both_histories_empty(self) -> None:
        result = classify_regimes([], [])
        assert result.empty

    def test_hiking_true_when_fedfunds_rises_over_6_months(self) -> None:
        fedfunds = _monthly_history([1.0] * 6 + [1.5, 2.5])
        result = classify_regimes(fedfunds, [])
        # Last two months: delta6 = 1.5-1.0=0.5 and 2.5-1.0=1.5, both >= 0.25
        assert result[REGIME_HIKING].iloc[-1]
        assert result[REGIME_HIKING].iloc[-2]

    def test_hiking_false_when_fedfunds_flat(self) -> None:
        fedfunds = _monthly_history([1.0] * 8)
        result = classify_regimes(fedfunds, [])
        assert not result[REGIME_HIKING].any()

    def test_cutting_true_when_fedfunds_falls_over_6_months(self) -> None:
        fedfunds = _monthly_history([3.0] * 6 + [2.0, 1.0])
        result = classify_regimes(fedfunds, [])
        assert result[REGIME_CUTTING].iloc[-1]

    def test_high_inflation_true_when_cpi_yoy_exceeds_threshold(self) -> None:
        # 13 months of CPI growing ~0.5%/month → YoY ~6% at month 13 (index 12)
        values = [100.0 * (1.005**i) for i in range(13)]
        cpi = _monthly_history(values)
        result = classify_regimes([], cpi)
        assert result[REGIME_HIGH_INFLATION].iloc[-1]

    def test_low_inflation_true_when_cpi_yoy_below_threshold(self) -> None:
        # 13 months of CPI growing ~0.1%/month → YoY ~1.2% at month 13
        values = [100.0 * (1.001**i) for i in range(13)]
        cpi = _monthly_history(values)
        result = classify_regimes([], cpi)
        assert result[REGIME_LOW_INFLATION].iloc[-1]

    def test_missing_series_treated_as_false(self) -> None:
        fedfunds = _monthly_history([1.0] * 8)
        result = classify_regimes(fedfunds, [])
        assert not result[REGIME_HIGH_INFLATION].any()
        assert not result[REGIME_LOW_INFLATION].any()


# ---------------------------------------------------------------------------
# find_runs
# ---------------------------------------------------------------------------


class TestFindRuns:
    def test_empty_series_returns_no_runs(self) -> None:
        assert find_runs(pd.Series(dtype=bool)) == []

    def test_run_shorter_than_minimum_excluded(self) -> None:
        index = pd.date_range("2020-01-31", periods=6, freq="ME")
        flags = pd.Series([False, True, True, False, False, False], index=index)
        assert find_runs(flags) == []

    def test_run_of_minimum_length_included(self) -> None:
        index = pd.date_range("2020-01-31", periods=6, freq="ME")
        flags = pd.Series([False, True, True, True, False, False], index=index)
        runs = find_runs(flags)
        assert len(runs) == 1
        assert runs[0] == (date(2020, 2, 1), date(2020, 4, 30))

    def test_run_extending_to_end_of_series_included(self) -> None:
        index = pd.date_range("2020-01-31", periods=6, freq="ME")
        flags = pd.Series([False, False, False, True, True, True], index=index)
        runs = find_runs(flags)
        assert len(runs) == 1
        assert runs[0][1] == date(2020, 6, 30)

    def test_multiple_runs_identified(self) -> None:
        index = pd.date_range("2020-01-31", periods=10, freq="ME")
        flags = pd.Series(
            [True, True, True, False, False, True, True, True, True, False],
            index=index,
        )
        runs = find_runs(flags)
        assert len(runs) == 2

    def test_all_true_series_returns_single_run(self) -> None:
        index = pd.date_range("2020-01-31", periods=4, freq="ME")
        flags = pd.Series([True, True, True, True], index=index)
        runs = find_runs(flags)
        assert len(runs) == 1
        assert runs[0][0] == date(2020, 1, 1)


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


def _make_thesis(title: str = "Duration Trade") -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.pod_id = uuid.uuid4()
    thesis.title = title
    thesis.instruments = []
    return thesis


def _make_macro_result(
    with_fedfunds: bool = True, with_cpi: bool = True
) -> WorkflowResult:
    series = {}
    if with_fedfunds:
        series["FEDFUNDS"] = {
            "historical_data": _monthly_history([1.0] * 6 + [1.5, 2.5])
        }
    if with_cpi:
        series["CPIAUCSL"] = {
            "historical_data": _monthly_history([100.0 * (1.005**i) for i in range(13)])
        }
    from app.workflows.base import Citation, CitationSourceType

    citations = [
        Citation(
            source_type=CitationSourceType.FRED,
            label="FRED:FEDFUNDS, retrieved 2025-01-15",
            url=None,
            retrieval_date=date(2025, 1, 15),
        ),
        Citation(
            source_type=CitationSourceType.FRED,
            label="FRED:CPIAUCSL, retrieved 2025-01-15",
            url=None,
            retrieval_date=date(2025, 1, 15),
        ),
    ]
    return WorkflowResult(
        workflow_name="MacroContextWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={"series": series},
        citations=citations,
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
    with_macro_result: bool = True,
    with_instrument_result: bool = True,
    macro_kwargs: dict | None = None,
) -> WorkflowContext:
    ctx = WorkflowContext(thesis=_make_thesis(), db=MagicMock())
    if with_macro_result:
        ctx.prior_results.append(_make_macro_result(**(macro_kwargs or {})))
    if with_instrument_result:
        ctx.prior_results.append(_make_instrument_result())
    return ctx


def _make_ibkr_client(history: IBKRPriceHistory | None = None) -> MagicMock:
    mock = MagicMock()
    mock.get_price_history.return_value = history or _make_price_history("TLT")
    return mock


def _make_anthropic_response(
    analysis: str = "Instrument performs best during cutting cycles.",
    agent_inferences: list[str] | None = None,
) -> AnthropicResponse:
    inferences = agent_inferences or ["[Agent inference] Limited regime coverage."]
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


# ---------------------------------------------------------------------------
# Workflow execute() — with classify_regimes patched to a controlled scenario
# ---------------------------------------------------------------------------


@pytest.fixture
def controlled_regimes(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Patch classify_regimes to return one clean run per regime, all inside
    the 2015-01-02..~2025-01-01 window covered by _make_daily_bars' 10y bars.
    """
    index = pd.date_range("2018-01-31", periods=8, freq="ME")
    df = pd.DataFrame(
        {
            REGIME_HIKING: [True] * 4 + [False] * 4,
            REGIME_CUTTING: [False] * 4 + [True] * 4,
            REGIME_HIGH_INFLATION: [True] * 4 + [False] * 4,
            REGIME_LOW_INFLATION: [False] * 8,
        },
        index=index,
    )
    monkeypatch.setattr(
        "app.workflows.regime_stress_test.classify_regimes", lambda *a, **k: df
    )
    return df


class TestRegimeStressTestWorkflowExecute:
    def test_returns_completed_status(self, controlled_regimes: pd.DataFrame) -> None:
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.COMPLETED

    def test_all_four_regimes_in_output(self, controlled_regimes: pd.DataFrame) -> None:
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        regimes = result.structured_output["regimes"]
        assert set(regimes.keys()) == {
            REGIME_HIKING,
            REGIME_CUTTING,
            REGIME_HIGH_INFLATION,
            REGIME_LOW_INFLATION,
        }

    def test_hiking_regime_has_data(self, controlled_regimes: pd.DataFrame) -> None:
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["regimes"][REGIME_HIKING]["n_runs"] == 1
        assert result.structured_output["regimes"][REGIME_HIKING]["aggregate"]

    def test_low_inflation_regime_has_no_runs(
        self, controlled_regimes: pd.DataFrame
    ) -> None:
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["regimes"][REGIME_LOW_INFLATION]["n_runs"] == 0

    def test_instrument_and_direction_in_output(
        self, controlled_regimes: pd.DataFrame
    ) -> None:
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["instrument"] == "TLT"
        assert result.structured_output["direction"] == "long"

    def test_citations_include_ibkr_and_fred(
        self, controlled_regimes: pd.DataFrame
    ) -> None:
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        labels = [c.label for c in result.citations]
        assert any("IBKR:TLT" in label for label in labels)
        assert any("FEDFUNDS" in label for label in labels)
        assert any("CPIAUCSL" in label for label in labels)

    def test_partial_when_no_macro_result(self) -> None:
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context(with_macro_result=False)
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL

    def test_partial_when_no_instrument_data(self) -> None:
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context(with_instrument_result=False)
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL

    def test_partial_when_series_data_missing(self) -> None:
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=_make_anthropic_client()
        )
        context = _make_context(
            macro_kwargs={"with_fedfunds": False, "with_cpi": False}
        )
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL

    def test_partial_when_ibkr_fails(self, controlled_regimes: pd.DataFrame) -> None:
        ibkr = MagicMock()
        ibkr.get_price_history.side_effect = IBKRClientError("unreachable")
        workflow = RegimeStressTestWorkflow(
            ibkr_client=ibkr, anthropic_client=_make_anthropic_client()
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.status == WorkflowStatus.PARTIAL
        assert any("IBKR" in inf for inf in result.agent_inferences)

    def test_analysis_narrative_stored(self, controlled_regimes: pd.DataFrame) -> None:
        anthropic = _make_anthropic_client(
            _make_anthropic_response(analysis="Cutting cycles are favorable.")
        )
        workflow = RegimeStressTestWorkflow(
            ibkr_client=_make_ibkr_client(), anthropic_client=anthropic
        )
        context = _make_context()
        result = workflow.execute(context.thesis, context)
        assert result.structured_output["analysis"] == "Cutting cycles are favorable."

    def test_model_attribute_is_sonnet(self) -> None:
        assert RegimeStressTestWorkflow.model == "claude-sonnet-4-6"

    def test_registered_with_name_and_description(self) -> None:
        assert RegimeStressTestWorkflow.name == "RegimeStressTestWorkflow"
        assert RegimeStressTestWorkflow.description
