"""BacktestWorkflow — proxy-based Historical Analog Analysis.

This workflow measures how the thesis instrument(s) have historically
performed during the macro periods identified as analogous to current
conditions (from HistoricalAnalogWorkflow). It is explicitly NOT a
rules-based backtest — it measures instrument behavior during analogous
macro regimes and presents the results honestly, including statistical
limitations.

All output is labeled "Historical Analog Analysis". Never "backtest".

Dependencies (consumed from WorkflowContext.prior_results):
  - HistoricalAnalogWorkflow — analog periods (start_date, end_date, label)
  - InstrumentAnalysisWorkflow — instrument symbols, directions, and roles

If either dependency is missing the workflow returns a PARTIAL result with
an explanatory agent inference.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from app.integrations.anthropic_client import AnthropicClient
from app.integrations.ibkr_client import IBKRClient, IBKRClientError
from app.workflows.analysis_utils import (
    bars_to_closes as _bars_to_closes,
)
from app.workflows.analysis_utils import (
    compute_aggregate_stats as _compute_aggregate_stats,
)
from app.workflows.analysis_utils import (
    compute_benchmark_aggregate as _compute_benchmark_aggregate,
)
from app.workflows.analysis_utils import (
    compute_period_stats as _compute_period_stats,
)
from app.workflows.analysis_utils import (
    parse_period_end as _parse_period_end,
)
from app.workflows.analysis_utils import (
    parse_period_start as _parse_period_start,
)
from app.workflows.base import (
    BaseWorkflow,
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

# Lookback period for price history — long enough to cover most analog periods
_BACKTEST_PRICE_PERIOD = "10y"
_BAR_SIZE = "1d"

# Benchmark instruments for comparison
_SPY = "SPY"
_AGG = "AGG"

# Fewer than this many analog periods triggers the statistical limitation note
_STATISTICAL_LIMITATION_THRESHOLD = 5

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a macro research analyst synthesizing the results of a \
Historical Analog Analysis.

CRITICAL: Always call this a "Historical Analog Analysis" — never a "backtest". \
You are measuring how an instrument has historically behaved during periods with \
macro configurations similar to today's. You are not simulating a trading strategy.

You will be given:
1. A trade thesis (direction, time horizon)
2. Per-period instrument performance during each historical analog
3. Aggregate statistics across all analog periods
4. Benchmark comparison (SPY and 60/40 = SPY 60%% + AGG 40%%) over the same periods

Write a 2-3 paragraph synthesis that:
- Summarizes what the historical analogs suggest about the thesis
- Addresses whether the instrument performed in the expected direction in most periods
- Honestly compares performance to benchmarks
- Is explicit about statistical limitations (small sample, limited data history, etc.)

Respond with a JSON object with exactly two keys:
- "analysis": 2-3 paragraph narrative. Never use the word "backtest".
- "agent_inferences": list of strings, each starting with "[Agent inference]"

Respond only with the JSON object. No markdown fences, no preamble.\
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_user_message(
    thesis,
    instrument: str,
    direction: str,
    period_results: list[dict],
    aggregate: dict,
    benchmark_comparison: dict,
) -> str:
    thesis_direction = (
        thesis.direction.value
        if hasattr(thesis.direction, "value")
        else str(thesis.direction)
    )

    lines = [
        f"Thesis: {thesis.title}",
        f"Direction: {thesis_direction}",
        f"Time Horizon: {thesis.time_horizon}",
        f"Instrument: {instrument} (direction: {direction})",
        "",
        "=== Historical Analog Analysis ===",
        "",
    ]

    if period_results:
        lines.append("Per-period performance:")
        for p in period_results:
            correct = "YES" if p.get("directionally_correct") else "NO"
            lines.append(
                f"  {p.get('label', 'Period')} "
                f"({p.get('start_date')} \u2013 {p.get('end_date')}):"
            )
            lines.append(f"    Total return:          {p.get('total_return', 0):.2%}")
            lines.append(
                f"    Annualized return:     {p.get('annualized_return', 0):.2%}"
            )
            lines.append(f"    Max drawdown:          {p.get('max_drawdown', 0):.2%}")
            lines.append(f"    Volatility (ann.):     {p.get('volatility', 0):.2%}")
            lines.append(f"    Directionally correct: {correct}")
            lines.append(f"    Trading days:          {p.get('n_trading_days', 0)}")
            if "spy_total_return" in p:
                lines.append(f"    SPY total return:      {p['spy_total_return']:.2%}")
            if "benchmark_60_40_total_return" in p:
                val = p["benchmark_60_40_total_return"]
                lines.append(f"    60/40 total return:    {val:.2%}")
        lines.append("")
    else:
        lines.append("No analog periods had sufficient price history for analysis.")
        lines.append("")

    if aggregate:
        n = aggregate.get("n_periods", 0)
        lines.append("Aggregate statistics:")
        lines.append(f"  Analog periods analyzed: {n}")
        lines.append(f"  Average return:          {aggregate.get('avg_return', 0):.2%}")
        lines.append(
            f"  Worst return:            {aggregate.get('worst_return', 0):.2%}"
        )
        lines.append(
            f"  Best return:             {aggregate.get('best_return', 0):.2%}"
        )
        lines.append(f"  Win rate:                {aggregate.get('win_rate', 0):.0%}")
        lines.append(
            f"  Avg max drawdown:        {aggregate.get('avg_max_drawdown', 0):.2%}"
        )
        if "statistical_limitation_note" in aggregate:
            lines.append(f"  NOTE: {aggregate['statistical_limitation_note']}")
        lines.append("")

    if benchmark_comparison:
        lines.append("Benchmark comparison (same analog periods):")
        if "spy" in benchmark_comparison:
            spy = benchmark_comparison["spy"]
            lines.append(
                f"  SPY:  avg={spy.get('avg_return', 0):.2%}  "
                f"worst={spy.get('worst_return', 0):.2%}  "
                f"best={spy.get('best_return', 0):.2%}  "
                f"win_rate={spy.get('win_rate', 0):.0%}"
            )
        if "60_40" in benchmark_comparison:
            b = benchmark_comparison["60_40"]
            lines.append(
                f"  60/40: avg={b.get('avg_return', 0):.2%}  "
                f"worst={b.get('worst_return', 0):.2%}  "
                f"best={b.get('best_return', 0):.2%}  "
                f"win_rate={b.get('win_rate', 0):.0%}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------

# Keys extracted from period_results when building the instrument-only stats
# list passed to _compute_aggregate_stats. Excludes benchmark fields.
_INSTRUMENT_STAT_KEYS = frozenset(
    {
        "total_return",
        "annualized_return",
        "max_drawdown",
        "volatility",
        "directionally_correct",
        "n_trading_days",
    }
)


class BacktestWorkflow(BaseWorkflow):
    """Proxy-based Historical Analog Analysis.

    Measures how thesis instruments have historically performed during the
    macro periods identified as analogous to today by HistoricalAnalogWorkflow.
    Computes per-period and aggregate statistics, then benchmarks against
    SPY and 60/40 over the same periods.

    Output is always labeled "Historical Analog Analysis" — never "backtest".

    Outputs (structured_output):
        label (str): Always "Historical Analog Analysis".
        instrument (str): Primary instrument symbol.
        direction (str): Thesis direction for the primary instrument.
        analog_periods (list[dict]): Per-period stats — total_return,
            annualized_return, max_drawdown, volatility, directionally_correct,
            n_trading_days, spy_total_return (if available),
            benchmark_60_40_total_return (if available).
        aggregate (dict): avg_return, worst_return, best_return, win_rate,
            avg_max_drawdown, n_periods, and statistical_limitation_note
            when n_periods < 5.
        benchmark_comparison (dict): spy and 60_40 sub-dicts, each with
            avg_return, worst_return, best_return, win_rate, n_periods.
        analysis (str): LLM narrative synthesis.
    """

    name = "BacktestWorkflow"
    description = (
        "Proxy-based Historical Analog Analysis. Measures thesis instrument "
        "performance during historical macro analog periods (from "
        "HistoricalAnalogWorkflow). Computes per-period return, drawdown, vol, "
        "and directional correctness; aggregates across all periods; benchmarks "
        "vs SPY and 60/40. Output is always labeled 'Historical Analog Analysis'."
    )
    required_inputs = ["title", "direction", "time_horizon"]
    model: str = "claude-sonnet-4-6"

    def __init__(
        self,
        ibkr_client: IBKRClient | None = None,
        anthropic_client: AnthropicClient | None = None,
    ) -> None:
        self._ibkr = ibkr_client
        self._anthropic = anthropic_client

    def execute(self, thesis, context: WorkflowContext) -> WorkflowResult:  # noqa: C901
        # --- Resolve clients ---
        if self._ibkr is None or self._anthropic is None:
            from app.core.settings import get_settings

            settings = get_settings()
        else:
            settings = None  # type: ignore[assignment]

        anthropic = self._anthropic or AnthropicClient(
            api_key=settings.anthropic_api_key,
            db=context.db,
        )

        if self._ibkr is None:
            if context.pod_settings is None:
                raise ValueError(
                    "BacktestWorkflow requires context.pod_settings to construct "
                    "IBKRClient — ensure WorkflowRunner sets it."
                )
            ibkr = IBKRClient(
                base_url=settings.ibkr_base_url,
                account_id=settings.ibkr_account_id,
                paper_account_id=settings.ibkr_paper_account_id,
                pod_settings=context.pod_settings,
            )
        else:
            ibkr = self._ibkr

        agent_inferences: list[str] = []
        citations: list[Citation] = []

        # --- Consume prior workflow results ---
        analog_result = context.get_result("HistoricalAnalogWorkflow")
        instrument_result = context.get_result("InstrumentAnalysisWorkflow")

        analogs: list[dict] = []
        if analog_result is not None:
            analogs = analog_result.structured_output.get("analogs", [])
        else:
            logger.warning(
                "BacktestWorkflow: HistoricalAnalogWorkflow result not in context"
            )
            agent_inferences.append(
                "[Agent inference] HistoricalAnalogWorkflow result was not available — "
                "analog periods could not be identified for analysis."
            )

        # Prefer instruments from InstrumentAnalysisWorkflow; fall back to thesis
        instruments_info: dict[str, dict] = {}
        if instrument_result is not None:
            for symbol, stats in instrument_result.structured_output.get(
                "instruments", {}
            ).items():
                instruments_info[symbol] = stats
        else:
            logger.warning(
                "BacktestWorkflow: InstrumentAnalysisWorkflow result not in context"
            )
            agent_inferences.append(
                "[Agent inference] InstrumentAnalysisWorkflow result was not "
                "available — instrument data read from thesis directly."
            )

        if not instruments_info:
            thesis_instruments = list(getattr(thesis, "instruments", []) or [])
            for instr in thesis_instruments:
                symbol = instr.instrument
                direction = (
                    instr.direction.value
                    if hasattr(instr.direction, "value")
                    else str(instr.direction)
                )
                role = (
                    instr.role.value
                    if hasattr(instr.role, "value")
                    else str(instr.role)
                )
                instruments_info[symbol] = {"direction": direction, "role": role}

        # Hard stop: no analogs or no instruments — cannot produce useful output
        if not analogs:
            agent_inferences.append(
                "[Agent inference] No historical analog periods were available — "
                "historical analog analysis could not be performed."
            )
        if not instruments_info:
            agent_inferences.append(
                "[Agent inference] No instrument data was available — "
                "historical analog analysis could not be performed."
            )

        if not analogs or not instruments_info:
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "label": "Historical Analog Analysis",
                    "instrument": None,
                    "direction": None,
                    "analog_periods": [],
                    "aggregate": {},
                    "benchmark_comparison": {},
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        # Identify the primary instrument
        primary_symbol: str | None = None
        primary_direction = "long"
        for symbol, info in instruments_info.items():
            if info.get("role", "primary") == "primary":
                primary_symbol = symbol
                primary_direction = info.get("direction", "long")
                break
        if primary_symbol is None:
            primary_symbol, _info = next(iter(instruments_info.items()))
            primary_direction = _info.get("direction", "long")

        # --- Fetch price histories: primary instrument + benchmarks ---
        price_series: dict[str, pd.Series] = {}

        for symbol in [primary_symbol, _SPY, _AGG]:
            try:
                price_history = ibkr.get_price_history(
                    symbol=symbol,
                    period=_BACKTEST_PRICE_PERIOD,
                    bar_size=_BAR_SIZE,
                )
                price_series[symbol] = _bars_to_closes(price_history.bars)
                citations.append(
                    Citation(
                        source_type=CitationSourceType.IBKR,
                        label=(
                            f"IBKR:{symbol} price_history, "
                            f"{price_history.retrieved_at.isoformat()}"
                        ),
                        url=None,
                        retrieval_date=price_history.retrieved_at.date(),
                    )
                )
            except IBKRClientError as exc:
                logger.warning(
                    "BacktestWorkflow: failed to fetch price history for %s: %s",
                    symbol,
                    exc,
                )
                if symbol == primary_symbol:
                    agent_inferences.append(
                        f"[Agent inference] Price history for {symbol} could not "
                        "be retrieved from IBKR — historical analog analysis is "
                        "unavailable."
                    )
                else:
                    agent_inferences.append(
                        f"[Agent inference] Benchmark price history for {symbol} could "
                        "not be retrieved — benchmark comparison will be partial."
                    )

        # Without the primary instrument's prices we cannot proceed
        if primary_symbol not in price_series:
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "label": "Historical Analog Analysis",
                    "instrument": primary_symbol,
                    "direction": primary_direction,
                    "analog_periods": [],
                    "aggregate": {},
                    "benchmark_comparison": {},
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        instrument_closes = price_series[primary_symbol]
        spy_closes = price_series.get(_SPY)
        agg_closes = price_series.get(_AGG)

        # --- Compute per-period statistics ---
        period_results: list[dict] = []

        for analog in analogs:
            start_str = analog.get("start_date", "")
            end_str = analog.get("end_date", "")
            label = analog.get("label", "")

            if not start_str or not end_str:
                continue

            try:
                start_date = _parse_period_start(start_str)
                end_date = _parse_period_end(end_str)
            except (ValueError, AttributeError):
                logger.warning(
                    "BacktestWorkflow: could not parse analog dates: %s \u2013 %s",
                    start_str,
                    end_str,
                )
                agent_inferences.append(
                    f"[Agent inference] Could not parse dates for analog period "
                    f"'{label}' ({start_str} \u2013 {end_str}) — period excluded."
                )
                continue

            stats = _compute_period_stats(
                instrument_closes, start_date, end_date, primary_direction
            )

            if stats is None:
                agent_inferences.append(
                    f"[Agent inference] Insufficient price data for analog period "
                    f"'{label}' ({start_str} \u2013 {end_str}) — period excluded. "
                    "The instrument may not have existed or IBKR history does not "
                    "reach this far back."
                )
                continue

            period_entry: dict = {
                "label": label,
                "start_date": start_str,
                "end_date": end_str,
                **stats,
            }

            # SPY benchmark for this period
            if spy_closes is not None:
                spy_stats = _compute_period_stats(
                    spy_closes, start_date, end_date, "long"
                )
                if spy_stats is not None:
                    period_entry["spy_total_return"] = spy_stats["total_return"]
                    period_entry["spy_max_drawdown"] = spy_stats["max_drawdown"]

            # 60/40 benchmark (SPY 60% + AGG 40%) for this period
            if spy_closes is not None and agg_closes is not None:
                spy_stats_b = _compute_period_stats(
                    spy_closes, start_date, end_date, "long"
                )
                agg_stats = _compute_period_stats(
                    agg_closes, start_date, end_date, "long"
                )
                if spy_stats_b is not None and agg_stats is not None:
                    period_entry["benchmark_60_40_total_return"] = round(
                        float(
                            0.6 * spy_stats_b["total_return"]
                            + 0.4 * agg_stats["total_return"]
                        ),
                        6,
                    )

            period_results.append(period_entry)

        # --- Aggregate statistics (instrument only, not benchmark fields) ---
        instrument_period_stats = [
            {k: v for k, v in p.items() if k in _INSTRUMENT_STAT_KEYS}
            for p in period_results
        ]
        aggregate = _compute_aggregate_stats(instrument_period_stats)

        # Attach statistical limitation note when fewer than 5 periods
        n_periods = aggregate.get("n_periods", 0)
        if 0 < n_periods < _STATISTICAL_LIMITATION_THRESHOLD:
            aggregate["statistical_limitation_note"] = (
                f"This analysis is based on {n_periods} analog period(s). "
                "With fewer than 5 analog periods, statistical significance is "
                "limited. Treat these results as directional reference points, "
                "not predictive models."
            )
            agent_inferences.append(
                f"[Agent inference] Only {n_periods} analog period(s) with sufficient "
                "price history were available. Small sample size limits statistical "
                "conclusions."
            )

        # --- Benchmark comparison aggregates ---
        spy_returns = [
            p["spy_total_return"] for p in period_results if "spy_total_return" in p
        ]
        benchmark_60_40_returns = [
            p["benchmark_60_40_total_return"]
            for p in period_results
            if "benchmark_60_40_total_return" in p
        ]

        benchmark_comparison: dict = {}
        if spy_returns:
            benchmark_comparison["spy"] = _compute_benchmark_aggregate(spy_returns)
        if benchmark_60_40_returns:
            benchmark_comparison["60_40"] = _compute_benchmark_aggregate(
                benchmark_60_40_returns
            )

        # --- LLM narrative synthesis ---
        user_message = _build_user_message(
            thesis,
            primary_symbol,
            primary_direction,
            period_results,
            aggregate,
            benchmark_comparison,
        )

        response = anthropic.complete(
            messages=[{"role": "user", "content": user_message}],
            model=self.model,
            task_type="backtest",
            workflow_run_id=context.current_workflow_run_id,
            thesis_id=thesis.id,
            pod_id=getattr(thesis, "pod_id", None),
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
        )

        try:
            parsed = json.loads(response.content)
            analysis = parsed.get("analysis", response.content)
            llm_inferences = parsed.get("agent_inferences", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("BacktestWorkflow: LLM response was not valid JSON")
            analysis = response.content
            llm_inferences = []

        agent_inferences.extend(llm_inferences)

        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={
                "label": "Historical Analog Analysis",
                "instrument": primary_symbol,
                "direction": primary_direction,
                "analog_periods": period_results,
                "aggregate": aggregate,
                "benchmark_comparison": benchmark_comparison,
                "analysis": analysis,
            },
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=response.content,
        )
