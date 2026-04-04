"""InstrumentAnalysisWorkflow — pulls ETF price history and assesses instrument fit.

This is the third workflow in the research chain. It pulls 5-year daily price
history from IBKR for each thesis instrument and computes:
  - Annualized return
  - Annualized volatility
  - Maximum drawdown
  - 60-day realized volatility (used later by position sizing)

It also correlates monthly instrument returns against the FRED macro series
from MacroContextWorkflow to assess how the instrument behaves in the macro
conditions the thesis describes.

If no instruments are attached to the thesis, the workflow returns a partial
result with an explanatory agent inference rather than raising.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from app.integrations.anthropic_client import AnthropicClient
from app.integrations.ibkr_client import IBKRBar, IBKRClient, IBKRClientError
from app.workflows.base import (
    BaseWorkflow,
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

# 5-year lookback for price history — matches ticket requirement
_PRICE_HISTORY_PERIOD = "5y"
_BAR_SIZE = "1d"

# Minimum bars to compute meaningful statistics
_MIN_BARS = 60

# Minimum months of overlap needed to include a correlation value
_MIN_CORR_MONTHS = 12

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a macro research analyst assessing whether an ETF instrument is \
well-suited to express a trade thesis.

You will be given:
1. A trade thesis with direction and time horizon
2. Price statistics for the thesis instrument(s): annualized return, volatility, \
max drawdown, 60-day realized vol
3. Correlations between monthly instrument returns and monthly changes in core \
FRED macro series (Pearson r, range -1 to +1)

Your task: analyze whether the instrument's historical behavior is consistent \
with the thesis direction and macro assumptions.

Respond with a JSON object with exactly two keys:
- "analysis": A 2-3 paragraph analysis covering: (1) whether the instrument's \
historical returns and vol are appropriate for the thesis, (2) what the macro \
correlations suggest about how the instrument behaves during the macro \
conditions the thesis describes, (3) any concerns about instrument fit.
- "agent_inferences": A list of strings for any interpretive judgments beyond \
the raw statistics. Each MUST start with "[Agent inference]".

Respond only with the JSON object. No markdown fences, no preamble.\
"""


def _build_user_message(
    thesis,
    instruments_stats: dict[str, dict],
    correlations: dict[str, dict[str, float]],
) -> str:
    direction = (
        thesis.direction.value
        if hasattr(thesis.direction, "value")
        else str(thesis.direction)
    )
    notes = thesis.notes or "(none)"

    stats_lines: list[str] = []
    for symbol, stats in instruments_stats.items():
        stats_lines.append(
            f"  {symbol} (direction: {stats['direction']}, role: {stats['role']}):\n"
            f"    Annualized return (5y): {stats['annualized_return']:.2%}\n"
            f"    Annualized volatility (5y): {stats['annualized_vol']:.2%}\n"
            f"    Maximum drawdown (5y): {stats['max_drawdown']:.2%}\n"
            f"    60-day realized vol: {stats['vol_60d_realized']:.2%}\n"
            f"    Period: {stats['start_date']} to {stats['end_date']} "
            f"({stats['n_trading_days']} trading days)"
        )

    corr_lines: list[str] = []
    for symbol, corrs in correlations.items():
        if corrs:
            corr_str = ", ".join(f"{k}: {v:+.3f}" for k, v in corrs.items())
            corr_lines.append(f"  {symbol}: {corr_str}")
        else:
            corr_lines.append(f"  {symbol}: (insufficient data for correlation)")

    stats_block = "\n".join(stats_lines) if stats_lines else "  (no data)"
    corr_block = "\n".join(corr_lines) if corr_lines else "  (no macro data available)"

    return (
        f"Thesis: {thesis.title}\n"
        f"Direction: {direction}\n"
        f"Time Horizon: {thesis.time_horizon}\n"
        f"Notes: {notes}\n\n"
        f"Instrument statistics (5-year history):\n{stats_block}\n\n"
        "Correlations (monthly instrument returns vs. "
        f"month-over-month macro changes):\n{corr_block}"
    )


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


def _compute_instrument_stats(bars: list[IBKRBar], symbol: str) -> dict | None:
    """Compute price statistics from a list of IBKR daily bars.

    Returns None if there are fewer than _MIN_BARS bars (insufficient data).
    All returns and vols are annualized (252 trading days per year convention).
    """
    if len(bars) < _MIN_BARS:
        logger.warning(
            "Insufficient bars for %s: got %d, need %d minimum",
            symbol,
            len(bars),
            _MIN_BARS,
        )
        return None

    timestamps = [b.timestamp for b in bars]
    closes = pd.Series(
        [b.close for b in bars],
        index=pd.DatetimeIndex(timestamps),
        dtype=float,
    ).sort_index()

    closes = closes[closes > 0].dropna()
    if len(closes) < _MIN_BARS:
        return None

    log_returns = np.log(closes / closes.shift(1)).dropna()
    n_days = len(closes)

    # Annualized return (geometric, 252-day convention)
    total_return = closes.iloc[-1] / closes.iloc[0] - 1.0
    n_years = n_days / 252.0
    annualized_return = (1.0 + total_return) ** (1.0 / n_years) - 1.0

    # Annualized volatility
    annualized_vol = float(log_returns.std()) * np.sqrt(252.0)

    # Max drawdown: largest peak-to-trough decline in cumulative return
    cum = (1.0 + log_returns).cumprod()
    rolling_peak = cum.expanding().max()
    drawdowns = (cum - rolling_peak) / rolling_peak
    max_drawdown = float(drawdowns.min())

    # 60-day realized volatility
    last_60 = log_returns.iloc[-60:]
    vol_60d = (
        float(last_60.std()) * np.sqrt(252.0) if len(last_60) >= 20 else float("nan")
    )

    return {
        "annualized_return": round(float(annualized_return), 6),
        "annualized_vol": round(float(annualized_vol), 6),
        "max_drawdown": round(float(max_drawdown), 6),
        "vol_60d_realized": round(float(vol_60d), 6) if not np.isnan(vol_60d) else None,
        "start_date": closes.index[0].date().isoformat(),
        "end_date": closes.index[-1].date().isoformat(),
        "n_trading_days": n_days,
    }


def _compute_correlations(
    bars: list[IBKRBar],
    macro_series: dict[str, dict],
) -> dict[str, float]:
    """Compute Pearson correlation between monthly instrument returns and
    month-over-month changes in each FRED macro series.

    Returns a dict of {series_id: correlation_coefficient}. Series with fewer
    than _MIN_CORR_MONTHS of overlapping observations are omitted.
    """
    if not bars or not macro_series:
        return {}

    timestamps = [b.timestamp for b in bars]
    closes = pd.Series(
        [b.close for b in bars],
        index=pd.DatetimeIndex(timestamps),
        dtype=float,
    ).sort_index()

    # Monthly instrument returns (pct_change of month-end close).
    # Strip timezone so the index aligns with tz-naive FRED series dates.
    tz_naive = closes.copy()
    tz_naive.index = (
        closes.index.tz_localize(None)
        if closes.index.tz is None
        else closes.index.tz_convert(None)
    )
    monthly_closes = tz_naive.resample("ME").last().dropna()
    if len(monthly_closes) < 2:
        return {}
    instrument_monthly = monthly_closes.pct_change().dropna()

    correlations: dict[str, float] = {}
    for series_id, series_dict in macro_series.items():
        # Skip OECD series — correlate only with core FRED series
        if series_id.startswith("OECD:"):
            continue

        hist = series_dict.get("historical_data", [])
        if not hist:
            continue

        macro_dates = pd.to_datetime([d["date"] for d in hist])
        macro_values = pd.Series(
            [d["value"] for d in hist],
            index=macro_dates,
            dtype=float,
        ).sort_index()

        # Month-over-month change in the macro series
        macro_monthly_change = macro_values.resample("ME").last().diff().dropna()

        # Align on common dates
        aligned = pd.DataFrame(
            {"instrument": instrument_monthly, "macro": macro_monthly_change}
        ).dropna()

        if len(aligned) < _MIN_CORR_MONTHS:
            continue

        corr = aligned["instrument"].corr(aligned["macro"])
        if not np.isnan(corr):
            correlations[series_id] = round(float(corr), 4)

    return correlations


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class InstrumentAnalysisWorkflow(BaseWorkflow):
    """Pull 5-year ETF price history from IBKR and assess instrument fit.

    Computes price statistics for each thesis instrument and correlates
    monthly returns against FRED macro series from MacroContextWorkflow.
    Passes the resulting picture to the LLM for an assessment of whether
    the instrument is a good fit for the thesis direction and assumptions.

    Outputs (structured_output):
        instruments (dict[str, dict]): per-instrument price statistics including
            annualized_return, annualized_vol, max_drawdown, vol_60d_realized,
            start_date, end_date, n_trading_days, direction, role.
        correlations (dict[str, dict]): per-instrument correlation of monthly
            returns vs. month-over-month FRED series changes.
        analysis (str): LLM narrative assessing instrument fit.
    """

    name = "InstrumentAnalysisWorkflow"
    description = (
        "Pulls 5-year daily price history from IBKR for each thesis instrument; "
        "computes annualized return, vol, max drawdown, and 60-day realized vol; "
        "correlates monthly returns against FRED macro series."
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

    def execute(self, thesis, context: WorkflowContext) -> WorkflowResult:
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
                    "InstrumentAnalysisWorkflow requires context.pod_settings "
                    "to construct IBKRClient — ensure WorkflowRunner sets it."
                )
            ibkr = IBKRClient(
                base_url=settings.ibkr_base_url,
                account_id=settings.ibkr_account_id,
                paper_account_id=settings.ibkr_paper_account_id,
                pod_settings=context.pod_settings,
            )
        else:
            ibkr = self._ibkr

        # --- Gather instruments from thesis ---
        instruments = list(getattr(thesis, "instruments", []) or [])
        if not instruments:
            logger.warning(
                "InstrumentAnalysisWorkflow: no instruments on thesis %s", thesis.id
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instruments": {},
                    "correlations": {},
                    "analysis": "",
                },
                citations=[],
                agent_inferences=[
                    "[Agent inference] No instruments were attached to this thesis — "
                    "instrument analysis could not be performed."
                ],
                raw_output="",
            )

        # --- Pull macro series from context for correlation ---
        macro_result = context.get_result("MacroContextWorkflow")
        macro_series: dict = (
            macro_result.structured_output.get("series", {})
            if macro_result is not None
            else {}
        )

        # --- Fetch price history and compute statistics ---
        instruments_stats: dict[str, dict] = {}
        correlations: dict[str, dict[str, float]] = {}
        citations: list[Citation] = []
        agent_inferences: list[str] = []

        for instr in instruments:
            symbol: str = instr.instrument
            direction = (
                instr.direction.value
                if hasattr(instr.direction, "value")
                else str(instr.direction)
            )
            role = instr.role.value if hasattr(instr.role, "value") else str(instr.role)

            try:
                price_history = ibkr.get_price_history(
                    symbol=symbol,
                    period=_PRICE_HISTORY_PERIOD,
                    bar_size=_BAR_SIZE,
                )
            except IBKRClientError as exc:
                logger.warning("Failed to fetch price history for %s: %s", symbol, exc)
                agent_inferences.append(
                    f"[Agent inference] Price history for {symbol} could not be "
                    f"retrieved from IBKR — instrument statistics are unavailable."
                )
                continue

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

            stats = _compute_instrument_stats(price_history.bars, symbol)
            if stats is None:
                n_bars = len(price_history.bars)
                agent_inferences.append(
                    f"[Agent inference] Insufficient price history for {symbol} "
                    f"({n_bars} bars) — statistics could not be computed."
                )
                continue

            stats["direction"] = direction
            stats["role"] = role
            instruments_stats[symbol] = stats

            corr = _compute_correlations(price_history.bars, macro_series)
            correlations[symbol] = corr

        # --- LLM analysis ---
        user_message = _build_user_message(thesis, instruments_stats, correlations)
        response = anthropic.complete(
            messages=[{"role": "user", "content": user_message}],
            model=self.model,
            task_type="instrument_analysis",
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
            logger.warning(
                "InstrumentAnalysisWorkflow: LLM response was not valid JSON"
            )
            analysis = response.content
            llm_inferences = []

        agent_inferences.extend(llm_inferences)

        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={
                "instruments": instruments_stats,
                "correlations": correlations,
                "analysis": analysis,
            },
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=response.content,
        )
