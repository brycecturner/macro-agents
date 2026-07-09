"""Shared price-history and analog-period math used across workflows that
measure instrument performance during historical macro analog periods.

Originally defined inside BacktestWorkflow; extracted here so the Tier 2
deep dive workflows (SensitivityAnalysisWorkflow, RegimeStressTestWorkflow,
HistoricalAnalogDetailWorkflow) can reuse the same return/drawdown/volatility
calculations instead of duplicating them. BacktestWorkflow imports from this
module too — the calculations themselves are unchanged.
"""

from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import pandas as pd

from app.integrations.ibkr_client import IBKRBar

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def parse_period_start(period_str: str) -> date:
    """Parse 'YYYY-MM' to the first calendar day of that month."""
    year, month = map(int, period_str.split("-"))
    return date(year, month, 1)


def parse_period_end(period_str: str) -> date:
    """Parse 'YYYY-MM' to the last calendar day of that month."""
    year, month = map(int, period_str.split("-"))
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, last_day)


# ---------------------------------------------------------------------------
# Price series helpers
# ---------------------------------------------------------------------------


def bars_to_closes(bars: list[IBKRBar]) -> pd.Series:
    """Convert a list of IBKR daily bars to a tz-naive close price Series.

    Strips timezone from the index so dates can be compared directly to
    ``datetime.date`` objects from the analog period definitions.
    Returns an empty Series if bars is empty.
    """
    if not bars:
        return pd.Series(dtype=float)

    timestamps = [b.timestamp for b in bars]
    closes = pd.Series(
        [b.close for b in bars],
        index=pd.DatetimeIndex(timestamps),
        dtype=float,
    ).sort_index()

    # Normalise to tz-naive — same pattern used by InstrumentAnalysisWorkflow
    closes.index = (
        closes.index.tz_localize(None)
        if closes.index.tz is None
        else closes.index.tz_convert(None)
    )
    return closes[closes > 0].dropna()


# ---------------------------------------------------------------------------
# Per-period computation
# ---------------------------------------------------------------------------


def compute_period_stats(
    closes: pd.Series,
    start_date: date,
    end_date: date,
    direction: str,
) -> dict | None:
    """Compute performance statistics for a single analog period.

    Returns None when fewer than 2 trading days of data exist within
    [start_date, end_date] — indicating missing or insufficient price history.

    Args:
        closes: Tz-naive daily close price Series.
        start_date: First calendar day of the analog period.
        end_date: Last calendar day of the analog period.
        direction: "long" or "short" — determines directional correctness.

    Returns:
        Dict with keys: total_return, annualized_return, max_drawdown,
        volatility, directionally_correct, n_trading_days. Or None.
    """
    mask = (closes.index.date >= start_date) & (closes.index.date <= end_date)
    period_closes = closes[mask]

    if len(period_closes) < 2:
        return None

    total_return = float(period_closes.iloc[-1] / period_closes.iloc[0] - 1.0)
    n_days = len(period_closes)
    # Guard against near-zero n_years to avoid exponentiation overflow
    n_years = max(n_days / 252.0, 1.0 / 252.0)
    annualized_return = float((1.0 + total_return) ** (1.0 / n_years) - 1.0)

    log_returns = np.log(period_closes / period_closes.shift(1)).dropna()

    # Max drawdown: largest peak-to-trough decline in cumulative log-return path
    if len(log_returns) > 0:
        cum = (1.0 + log_returns).cumprod()
        rolling_peak = cum.expanding().max()
        drawdowns = (cum - rolling_peak) / rolling_peak
        max_drawdown = float(drawdowns.min())
    else:
        max_drawdown = 0.0

    # Annualized volatility (252-day convention)
    volatility = (
        float(log_returns.std()) * np.sqrt(252.0) if len(log_returns) > 1 else 0.0
    )

    # Directional correctness relative to thesis direction
    directionally_correct = (direction == "long" and total_return > 0) or (
        direction == "short" and total_return < 0
    )

    return {
        "total_return": round(total_return, 6),
        "annualized_return": round(annualized_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "volatility": round(volatility, 6),
        "directionally_correct": directionally_correct,
        "n_trading_days": int(n_days),
    }


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------


def compute_aggregate_stats(period_stats: list[dict]) -> dict:
    """Compute aggregate statistics across all valid analog period results.

    Returns an empty dict if period_stats is empty.
    """
    if not period_stats:
        return {}

    returns = [s["total_return"] for s in period_stats]
    drawdowns = [s["max_drawdown"] for s in period_stats]
    n = len(returns)
    win_count = sum(1 for s in period_stats if s["directionally_correct"])

    return {
        "n_periods": n,
        "avg_return": round(float(np.mean(returns)), 6),
        "worst_return": round(float(min(returns)), 6),
        "best_return": round(float(max(returns)), 6),
        "win_rate": round(float(win_count / n), 4),
        "avg_max_drawdown": round(float(np.mean(drawdowns)), 6),
    }


def compute_benchmark_aggregate(returns: list[float]) -> dict:
    """Compute aggregate return statistics for a benchmark over analog periods.

    Win rate counts periods where the benchmark return was positive.
    Returns an empty dict if returns is empty.
    """
    if not returns:
        return {}
    n = len(returns)
    return {
        "n_periods": n,
        "avg_return": round(float(np.mean(returns)), 6),
        "worst_return": round(float(min(returns)), 6),
        "best_return": round(float(max(returns)), 6),
        "win_rate": round(float(sum(1 for r in returns if r > 0) / n), 4),
    }
