"""RegimeStressTestWorkflow — Tier 2 deep dive (PRD Section 4.4).

Filters 30 years of macro history (from MacroContextWorkflow) into four
deterministic regimes — hiking cycle, cutting cycle, high inflation, low
inflation — and measures how the thesis instrument performed during each.
Regime classification is rule-based on FEDFUNDS and CPI YoY, not an LLM
judgment call, so the same inputs always produce the same regimes.

User-initiated only — never run automatically. Consumes MacroContextWorkflow
(FEDFUNDS/CPIAUCSL historical series) and InstrumentAnalysisWorkflow/thesis
(instrument, direction) from WorkflowContext. Fetches its own IBKR price
history, same as BacktestWorkflow and SensitivityAnalysisWorkflow.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd

from app.integrations.anthropic_client import AnthropicClient
from app.integrations.ibkr_client import IBKRClient, IBKRClientError
from app.workflows.analysis_utils import (
    bars_to_closes,
    compute_aggregate_stats,
    compute_period_stats,
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

_PRICE_PERIOD = "10y"
_BAR_SIZE = "1d"

# Minimum consecutive months for a regime run to be included.
_MIN_RUN_MONTHS = 3

# 6-month trailing FEDFUNDS change (percentage points) to classify a
# hiking/cutting cycle.
_HIKING_THRESHOLD = 0.25
_CUTTING_THRESHOLD = -0.25

# CPI YoY %% thresholds for high/low inflation regimes.
_HIGH_INFLATION_THRESHOLD = 4.0
_LOW_INFLATION_THRESHOLD = 2.0

REGIME_HIKING = "hiking_cycle"
REGIME_CUTTING = "cutting_cycle"
REGIME_HIGH_INFLATION = "high_inflation"
REGIME_LOW_INFLATION = "low_inflation"
_ALL_REGIMES = [
    REGIME_HIKING,
    REGIME_CUTTING,
    REGIME_HIGH_INFLATION,
    REGIME_LOW_INFLATION,
]

_SYSTEM_PROMPT = """\
You are a macro research analyst assessing how a thesis instrument performs \
under different rules-based macro regimes.

You will be given aggregate return/win-rate statistics for the instrument \
during four historical regimes: hiking cycles, cutting cycles, high \
inflation, and low inflation, each identified mechanically from FEDFUNDS \
and CPI data (not by judgment).

Write a 1-2 paragraph synthesis that:
- Identifies which regime(s) the instrument performed best/worst in
- Notes any regime with too little historical data to draw conclusions

Respond with a JSON object with exactly two keys:
- "analysis": 1-2 paragraph narrative.
- "agent_inferences": list of strings, each starting with "[Agent inference]"

Respond only with the JSON object. No markdown fences, no preamble.\
"""


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def _historical_data_to_series(historical_data: list[dict]) -> pd.Series:
    """Convert MacroContextWorkflow's historical_data list to a monthly Series
    indexed by month-end date."""
    if not historical_data:
        return pd.Series(dtype=float)
    dates = pd.to_datetime([d["date"] for d in historical_data])
    values = [d["value"] for d in historical_data]
    return pd.Series(values, index=dates).sort_index()


def classify_regimes(
    fedfunds_history: list[dict], cpi_history: list[dict]
) -> pd.DataFrame:
    """Build a monthly boolean DataFrame with one column per regime.

    Index is month-end date. A month may belong to multiple regimes
    simultaneously (e.g. hiking + high inflation).
    """
    fedfunds = _historical_data_to_series(fedfunds_history)
    cpi = _historical_data_to_series(cpi_history)

    if fedfunds.empty and cpi.empty:
        return pd.DataFrame()

    df = pd.DataFrame({"fedfunds": fedfunds, "cpi": cpi}).sort_index()

    fedfunds_delta_6m = df["fedfunds"] - df["fedfunds"].shift(6)
    cpi_yoy = df["cpi"].pct_change(12) * 100

    return pd.DataFrame(
        {
            REGIME_HIKING: fedfunds_delta_6m >= _HIKING_THRESHOLD,
            REGIME_CUTTING: fedfunds_delta_6m <= _CUTTING_THRESHOLD,
            REGIME_HIGH_INFLATION: cpi_yoy > _HIGH_INFLATION_THRESHOLD,
            REGIME_LOW_INFLATION: cpi_yoy <= _LOW_INFLATION_THRESHOLD,
        }
    ).fillna(False)


def find_runs(flags: pd.Series) -> list[tuple[date, date]]:
    """Return contiguous True runs of at least _MIN_RUN_MONTHS length.

    Each run is (start_date, end_date): start_date is the first day of the
    first month in the run; end_date is the (month-end) date of the last
    month in the run, taken directly from the series index.
    """
    if flags.empty:
        return []

    runs: list[tuple[date, date]] = []
    run_start_idx: int = -1

    values = flags.tolist()
    index = flags.index

    for i, val in enumerate(values):
        if val and run_start_idx == -1:
            run_start_idx = i
        elif not val and run_start_idx != -1:
            _append_run_if_long_enough(runs, index, run_start_idx, i - 1)
            run_start_idx = -1

    if run_start_idx != -1:
        _append_run_if_long_enough(runs, index, run_start_idx, len(values) - 1)

    return runs


def _append_run_if_long_enough(
    runs: list[tuple[date, date]], index: pd.Index, start_idx: int, end_idx: int
) -> None:
    if (end_idx - start_idx + 1) < _MIN_RUN_MONTHS:
        return
    start_ts = index[start_idx]
    end_ts = index[end_idx]
    runs.append((date(start_ts.year, start_ts.month, 1), end_ts.date()))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_user_message(instrument: str, direction: str, regimes_out: dict) -> str:
    lines = [f"Instrument: {instrument} (direction: {direction})", ""]
    for regime_name in _ALL_REGIMES:
        entry = regimes_out.get(regime_name, {})
        aggregate = entry.get("aggregate", {})
        lines.append(f"{regime_name}: {entry.get('n_runs', 0)} run(s) identified")
        if aggregate:
            lines.append(
                f"  avg_return={aggregate.get('avg_return', 0):.2%} "
                f"win_rate={aggregate.get('win_rate', 0):.0%} "
                f"n_periods={aggregate.get('n_periods', 0)}"
            )
        else:
            lines.append("  no periods with sufficient price history")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class RegimeStressTestWorkflow(BaseWorkflow):
    """Filters macro history into rule-based regimes and measures instrument
    performance during each.

    Outputs (structured_output):
        instrument (str), direction (str)
        regimes (dict): keyed by regime name (hiking_cycle, cutting_cycle,
            high_inflation, low_inflation), each with n_runs (contiguous
            runs identified), runs (list of {start_date, end_date, ...stats}
            for runs with sufficient price data), and aggregate (from
            analysis_utils, or {} if no run had valid data).
        analysis (str): LLM narrative comparing regime performance.
    """

    name = "RegimeStressTestWorkflow"
    description = (
        "Deep dive: filters macro history into rule-based regimes (hiking "
        "cycle, cutting cycle, high inflation, low inflation) using FEDFUNDS "
        "and CPI data, and measures thesis instrument performance in each."
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
                    "RegimeStressTestWorkflow requires context.pod_settings to "
                    "construct IBKRClient — ensure the caller sets it."
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

        macro_result = context.get_result("MacroContextWorkflow")
        instrument_result = context.get_result("InstrumentAnalysisWorkflow")

        series: dict = (
            macro_result.structured_output.get("series", {})
            if macro_result is not None
            else {}
        )
        if macro_result is None:
            agent_inferences.append(
                "[Agent inference] MacroContextWorkflow result was not "
                "available — regime classification could not be performed."
            )

        instruments_info: dict[str, dict] = {}
        if instrument_result is not None:
            instruments_info = instrument_result.structured_output.get(
                "instruments", {}
            )
        if not instruments_info:
            for instr in list(getattr(thesis, "instruments", []) or []):
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
                instruments_info[instr.instrument] = {
                    "direction": direction,
                    "role": role,
                }

        primary_symbol: str | None = None
        primary_direction = "long"
        for symbol, info in instruments_info.items():
            if info.get("role", "primary") == "primary":
                primary_symbol = symbol
                primary_direction = info.get("direction", "long")
                break
        if primary_symbol is None and instruments_info:
            primary_symbol, _info = next(iter(instruments_info.items()))
            primary_direction = _info.get("direction", "long")

        if primary_symbol is None:
            agent_inferences.append(
                "[Agent inference] No instrument data was available — regime "
                "stress test could not be performed."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": None,
                    "direction": primary_direction,
                    "regimes": {},
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        regime_flags = classify_regimes(
            series.get("FEDFUNDS", {}).get("historical_data", []),
            series.get("CPIAUCSL", {}).get("historical_data", []),
        )

        if regime_flags.empty:
            agent_inferences.append(
                "[Agent inference] FEDFUNDS/CPI history was not available — "
                "regime classification could not be performed."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": primary_symbol,
                    "direction": primary_direction,
                    "regimes": {},
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        try:
            price_history = ibkr.get_price_history(
                symbol=primary_symbol, period=_PRICE_PERIOD, bar_size=_BAR_SIZE
            )
        except IBKRClientError as exc:
            logger.warning(
                "RegimeStressTestWorkflow: failed to fetch price history for " "%s: %s",
                primary_symbol,
                exc,
            )
            agent_inferences.append(
                f"[Agent inference] Price history for {primary_symbol} could "
                "not be retrieved from IBKR — regime stress test is unavailable."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": primary_symbol,
                    "direction": primary_direction,
                    "regimes": {},
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        citations.append(
            Citation(
                source_type=CitationSourceType.IBKR,
                label=(
                    f"IBKR:{primary_symbol} price_history, "
                    f"{price_history.retrieved_at.isoformat()}"
                ),
                url=None,
                retrieval_date=price_history.retrieved_at.date(),
            )
        )
        if macro_result is not None:
            for series_id in ("FEDFUNDS", "CPIAUCSL"):
                for c in macro_result.citations:
                    if (
                        c.source_type == CitationSourceType.FRED
                        and series_id in c.label
                    ):
                        citations.append(c)
                        break

        closes = bars_to_closes(price_history.bars)

        regimes_out: dict = {}
        for regime_name in _ALL_REGIMES:
            runs = find_runs(regime_flags[regime_name])
            run_stats: list[dict] = []
            for start_date, end_date in runs:
                stats = compute_period_stats(
                    closes, start_date, end_date, primary_direction
                )
                if stats is not None:
                    run_stats.append(
                        {
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                            **stats,
                        }
                    )
            regimes_out[regime_name] = {
                "n_runs": len(runs),
                "runs": run_stats,
                "aggregate": compute_aggregate_stats(run_stats),
            }

        if not any(entry["aggregate"] for entry in regimes_out.values()):
            agent_inferences.append(
                "[Agent inference] No regime run had sufficient instrument "
                "price history for analysis."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": primary_symbol,
                    "direction": primary_direction,
                    "regimes": regimes_out,
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        user_message = _build_user_message(
            primary_symbol, primary_direction, regimes_out
        )
        response = anthropic.complete(
            messages=[{"role": "user", "content": user_message}],
            model=self.model,
            task_type="regime_stress_test",
            workflow_run_id=context.current_workflow_run_id,
            thesis_id=thesis.id,
            pod_id=getattr(thesis, "pod_id", None),
            max_tokens=768,
            system=_SYSTEM_PROMPT,
        )

        try:
            parsed = json.loads(response.content)
            analysis = parsed.get("analysis", response.content)
            llm_inferences = parsed.get("agent_inferences", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("RegimeStressTestWorkflow: LLM response was not valid JSON")
            analysis = response.content
            llm_inferences = []

        agent_inferences.extend(llm_inferences)

        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={
                "instrument": primary_symbol,
                "direction": primary_direction,
                "regimes": regimes_out,
                "analysis": analysis,
            },
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=response.content,
        )
