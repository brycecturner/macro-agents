"""SensitivityAnalysisWorkflow — Tier 2 deep dive (PRD Section 4.4).

Reruns the Historical Analog Analysis math from BacktestWorkflow while
shifting the entry (start) date of each analog period by ±1, ±2, and ±3
months, holding the exit (end) date fixed. This tests how sensitive the
result is to exactly when the thesis is entered relative to the identified
macro analog — a thesis whose apparent edge evaporates when entry timing
shifts by a month or two is a weaker thesis than one that is robust to it.

User-initiated only — never run automatically. Consumes HistoricalAnalogWorkflow
(analog periods) and InstrumentAnalysisWorkflow/thesis (instrument, direction)
from WorkflowContext, exactly like BacktestWorkflow. Fetches its own IBKR price
history since raw closes are not passed through structured_output between
workflows (only computed summary stats are).
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from app.integrations.anthropic_client import AnthropicClient
from app.integrations.ibkr_client import IBKRClient, IBKRClientError
from app.workflows.analysis_utils import (
    bars_to_closes,
    compute_aggregate_stats,
    compute_period_stats,
    parse_period_end,
    parse_period_start,
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

# Entry timing offsets to test, in months. 0 is the original (unshifted) entry.
_OFFSETS_MONTHS = [-3, -2, -1, 0, 1, 2, 3]

_SYSTEM_PROMPT = """\
You are a macro research analyst assessing how sensitive a Historical Analog \
Analysis is to entry timing.

You will be given the average return and win rate at each of several entry \
timing offsets (in months, relative to the originally identified analog \
periods), holding the exit date fixed.

Write a 1-2 paragraph synthesis that:
- States whether the thesis's apparent edge is robust across nearby entry \
timings or fragile (i.e. concentrated at only the exact original entry point)
- Calls out the offset(s) that perform notably better or worse than the rest

Respond with a JSON object with exactly two keys:
- "analysis": 1-2 paragraph narrative.
- "agent_inferences": list of strings, each starting with "[Agent inference]"

Respond only with the JSON object. No markdown fences, no preamble.\
"""


def _shift_months(d, months: int):
    """Shift a date by a whole number of months, clamping to month length."""
    return (pd.Timestamp(d) + pd.DateOffset(months=months)).date()


def _build_user_message(instrument: str, direction: str, offsets: list[dict]) -> str:
    lines = [
        f"Instrument: {instrument} (direction: {direction})",
        "",
        "Average return and win rate by entry timing offset:",
    ]
    for entry in offsets:
        agg = entry["aggregate"]
        offset = entry["offset_months"]
        label = "original entry" if offset == 0 else f"{offset:+d} month(s)"
        if not agg:
            lines.append(f"  {label}: no valid periods")
            continue
        lines.append(
            f"  {label}: avg_return={agg.get('avg_return', 0):.2%} "
            f"win_rate={agg.get('win_rate', 0):.0%} "
            f"n_periods={agg.get('n_periods', 0)}"
        )
    return "\n".join(lines)


class SensitivityAnalysisWorkflow(BaseWorkflow):
    """Reruns the Historical Analog Analysis with entry timing shifted.

    Outputs (structured_output):
        instrument (str), direction (str)
        offsets (list[dict]): one entry per offset in _OFFSETS_MONTHS, each
            with offset_months and aggregate (from analysis_utils, or {} if
            no analog period had valid data at that offset).
        analysis (str): LLM narrative on robustness to entry timing.
    """

    name = "SensitivityAnalysisWorkflow"
    description = (
        "Deep dive: reruns the Historical Analog Analysis with each analog "
        "period's entry date shifted by ±1, ±2, and ±3 months, to test "
        "sensitivity to entry timing."
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
                    "SensitivityAnalysisWorkflow requires context.pod_settings "
                    "to construct IBKRClient — ensure the caller sets it."
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

        analog_result = context.get_result("HistoricalAnalogWorkflow")
        instrument_result = context.get_result("InstrumentAnalysisWorkflow")

        analogs: list[dict] = (
            analog_result.structured_output.get("analogs", [])
            if analog_result is not None
            else []
        )
        if analog_result is None:
            agent_inferences.append(
                "[Agent inference] HistoricalAnalogWorkflow result was not "
                "available — sensitivity analysis could not be performed."
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

        if not analogs or primary_symbol is None:
            if primary_symbol is None:
                agent_inferences.append(
                    "[Agent inference] No instrument data was available — "
                    "sensitivity analysis could not be performed."
                )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": primary_symbol,
                    "direction": primary_direction,
                    "offsets": [],
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
                "SensitivityAnalysisWorkflow: failed to fetch price history for "
                "%s: %s",
                primary_symbol,
                exc,
            )
            agent_inferences.append(
                f"[Agent inference] Price history for {primary_symbol} could not "
                "be retrieved from IBKR — sensitivity analysis is unavailable."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": primary_symbol,
                    "direction": primary_direction,
                    "offsets": [],
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
        closes = bars_to_closes(price_history.bars)

        offsets_out: list[dict] = []
        for offset in _OFFSETS_MONTHS:
            period_stats: list[dict] = []
            for analog in analogs:
                start_str = analog.get("start_date", "")
                end_str = analog.get("end_date", "")
                if not start_str or not end_str:
                    continue
                try:
                    start_date = parse_period_start(start_str)
                    end_date = parse_period_end(end_str)
                except (ValueError, AttributeError):
                    continue

                shifted_start = _shift_months(start_date, offset)
                if shifted_start >= end_date:
                    continue

                stats = compute_period_stats(
                    closes, shifted_start, end_date, primary_direction
                )
                if stats is not None:
                    period_stats.append(stats)

            offsets_out.append(
                {
                    "offset_months": offset,
                    "aggregate": compute_aggregate_stats(period_stats),
                }
            )

        if not any(entry["aggregate"] for entry in offsets_out):
            agent_inferences.append(
                "[Agent inference] No analog period had sufficient price "
                "history at any tested entry offset — sensitivity analysis "
                "could not be performed."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": primary_symbol,
                    "direction": primary_direction,
                    "offsets": offsets_out,
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        user_message = _build_user_message(
            primary_symbol, primary_direction, offsets_out
        )
        response = anthropic.complete(
            messages=[{"role": "user", "content": user_message}],
            model=self.model,
            task_type="sensitivity_analysis",
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
            logger.warning(
                "SensitivityAnalysisWorkflow: LLM response was not valid JSON"
            )
            analysis = response.content
            llm_inferences = []

        agent_inferences.extend(llm_inferences)

        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={
                "instrument": primary_symbol,
                "direction": primary_direction,
                "offsets": offsets_out,
                "analysis": analysis,
            },
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=response.content,
        )
