"""PortfolioCorrelationWorkflow — Tier 2 deep dive (PRD Section 4.4).

Computes the correlation between the thesis's primary instrument and every
instrument currently held in a position elsewhere in the same pod, to catch
inadvertent factor doubling before a new thesis is approved.

User-initiated only — never run automatically. Queries the `positions` table
directly via context.db (positions belonging to other theses in the same
pod) rather than consuming a prior WorkflowResult, since portfolio state is
not itself a research workflow output. Fetches its own IBKR price history
for correlation inputs.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from app.integrations.anthropic_client import AnthropicClient
from app.integrations.ibkr_client import IBKRClient, IBKRClientError
from app.models.execution import Position
from app.workflows.analysis_utils import bars_to_closes
from app.workflows.base import (
    BaseWorkflow,
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)

_PRICE_PERIOD = "1y"
_BAR_SIZE = "1d"

# Minimum overlapping trading days required to compute a meaningful correlation.
_MIN_OVERLAP_DAYS = 20

# |correlation| above this is flagged in the LLM prompt as a risk of factor doubling.
_HIGH_CORRELATION_THRESHOLD = 0.7

_SYSTEM_PROMPT = """\
You are a macro research analyst checking a proposed thesis for portfolio \
concentration risk.

You will be given the correlation between the thesis instrument's daily \
returns and the daily returns of every other instrument currently held \
elsewhere in the pod.

Write a 1 paragraph synthesis that:
- Flags any instrument with |correlation| above 0.7 as a factor-doubling risk
- Notes if the portfolio shows no meaningful overlap with this thesis

Respond with a JSON object with exactly two keys:
- "analysis": 1 paragraph narrative.
- "agent_inferences": list of strings, each starting with "[Agent inference]"

Respond only with the JSON object. No markdown fences, no preamble.\
"""


def _compute_correlation(
    closes_a: pd.Series, closes_b: pd.Series
) -> tuple[float, int] | None:
    """Pearson correlation of daily log returns over the overlapping dates.

    Returns None if fewer than _MIN_OVERLAP_DAYS overlapping observations
    exist after aligning and differencing.
    """
    aligned = pd.DataFrame({"a": closes_a, "b": closes_b}).dropna()
    if len(aligned) < _MIN_OVERLAP_DAYS + 1:
        return None

    returns = np.log(aligned / aligned.shift(1)).dropna()
    if len(returns) < _MIN_OVERLAP_DAYS:
        return None

    corr = float(returns["a"].corr(returns["b"]))
    if pd.isna(corr):
        return None
    return round(corr, 4), int(len(returns))


def _build_user_message(instrument: str, correlations: dict[str, dict]) -> str:
    if not correlations:
        return f"Instrument: {instrument}\n\nNo other positions found in this pod."
    lines = [f"Instrument: {instrument}", "", "Correlation to other pod positions:"]
    for other_symbol, entry in correlations.items():
        flag = (
            "  <- HIGH CORRELATION RISK"
            if abs(entry["correlation"]) > _HIGH_CORRELATION_THRESHOLD
            else ""
        )
        lines.append(
            f"  {other_symbol}: correlation={entry['correlation']:.2f} "
            f"(n={entry['n_overlapping_days']} overlapping days){flag}"
        )
    return "\n".join(lines)


class PortfolioCorrelationWorkflow(BaseWorkflow):
    """Correlates the thesis instrument against every other position in the pod.

    Outputs (structured_output):
        instrument (str): Primary thesis instrument.
        correlations (dict): keyed by other instrument symbol, each with
            correlation (float, -1..1) and n_overlapping_days (int).
        analysis (str): LLM narrative flagging high-correlation risk pairs.
    """

    name = "PortfolioCorrelationWorkflow"
    description = (
        "Deep dive: computes correlation between the thesis instrument and "
        "every instrument currently held in another position in the same "
        "pod, to catch inadvertent factor doubling."
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
                    "PortfolioCorrelationWorkflow requires context.pod_settings "
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

        instrument_result = context.get_result("InstrumentAnalysisWorkflow")
        instruments_info: dict[str, dict] = {}
        if instrument_result is not None:
            instruments_info = instrument_result.structured_output.get(
                "instruments", {}
            )
        if not instruments_info:
            for instr in list(getattr(thesis, "instruments", []) or []):
                role = (
                    instr.role.value
                    if hasattr(instr.role, "value")
                    else str(instr.role)
                )
                instruments_info[instr.instrument] = {"role": role}

        primary_symbol: str | None = None
        for symbol, info in instruments_info.items():
            if info.get("role", "primary") == "primary":
                primary_symbol = symbol
                break
        if primary_symbol is None and instruments_info:
            primary_symbol = next(iter(instruments_info))

        if primary_symbol is None:
            agent_inferences.append(
                "[Agent inference] No instrument data was available — "
                "portfolio correlation check could not be performed."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": None,
                    "correlations": {},
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        other_symbols: set[str] = set()
        if context.db is not None:
            positions = (
                context.db.query(Position)
                .filter(
                    Position.pod_id == thesis.pod_id,
                    Position.thesis_id != thesis.id,
                )
                .all()
            )
            other_symbols = {
                p.instrument for p in positions if p.instrument != primary_symbol
            }

        if not other_symbols:
            agent_inferences.append(
                "[Agent inference] No other active positions were found in "
                "this pod — there is nothing to compare correlation against."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": primary_symbol,
                    "correlations": {},
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        try:
            primary_history = ibkr.get_price_history(
                symbol=primary_symbol, period=_PRICE_PERIOD, bar_size=_BAR_SIZE
            )
        except IBKRClientError as exc:
            logger.warning(
                "PortfolioCorrelationWorkflow: failed to fetch price history "
                "for %s: %s",
                primary_symbol,
                exc,
            )
            agent_inferences.append(
                f"[Agent inference] Price history for {primary_symbol} could "
                "not be retrieved from IBKR — correlation check is unavailable."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": primary_symbol,
                    "correlations": {},
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
                    f"{primary_history.retrieved_at.isoformat()}"
                ),
                url=None,
                retrieval_date=primary_history.retrieved_at.date(),
            )
        )
        primary_closes = bars_to_closes(primary_history.bars)

        correlations: dict[str, dict] = {}
        for other_symbol in sorted(other_symbols):
            try:
                other_history = ibkr.get_price_history(
                    symbol=other_symbol, period=_PRICE_PERIOD, bar_size=_BAR_SIZE
                )
            except IBKRClientError as exc:
                logger.warning(
                    "PortfolioCorrelationWorkflow: failed to fetch price "
                    "history for %s: %s",
                    other_symbol,
                    exc,
                )
                agent_inferences.append(
                    f"[Agent inference] Price history for {other_symbol} could "
                    "not be retrieved from IBKR — excluded from correlation check."
                )
                continue

            citations.append(
                Citation(
                    source_type=CitationSourceType.IBKR,
                    label=(
                        f"IBKR:{other_symbol} price_history, "
                        f"{other_history.retrieved_at.isoformat()}"
                    ),
                    url=None,
                    retrieval_date=other_history.retrieved_at.date(),
                )
            )
            other_closes = bars_to_closes(other_history.bars)

            corr_result = _compute_correlation(primary_closes, other_closes)
            if corr_result is None:
                agent_inferences.append(
                    f"[Agent inference] Insufficient overlapping price history "
                    f"between {primary_symbol} and {other_symbol} — excluded "
                    "from correlation check."
                )
                continue

            corr, n_overlap = corr_result
            correlations[other_symbol] = {
                "correlation": corr,
                "n_overlapping_days": n_overlap,
            }

        if not correlations:
            agent_inferences.append(
                "[Agent inference] No other position had sufficient "
                "overlapping price history for a correlation to be computed."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": primary_symbol,
                    "correlations": {},
                    "analysis": "",
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output="",
            )

        user_message = _build_user_message(primary_symbol, correlations)
        response = anthropic.complete(
            messages=[{"role": "user", "content": user_message}],
            model=self.model,
            task_type="portfolio_correlation",
            workflow_run_id=context.current_workflow_run_id,
            thesis_id=thesis.id,
            pod_id=getattr(thesis, "pod_id", None),
            max_tokens=512,
            system=_SYSTEM_PROMPT,
        )

        try:
            parsed = json.loads(response.content)
            analysis = parsed.get("analysis", response.content)
            llm_inferences = parsed.get("agent_inferences", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "PortfolioCorrelationWorkflow: LLM response was not valid JSON"
            )
            analysis = response.content
            llm_inferences = []

        agent_inferences.extend(llm_inferences)

        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={
                "instrument": primary_symbol,
                "correlations": correlations,
                "analysis": analysis,
            },
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=response.content,
        )
