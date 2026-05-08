"""Step 6/7: translate qualitative thesis beliefs into kill conditions."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date

from app.integrations.anthropic_client import AnthropicClient
from app.models.enums import ConditionType
from app.models.thesis import FalsificationCondition, Thesis
from app.workflows.base import (
    BaseWorkflow,
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.index("\n")
        stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


_SCHEDULED_TRIGGER_TYPES = frozenset(
    {
        "CPI_RELEASE",
        "FOMC_DECISION",
        "NFP_RELEASE",
        "PMI_RELEASE",
        "GDP_RELEASE",
        "PCE_RELEASE",
    }
)

_UNSCHEDULED_TRIGGER_TYPES = frozenset(
    {
        "TARIFF_ANNOUNCEMENT",
        "FED_SPEECH",
        "GEOPOLITICAL_EVENT",
        "SURPRISE_RATE_MOVE",
    }
)

_ALL_TRIGGER_TYPES = _SCHEDULED_TRIGGER_TYPES | _UNSCHEDULED_TRIGGER_TYPES

_SYSTEM_PROMPT = """\
You are a senior macro research analyst responsible for translating \
qualitative investment thesis beliefs into discrete, programmatically \
testable kill conditions (falsification conditions).

A falsification condition is a statement about the world that, if violated, \
signals the thesis is no longer valid. Every condition MUST have a measurable \
proxy — no condition may rely on subjective judgment at evaluation time.

Generate 3-5 falsification conditions for the thesis below. For each condition \
provide:

1. "description" — a one-sentence narrative explanation of what this condition tests.
2. "condition_type" — either "state" (evaluated daily against current data) or \
"event" (evaluated only when a specific trigger event occurs).
3. "trigger_type" — ONLY for event conditions. Must be one of: \
CPI_RELEASE, FOMC_DECISION, NFP_RELEASE, PMI_RELEASE, GDP_RELEASE, PCE_RELEASE, \
TARIFF_ANNOUNCEMENT, FED_SPEECH, GEOPOLITICAL_EVENT, SURPRISE_RATE_MOVE. \
Set to null for state conditions.
4. "measurable_proxy" — the specific data source and metric to check \
(e.g., "FRED:T10Y2Y", "FRED:DFF", "TLT 5-day return post-event", "VIX level").
5. "evaluation_logic" — the threshold and comparison operator that constitutes \
falsification (e.g., "> 4.5%%", "< -0.1%%", "spread narrows below 50bps").

Guidelines:
- Qualitative beliefs MUST become quantitative thresholds. \
"The Fed is dovish" → "Fed Funds futures pricing >2 cuts over next 3 meetings".
- Use the current macro data to set thresholds relative to current values — \
a threshold that is already violated today is useless.
- Mix state and event conditions where appropriate.
- Conditions should be independent — each tests a different aspect of the thesis.
- Be specific about data sources. Prefer FRED series IDs where applicable.

Respond with a JSON object containing exactly two keys:
- "conditions": an array of 3-5 condition objects with the fields above.
- "rationale": a brief paragraph explaining the overall falsification strategy.

Respond only with the JSON object. No markdown fences, no preamble.\
"""


def _build_user_message(
    thesis: Thesis,
    macro_context: dict | None,
    backtest_data: dict | None,
    instrument_data: dict | None,
    web_research_data: dict | None,
) -> str:
    thesis_direction = (
        thesis.direction.value
        if hasattr(thesis.direction, "value")
        else str(thesis.direction)
    )

    lines = [
        "=== THESIS ===",
        f"Title: {thesis.title}",
        f"Direction: {thesis_direction}",
        f"Time Horizon: {thesis.time_horizon}",
    ]
    if thesis.notes:
        lines.append(f"Notes: {thesis.notes}")
    lines.append("")

    if macro_context:
        lines.append("=== CURRENT MACRO CONTEXT ===")
        summary = macro_context.get("summary", "")
        if summary:
            lines.append(summary)
        series = macro_context.get("series", {})
        if series:
            lines.append("")
            lines.append("Key data points:")
            for series_id, series_data in series.items():
                if isinstance(series_data, dict):
                    latest = series_data.get("latest_value")
                    if latest is not None:
                        lines.append(f"  {series_id}: {latest}")
        lines.append("")

    if backtest_data:
        lines.append("=== HISTORICAL ANALOG ANALYSIS ===")
        aggregate = backtest_data.get("aggregate", {})
        if aggregate:
            lines.append(
                f"  Avg return across analogs: " f"{aggregate.get('avg_return', 'N/A')}"
            )
            lines.append(
                f"  Avg max drawdown: " f"{aggregate.get('avg_max_drawdown', 'N/A')}"
            )
            lines.append(f"  Win rate: {aggregate.get('win_rate', 'N/A')}")
        analysis = backtest_data.get("analysis", "")
        if analysis:
            lines.append(f"  Analysis: {analysis}")
        lines.append("")

    if instrument_data:
        lines.append("=== INSTRUMENT ANALYSIS ===")
        instruments = instrument_data.get("instruments", {})
        for symbol, stats in instruments.items():
            lines.append(f"  {symbol}:")
            if isinstance(stats, dict):
                for key in (
                    "direction",
                    "role",
                    "annualized_return",
                    "annualized_vol",
                    "max_drawdown",
                ):
                    if key in stats:
                        lines.append(f"    {key}: {stats[key]}")
        analysis = instrument_data.get("analysis", "")
        if analysis:
            lines.append(f"  Analysis: {analysis}")
        lines.append("")

    if web_research_data:
        lines.append("=== RECENT WEB RESEARCH ===")
        sources = web_research_data.get("sources", [])
        if sources:
            for src in sources[:5]:
                if isinstance(src, dict):
                    title = src.get("title", "")
                    annotation = src.get("annotation", "")
                    lines.append(f"  - {title}: {annotation}")
        lines.append("")

    return "\n".join(lines)


class FalsificationGenerationWorkflow(BaseWorkflow):
    """Step 6/7: generate 3-5 falsification conditions and persist them."""

    name = "FalsificationGenerationWorkflow"
    description = (
        "Translates qualitative thesis beliefs into 3-5 discrete, "
        "programmatically testable kill conditions (falsification conditions). "
        "Uses Claude Opus for high-judgment translation of qualitative "
        "assumptions into rigorous falsifiable conditions with measurable proxies."
    )
    required_inputs = ["title", "direction", "time_horizon"]
    model: str = "claude-opus-4-6"

    def __init__(
        self,
        anthropic_client: AnthropicClient | None = None,
    ) -> None:
        self._anthropic = anthropic_client

    def execute(self, thesis: Thesis, context: WorkflowContext) -> WorkflowResult:
        if self._anthropic is None:
            from app.core.settings import get_settings

            settings = get_settings()
            anthropic = AnthropicClient(
                api_key=settings.anthropic_api_key,
                db=context.db,
            )
        else:
            anthropic = self._anthropic

        agent_inferences: list[str] = []
        citations: list[Citation] = []

        macro_result = context.get_result("MacroContextWorkflow")
        backtest_result = context.get_result("BacktestWorkflow")
        instrument_result = context.get_result("InstrumentAnalysisWorkflow")
        web_result = context.get_result("WebResearchWorkflow")

        macro_data: dict | None = None
        backtest_data: dict | None = None
        instrument_data: dict | None = None
        web_data: dict | None = None

        if macro_result is not None:
            macro_data = macro_result.structured_output
        else:
            agent_inferences.append(
                "[Agent inference] MacroContextWorkflow result was not available — "
                "conditions may lack precise threshold grounding relative to "
                "current macro levels."
            )

        if backtest_result is not None:
            backtest_data = backtest_result.structured_output
        else:
            agent_inferences.append(
                "[Agent inference] BacktestWorkflow result was not available — "
                "risk-based conditions (drawdown, volatility) may lack historical "
                "context for threshold calibration."
            )

        if instrument_result is not None:
            instrument_data = instrument_result.structured_output
        else:
            agent_inferences.append(
                "[Agent inference] InstrumentAnalysisWorkflow result was not "
                "available — instrument-specific conditions may be less precise."
            )

        if web_result is not None:
            web_data = web_result.structured_output
        else:
            agent_inferences.append(
                "[Agent inference] WebResearchWorkflow result was not available — "
                "event-type conditions may lack recent context."
            )

        user_message = _build_user_message(
            thesis, macro_data, backtest_data, instrument_data, web_data
        )

        response = anthropic.complete(
            messages=[{"role": "user", "content": user_message}],
            model=self.model,
            task_type="falsification_generation",
            workflow_run_id=context.current_workflow_run_id,
            thesis_id=thesis.id,
            pod_id=getattr(thesis, "pod_id", None),
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
        )

        try:
            content = _strip_markdown_fences(response.content)
            parsed = json.loads(content)
            conditions_raw = parsed.get("conditions", [])
            rationale = parsed.get("rationale", "")
        except (json.JSONDecodeError, AttributeError):
            logger.warning(
                "FalsificationGenerationWorkflow: LLM response was not valid JSON"
            )
            agent_inferences.append(
                "[Agent inference] LLM response could not be parsed as JSON — "
                "returning partial result with raw output preserved."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={"conditions": [], "rationale": ""},
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output=response.content,
            )

        conditions_output: list[dict] = []

        for condition_data in conditions_raw:
            if not isinstance(condition_data, dict):
                continue

            description = condition_data.get("description", "")
            condition_type_str = condition_data.get("condition_type", "")
            trigger_type = condition_data.get("trigger_type")
            measurable_proxy = condition_data.get("measurable_proxy", "")
            evaluation_logic = condition_data.get("evaluation_logic", "")

            if not description or not condition_type_str or not measurable_proxy:
                continue

            try:
                condition_type = ConditionType(condition_type_str)
            except ValueError:
                agent_inferences.append(
                    f"[Agent inference] Skipped condition with invalid "
                    f"condition_type '{condition_type_str}'."
                )
                continue

            if condition_type == ConditionType.event:
                if trigger_type and trigger_type not in _ALL_TRIGGER_TYPES:
                    agent_inferences.append(
                        f"[Agent inference] Event condition has unsupported "
                        f"trigger_type '{trigger_type}' — condition included "
                        f"but trigger_type may need manual correction."
                    )
            else:
                trigger_type = None

            condition = FalsificationCondition(
                id=uuid.uuid4(),
                thesis_id=thesis.id,
                description=description,
                condition_type=condition_type,
                trigger_type=trigger_type,
                measurable_proxy=measurable_proxy,
                evaluation_logic=evaluation_logic,
                chain_operator=None,
                chain_group=None,
            )
            context.db.add(condition)

            conditions_output.append(
                {
                    "id": str(condition.id),
                    "description": description,
                    "condition_type": condition_type.value,
                    "trigger_type": trigger_type,
                    "measurable_proxy": measurable_proxy,
                    "evaluation_logic": evaluation_logic,
                }
            )

        context.db.flush()

        if len(conditions_output) < 3:
            agent_inferences.append(
                f"[Agent inference] Only {len(conditions_output)} condition(s) "
                f"were generated (expected 3-5). The LLM may have had limited "
                f"context or the thesis may be narrowly scoped."
            )

        citations.append(
            Citation(
                source_type=CitationSourceType.AGENT_INFERENCE,
                label="[Agent inference]",
                url=None,
                retrieval_date=date.today(),
            )
        )

        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={
                "conditions": conditions_output,
                "rationale": rationale,
            },
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=response.content,
        )
