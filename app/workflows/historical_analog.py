"""HistoricalAnalogWorkflow — identifies historical macro analog periods.

Consumes MacroContextWorkflow output (30 years of FRED series history) and
asks the LLM to identify 2-4 periods where the macro configuration was
genuinely similar to the current backdrop.

All similarity judgments and outcome summaries are explicitly flagged as
[Agent inference] — they are never presented as data-backed conclusions.

This workflow does not call FRED directly. All series data comes from the
MacroContextWorkflow result in context. If that result is absent, the
workflow proceeds with an empty data set and flags it in agent_inferences.
"""

from __future__ import annotations

import json
import logging
from datetime import date

import pandas as pd

from app.integrations.anthropic_client import AnthropicClient
from app.workflows.base import (
    BaseWorkflow,
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a macro research analyst identifying historical periods that are analogous \
to current market conditions for the purpose of evaluating a trade thesis.

You will be given:
1. A trade thesis with direction and time horizon
2. A current macro summary from MacroContextWorkflow
3. ~30 years of monthly historical data for core FRED macro series

Your task: identify 2-4 historical periods (each 3-18 months long) where the \
macro configuration was genuinely similar to the current backdrop. Focus on the \
combination of conditions across all series — not any single indicator in isolation.

Respond with a JSON object containing exactly two keys:
- "analogs": A list of 2-4 objects, each with:
  - "start_date": First month of the period, format "YYYY-MM"
  - "end_date": Last month of the period, format "YYYY-MM"
  - "label": A concise descriptive label \
    (e.g. "2006 Fed pause", "2018 late-cycle tightening")
  - "macro_conditions": Object with approximate series values at the start of the \
    period, using the same column names as the data provided
  - "similarity_rationale": Why this period is analogous to today. \
    MUST start with "[Agent inference]".
  - "outcome_summary": What happened to macro conditions and relevant markets during \
    this period. MUST start with "[Agent inference]".
- "agent_inferences": Flat list of any additional reasoning not captured per-analog. \
  Each string MUST start with "[Agent inference]".

Be honest: if fewer than 2 strong analogs exist in the data, say so in \
agent_inferences and return what you can. Do not fabricate periods. \
Respond only with the JSON object. No markdown fences, no preamble.\
"""


def _build_history_table(series_data: dict[str, dict]) -> str:
    """Build a compact date-aligned CSV from MacroContextWorkflow series data.

    Aligns all series by calendar month. Converts raw CPIAUCSL level to
    YoY % change so the LLM can reason about inflation rates rather than
    index levels. OECD series are included with shortened column names.

    Returns a CSV string suitable for inclusion in an LLM prompt.
    """
    frames: dict[str, pd.Series] = {}

    for series_id, series_dict in series_data.items():
        hist = series_dict.get("historical_data", [])
        if not hist:
            continue
        dates = pd.to_datetime([d["date"] for d in hist])
        values = [d["value"] for d in hist]
        # Clean column name: strip OECD: prefix, collapse spaces
        col = series_id.replace("OECD:", "").strip().replace(" ", "_")
        frames[col] = pd.Series(values, index=dates)

    if not frames:
        return "(no historical data available)"

    df = pd.DataFrame(frames).sort_index()
    df.index = df.index.to_period("M")

    # Replace raw CPIAUCSL level with YoY % change — more interpretable
    if "CPIAUCSL" in df.columns:
        df["CPI_YoY%"] = (df["CPIAUCSL"].pct_change(12) * 100).round(4)
        df = df.drop(columns=["CPIAUCSL"])

    df = df.dropna(how="all").sort_index()
    df.index = [str(p) for p in df.index]

    return df.round(4).to_csv()


def _build_user_message(thesis, macro_summary: str, history_csv: str) -> str:
    direction = (
        thesis.direction.value
        if hasattr(thesis.direction, "value")
        else str(thesis.direction)
    )
    return (
        f"Thesis: {thesis.title}\n"
        f"Direction: {direction}\n"
        f"Time Horizon: {thesis.time_horizon}\n\n"
        f"Current macro summary:\n{macro_summary}\n\n"
        f"Historical macro data (monthly):\n{history_csv}"
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class HistoricalAnalogWorkflow(BaseWorkflow):
    """Identify 2-4 historical macro periods analogous to current conditions.

    Requires MacroContextWorkflow to have run first — its historical series
    data is consumed from context. No additional FRED calls are made.

    Outputs (structured_output):
        analogs (list[dict]): 2-4 analog periods, each containing:
            start_date, end_date, label, macro_conditions,
            similarity_rationale, outcome_summary.
            Similarity and outcome fields are always [Agent inference].
    """

    name = "HistoricalAnalogWorkflow"
    description = (
        "Identifies 2-4 historical periods with macro configurations similar to "
        "current conditions, using 30 years of FRED data from MacroContextWorkflow."
    )
    required_inputs = ["title", "direction", "time_horizon"]
    model: str = "claude-sonnet-4-6"

    def __init__(
        self,
        anthropic_client: AnthropicClient | None = None,
    ) -> None:
        self._anthropic = anthropic_client

    def execute(self, thesis, context: WorkflowContext) -> WorkflowResult:
        if self._anthropic is None:
            from app.core.settings import get_settings

            settings = get_settings()
            anthropic = AnthropicClient(
                api_key=settings.anthropic_api_key,
                db=context.db,
            )
        else:
            anthropic = self._anthropic

        # Pull MacroContextWorkflow result from context
        macro_result = context.get_result("MacroContextWorkflow")
        if macro_result is None:
            logger.warning(
                "HistoricalAnalogWorkflow: MacroContextWorkflow result not in context"
            )
            series_data: dict = {}
            macro_summary = "(MacroContextWorkflow result unavailable)"
        else:
            series_data = macro_result.structured_output.get("series", {})
            macro_summary = macro_result.structured_output.get("summary", "")

        history_csv = _build_history_table(series_data)

        response = anthropic.complete(
            messages=[
                {
                    "role": "user",
                    "content": _build_user_message(thesis, macro_summary, history_csv),
                }
            ],
            model=self.model,
            task_type="historical_analog",
            workflow_run_id=context.current_workflow_run_id,
            thesis_id=thesis.id,
            pod_id=getattr(thesis, "pod_id", None),
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
        )

        try:
            parsed = json.loads(response.content)
            analogs = parsed.get("analogs", [])
            agent_inferences = parsed.get("agent_inferences", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("HistoricalAnalogWorkflow: LLM response was not valid JSON")
            analogs = []
            agent_inferences = []

        # Cite the FRED series consumed from MacroContextWorkflow.
        # Retrieval dates come from the prior result's citations where available.
        citations: list[Citation] = []
        today = date.today()

        if macro_result is not None:
            prior_fred = {
                c.label.split(",")[0].replace("FRED:", "").strip(): c.retrieval_date
                for c in macro_result.citations
                if c.source_type == CitationSourceType.FRED
            }
        else:
            prior_fred = {}

        for series_id in series_data:
            if series_id.startswith("OECD:"):
                continue
            retrieval_date = prior_fred.get(series_id, today)
            citations.append(
                Citation(
                    source_type=CitationSourceType.FRED,
                    label=f"FRED:{series_id}, retrieved {retrieval_date}",
                    url=None,
                    retrieval_date=retrieval_date,
                )
            )

        # Flag missing context prominently
        if macro_result is None:
            agent_inferences = [
                "[Agent inference] MacroContextWorkflow result was not available — "
                "analog identification ran without historical series data and results "
                "should not be relied upon."
            ] + agent_inferences

        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={"analogs": analogs},
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=response.content,
        )
