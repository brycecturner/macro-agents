"""Step 7/7: synthesize all prior workflow results into a Go/No-Go recommendation."""

from __future__ import annotations

import json
import logging
from datetime import date

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

_WORKFLOW_SEQUENCE = [
    "MacroContextWorkflow",
    "HistoricalAnalogWorkflow",
    "InstrumentAnalysisWorkflow",
    "WebResearchWorkflow",
    "BacktestWorkflow",
    "FalsificationGenerationWorkflow",
]

_SYSTEM_PROMPT = """\
You are a senior macro portfolio manager reviewing a complete research package \
for a proposed trade thesis. You have been provided with the structured outputs \
from six prior research workflows. Your task is to synthesize this body of work \
into a final Go / No-Go recommendation.

Your recommendation must be grounded in the research — not intuition. Every \
claim in your rationale must trace back to a specific workflow output. Agent \
inferences must be explicitly labeled as "[Agent inference]" and distinguished \
from data-backed findings.

Produce a JSON object with exactly these keys:

1. "recommendation" — one of: "go", "no_go", or "hold"
   - "go": evidence is sufficiently supportive; proceed to position
   - "no_go": evidence contradicts or fails to support the thesis; do not proceed
   - "hold": evidence is mixed or incomplete; defer pending additional information

2. "rationale" — a single paragraph (3-6 sentences) explaining the recommendation. \
Each sentence should reference the workflow that generated the supporting or \
contradicting evidence. Agent inferences must be labeled "[Agent inference]". \
Do not repeat yourself.

3. "key_assumptions" — a list of 3-5 strings, each a discrete assumption that \
must hold for the thesis to remain valid. These should map to the falsification \
conditions where available.

4. "confidence_level" — one of: "high", "medium", or "low"
   - "high": multiple workflow outputs are consistent and supportive
   - "medium": evidence is directionally supportive but incomplete or mixed
   - "low": significant gaps, contradictions, or missing workflow results

5. "workflow_citations" — a list of workflow names whose outputs directly \
informed this recommendation (e.g. ["MacroContextWorkflow", "BacktestWorkflow"]).

Respond only with the JSON object. No markdown fences, no preamble.\
"""


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.index("\n")
        stripped = stripped[first_newline + 1 :]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    return stripped


def _build_user_message(thesis, prior_results: list[WorkflowResult]) -> str:
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

    results_by_name = {r.workflow_name: r for r in prior_results}

    for workflow_name in _WORKFLOW_SEQUENCE:
        result = results_by_name.get(workflow_name)
        if result is None:
            lines.append(f"=== {workflow_name.upper()} ===")
            lines.append("(Result not available — workflow did not complete.)")
            lines.append("")
            continue

        status_label = (
            "COMPLETED"
            if result.status == WorkflowStatus.COMPLETED
            else result.status.upper()
        )
        lines.append(f"=== {workflow_name.upper()} [{status_label}] ===")

        output = result.structured_output
        if output:
            lines.append(json.dumps(output, indent=2, default=str))
        else:
            lines.append("(No structured output.)")

        if result.agent_inferences:
            lines.append("Agent inferences:")
            for inference in result.agent_inferences:
                lines.append(f"  {inference}")

        lines.append("")

    return "\n".join(lines)


_VALID_RECOMMENDATIONS = frozenset({"go", "no_go", "hold"})
_VALID_CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})


class RecommendationWorkflow(BaseWorkflow):
    """Step 7/7: synthesize all prior results into a Go/No-Go recommendation."""

    name = "RecommendationWorkflow"
    description = (
        "Synthesizes all prior WorkflowResult objects into a Go/No-Go "
        "recommendation with rationale, key assumptions, confidence level, "
        "and explicit citations of which workflow outputs informed the decision. "
        "Uses Claude Opus for high-judgment synthesis of ambiguous research."
    )
    required_inputs = ["title", "direction", "time_horizon"]
    model: str = "claude-opus-4-6"

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

        agent_inferences: list[str] = []
        citations: list[Citation] = []

        missing = [
            name for name in _WORKFLOW_SEQUENCE if context.get_result(name) is None
        ]
        if missing:
            agent_inferences.append(
                f"[Agent inference] The following upstream workflows did not "
                f"produce results and could not inform this recommendation: "
                f"{', '.join(missing)}. The recommendation may be less reliable "
                f"than it would be with a complete research package."
            )

        user_message = _build_user_message(thesis, context.prior_results)

        response = anthropic.complete(
            messages=[{"role": "user", "content": user_message}],
            model=self.model,
            task_type="recommendation",
            workflow_run_id=context.current_workflow_run_id,
            thesis_id=thesis.id,
            pod_id=getattr(thesis, "pod_id", None),
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
        )

        try:
            content = _strip_markdown_fences(response.content)
            parsed = json.loads(content)
        except (json.JSONDecodeError, AttributeError):
            logger.warning("RecommendationWorkflow: LLM response was not valid JSON")
            agent_inferences.append(
                "[Agent inference] LLM response could not be parsed as JSON — "
                "returning partial result with raw output preserved."
            )
            return WorkflowResult(
                workflow_name=self.name,
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "recommendation": None,
                    "rationale": "",
                    "key_assumptions": [],
                    "confidence_level": None,
                    "workflow_citations": [],
                },
                citations=citations,
                agent_inferences=agent_inferences,
                raw_output=response.content,
            )

        recommendation = parsed.get("recommendation")
        rationale = parsed.get("rationale", "")
        key_assumptions = parsed.get("key_assumptions", [])
        confidence_level = parsed.get("confidence_level")
        workflow_citations = parsed.get("workflow_citations", [])

        if recommendation not in _VALID_RECOMMENDATIONS:
            agent_inferences.append(
                f"[Agent inference] LLM returned an unrecognised recommendation "
                f"value '{recommendation}'. Defaulting to 'hold'."
            )
            recommendation = "hold"

        if confidence_level not in _VALID_CONFIDENCE_LEVELS:
            agent_inferences.append(
                f"[Agent inference] LLM returned an unrecognised confidence_level "
                f"value '{confidence_level}'. Defaulting to 'low'."
            )
            confidence_level = "low"

        if not isinstance(key_assumptions, list):
            key_assumptions = []

        if not isinstance(workflow_citations, list):
            workflow_citations = []

        citations.append(
            Citation(
                source_type=CitationSourceType.AGENT_INFERENCE,
                label="[Agent inference]",
                url=None,
                retrieval_date=date.today(),
            )
        )

        for cited_workflow in workflow_citations:
            prior = context.get_result(cited_workflow)
            if prior is not None:
                citations.extend(prior.citations)

        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={
                "recommendation": recommendation,
                "rationale": rationale,
                "key_assumptions": key_assumptions,
                "confidence_level": confidence_level,
                "workflow_citations": workflow_citations,
            },
            citations=citations,
            agent_inferences=agent_inferences,
            raw_output=response.content,
        )
