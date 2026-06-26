"""Tests for RecommendationWorkflow."""

from __future__ import annotations

import json
import uuid
from datetime import date
from unittest.mock import MagicMock

from app.integrations.anthropic_client import AnthropicResponse
from app.workflows.base import (
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)
from app.workflows.recommendation import RecommendationWorkflow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_REC_RESPONSE = {
    "recommendation": "go",
    "rationale": (
        "MacroContextWorkflow confirms the yield curve is deeply inverted, "
        "supporting the steepener thesis. BacktestWorkflow shows TLT returned "
        "an average of 8% during analog periods. FalsificationGenerationWorkflow "
        "has locked three quantitative kill conditions. [Agent inference] The "
        "limited number of analog periods reduces statistical confidence."
    ),
    "key_assumptions": [
        "Fed pauses rate hikes within 6 months",
        "Inflation continues to decelerate toward 2.5%",
        "10Y-2Y spread eventually normalizes above 0%",
    ],
    "confidence_level": "medium",
    "workflow_citations": [
        "MacroContextWorkflow",
        "BacktestWorkflow",
        "FalsificationGenerationWorkflow",
    ],
}


def _make_anthropic_response(payload: dict | None = None) -> AnthropicResponse:
    return AnthropicResponse(
        content=json.dumps(payload or _DEFAULT_REC_RESPONSE),
        model="claude-opus-4-6",
        input_tokens=3000,
        output_tokens=600,
        stop_reason="end_turn",
    )


def _make_anthropic_client(response: AnthropicResponse | None = None) -> MagicMock:
    mock = MagicMock()
    mock.complete.return_value = response or _make_anthropic_response()
    return mock


def _make_thesis(title: str = "Yield Curve Steepener") -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.pod_id = uuid.uuid4()
    thesis.title = title
    thesis.direction = MagicMock()
    thesis.direction.value = "long"
    thesis.time_horizon = "6 months"
    thesis.notes = "Long TLT as yield curve steepens."
    return thesis


def _make_prior_result(
    workflow_name: str,
    structured_output: dict | None = None,
    status: WorkflowStatus = WorkflowStatus.COMPLETED,
    citations: list[Citation] | None = None,
) -> WorkflowResult:
    return WorkflowResult(
        workflow_name=workflow_name,
        status=status,
        structured_output=structured_output or {"summary": f"{workflow_name} output"},
        citations=citations or [],
        agent_inferences=[],
        raw_output="{}",
    )


def _make_full_context() -> WorkflowContext:
    ctx = WorkflowContext(thesis=_make_thesis(), db=MagicMock())
    ctx.current_workflow_run_id = uuid.uuid4()
    ctx.prior_results = [
        _make_prior_result("MacroContextWorkflow"),
        _make_prior_result("HistoricalAnalogWorkflow"),
        _make_prior_result("InstrumentAnalysisWorkflow"),
        _make_prior_result("WebResearchWorkflow"),
        _make_prior_result("BacktestWorkflow"),
        _make_prior_result("FalsificationGenerationWorkflow"),
    ]
    return ctx


def _make_empty_context() -> WorkflowContext:
    ctx = WorkflowContext(thesis=_make_thesis(), db=MagicMock())
    ctx.current_workflow_run_id = uuid.uuid4()
    return ctx


# ---------------------------------------------------------------------------
# TestRecommendationWorkflow — happy path
# ---------------------------------------------------------------------------


class TestRecommendationWorkflow:
    def test_status_is_completed(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.status == WorkflowStatus.COMPLETED

    def test_workflow_name_is_correct(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.workflow_name == "RecommendationWorkflow"

    def test_uses_opus_model(self):
        anthropic = _make_anthropic_client()
        workflow = RecommendationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_full_context())
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-6"

    def test_recommendation_is_valid_value(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.structured_output["recommendation"] in {"go", "no_go", "hold"}

    def test_rationale_is_non_empty_string(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert isinstance(result.structured_output["rationale"], str)
        assert len(result.structured_output["rationale"]) > 0

    def test_key_assumptions_is_list(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert isinstance(result.structured_output["key_assumptions"], list)
        assert len(result.structured_output["key_assumptions"]) >= 1

    def test_confidence_level_is_valid_value(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.structured_output["confidence_level"] in {
            "high",
            "medium",
            "low",
        }

    def test_workflow_citations_is_list(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert isinstance(result.structured_output["workflow_citations"], list)

    def test_structured_output_has_all_required_keys(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        output = result.structured_output
        assert "recommendation" in output
        assert "rationale" in output
        assert "key_assumptions" in output
        assert "confidence_level" in output
        assert "workflow_citations" in output


# ---------------------------------------------------------------------------
# TestRecommendationWorkflowCitations — citation propagation
# ---------------------------------------------------------------------------


class TestRecommendationWorkflowCitations:
    def test_citations_include_agent_inference_entry(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        source_types = [c.source_type for c in result.citations]
        assert CitationSourceType.AGENT_INFERENCE in source_types

    def test_citations_propagated_from_cited_workflows(self):
        fred_citation = Citation(
            source_type=CitationSourceType.FRED,
            label="FRED:T10Y2Y, retrieved 2025-01-01",
            url=None,
            retrieval_date=date(2025, 1, 1),
        )
        ctx = _make_full_context()
        ctx.prior_results = [
            _make_prior_result("MacroContextWorkflow", citations=[fred_citation]),
            _make_prior_result("BacktestWorkflow"),
            _make_prior_result("FalsificationGenerationWorkflow"),
        ]
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), ctx)
        labels = [c.label for c in result.citations]
        assert "FRED:T10Y2Y, retrieved 2025-01-01" in labels

    def test_citations_not_propagated_from_uncited_workflows(self):
        ibkr_citation = Citation(
            source_type=CitationSourceType.IBKR,
            label="IBKR:TLT price, 2025-01-01",
            url=None,
            retrieval_date=date(2025, 1, 1),
        )
        ctx = _make_empty_context()
        ctx.prior_results = [
            _make_prior_result("HistoricalAnalogWorkflow", citations=[ibkr_citation]),
        ]
        # Response only cites MacroContextWorkflow (not HistoricalAnalogWorkflow)
        payload = {
            **_DEFAULT_REC_RESPONSE,
            "workflow_citations": ["MacroContextWorkflow"],
        }
        workflow = RecommendationWorkflow(
            anthropic_client=_make_anthropic_client(_make_anthropic_response(payload))
        )
        result = workflow.execute(_make_thesis(), ctx)
        labels = [c.label for c in result.citations]
        assert "IBKR:TLT price, 2025-01-01" not in labels


# ---------------------------------------------------------------------------
# TestRecommendationWorkflowLLM — LLM call details
# ---------------------------------------------------------------------------


class TestRecommendationWorkflowLLM:
    def test_llm_called_exactly_once(self):
        anthropic = _make_anthropic_client()
        workflow = RecommendationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_full_context())
        assert anthropic.complete.call_count == 1

    def test_task_type_is_recommendation(self):
        anthropic = _make_anthropic_client()
        workflow = RecommendationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_full_context())
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["task_type"] == "recommendation"

    def test_workflow_run_id_passed_to_llm(self):
        anthropic = _make_anthropic_client()
        workflow = RecommendationWorkflow(anthropic_client=anthropic)
        ctx = _make_full_context()
        run_id = ctx.current_workflow_run_id
        workflow.execute(_make_thesis(), ctx)
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["workflow_run_id"] == run_id

    def test_thesis_title_in_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = RecommendationWorkflow(anthropic_client=anthropic)
        thesis = _make_thesis(title="EM Duration Short")
        workflow.execute(thesis, _make_full_context())
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "EM Duration Short" in content

    def test_prior_workflow_outputs_in_prompt(self):
        anthropic = _make_anthropic_client()
        ctx = _make_full_context()
        workflow = RecommendationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), ctx)
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "MACROCONTEXTWORKFLOW" in content
        assert "BACKTESTWORKFLOW" in content
        assert "FALSIFICATIONGENERATIONWORKFLOW" in content

    def test_missing_workflow_marked_unavailable_in_prompt(self):
        anthropic = _make_anthropic_client()
        ctx = _make_empty_context()  # no prior results
        workflow = RecommendationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), ctx)
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "not available" in content

    def test_system_prompt_contains_recommendation_values(self):
        anthropic = _make_anthropic_client()
        workflow = RecommendationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_full_context())
        call_kwargs = anthropic.complete.call_args[1]
        system = call_kwargs["system"]
        assert '"go"' in system
        assert '"no_go"' in system
        assert '"hold"' in system


# ---------------------------------------------------------------------------
# TestRecommendationWorkflowMissingResults — incomplete context
# ---------------------------------------------------------------------------


class TestRecommendationWorkflowMissingResults:
    def test_flags_missing_workflows_in_agent_inferences(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_empty_context())
        combined = " ".join(result.agent_inferences)
        assert "MacroContextWorkflow" in combined
        assert "BacktestWorkflow" in combined

    def test_still_completes_with_no_prior_results(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_empty_context())
        assert result.status == WorkflowStatus.COMPLETED

    def test_partial_context_still_produces_valid_recommendation(self):
        ctx = _make_empty_context()
        ctx.prior_results = [_make_prior_result("MacroContextWorkflow")]
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), ctx)
        assert result.structured_output["recommendation"] in {"go", "no_go", "hold"}


# ---------------------------------------------------------------------------
# TestRecommendationWorkflowEdgeCases — validation and fallbacks
# ---------------------------------------------------------------------------


class TestRecommendationWorkflowEdgeCases:
    def test_partial_result_on_json_parse_failure(self):
        bad_response = AnthropicResponse(
            content="not valid json",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
        )
        workflow = RecommendationWorkflow(
            anthropic_client=_make_anthropic_client(bad_response)
        )
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.status == WorkflowStatus.PARTIAL
        assert result.structured_output["recommendation"] is None
        assert result.raw_output == "not valid json"

    def test_invalid_recommendation_defaults_to_hold(self):
        payload = {**_DEFAULT_REC_RESPONSE, "recommendation": "maybe"}
        workflow = RecommendationWorkflow(
            anthropic_client=_make_anthropic_client(_make_anthropic_response(payload))
        )
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.structured_output["recommendation"] == "hold"
        combined = " ".join(result.agent_inferences)
        assert "maybe" in combined

    def test_invalid_confidence_defaults_to_low(self):
        payload = {**_DEFAULT_REC_RESPONSE, "confidence_level": "very_high"}
        workflow = RecommendationWorkflow(
            anthropic_client=_make_anthropic_client(_make_anthropic_response(payload))
        )
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.structured_output["confidence_level"] == "low"
        combined = " ".join(result.agent_inferences)
        assert "very_high" in combined

    def test_non_list_key_assumptions_becomes_empty_list(self):
        payload = {**_DEFAULT_REC_RESPONSE, "key_assumptions": "a single string"}
        workflow = RecommendationWorkflow(
            anthropic_client=_make_anthropic_client(_make_anthropic_response(payload))
        )
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.structured_output["key_assumptions"] == []

    def test_non_list_workflow_citations_becomes_empty_list(self):
        payload = {
            **_DEFAULT_REC_RESPONSE,
            "workflow_citations": "MacroContextWorkflow",
        }
        workflow = RecommendationWorkflow(
            anthropic_client=_make_anthropic_client(_make_anthropic_response(payload))
        )
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.structured_output["workflow_citations"] == []

    def test_handles_markdown_fenced_json(self):
        fenced = f"```json\n{json.dumps(_DEFAULT_REC_RESPONSE)}\n```"
        response = AnthropicResponse(
            content=fenced,
            model="claude-opus-4-6",
            input_tokens=3000,
            output_tokens=600,
            stop_reason="end_turn",
        )
        workflow = RecommendationWorkflow(
            anthropic_client=_make_anthropic_client(response)
        )
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert result.status == WorkflowStatus.COMPLETED
        assert result.structured_output["recommendation"] == "go"

    def test_raw_output_preserved(self):
        workflow = RecommendationWorkflow(anthropic_client=_make_anthropic_client())
        result = workflow.execute(_make_thesis(), _make_full_context())
        assert "recommendation" in result.raw_output

    def test_all_three_recommendation_values_accepted(self):
        for rec_value in ("go", "no_go", "hold"):
            payload = {**_DEFAULT_REC_RESPONSE, "recommendation": rec_value}
            workflow = RecommendationWorkflow(
                anthropic_client=_make_anthropic_client(
                    _make_anthropic_response(payload)
                )
            )
            result = workflow.execute(_make_thesis(), _make_full_context())
            assert result.structured_output["recommendation"] == rec_value
