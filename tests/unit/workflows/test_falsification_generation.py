"""Tests for FalsificationGenerationWorkflow."""

from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

from app.integrations.anthropic_client import AnthropicResponse
from app.models.thesis import FalsificationCondition
from app.workflows.base import (
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)
from app.workflows.falsification_generation import FalsificationGenerationWorkflow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conditions_response(
    conditions: list[dict] | None = None,
    rationale: str = "These conditions test the core assumptions of the thesis.",
) -> list[dict]:
    """Return a default set of conditions for a mock LLM response."""
    if conditions is not None:
        return conditions
    return [
        {
            "description": "10Y-2Y spread must remain negative (inverted curve)",
            "condition_type": "state",
            "trigger_type": None,
            "measurable_proxy": "FRED:T10Y2Y",
            "evaluation_logic": "> -0.1%",
        },
        {
            "description": "Fed Funds rate must stay above 4.5%",
            "condition_type": "state",
            "trigger_type": None,
            "measurable_proxy": "FRED:DFF",
            "evaluation_logic": "< 4.5%",
        },
        {
            "description": "TLT must not decline more than 3% after FOMC decision",
            "condition_type": "event",
            "trigger_type": "FOMC_DECISION",
            "measurable_proxy": "TLT 5-day return post-event",
            "evaluation_logic": "< -3%",
        },
        {
            "description": "VIX must remain below 35 (no panic regime)",
            "condition_type": "state",
            "trigger_type": None,
            "measurable_proxy": "VIX level",
            "evaluation_logic": "> 35",
        },
    ]


def _make_anthropic_response(
    conditions: list[dict] | None = None,
    rationale: str = "These conditions test the core assumptions of the thesis.",
) -> AnthropicResponse:
    content = json.dumps(
        {
            "conditions": _make_conditions_response(conditions),
            "rationale": rationale,
        }
    )
    return AnthropicResponse(
        content=content,
        model="claude-opus-4-6",
        input_tokens=1200,
        output_tokens=800,
        stop_reason="end_turn",
    )


def _make_anthropic_client(
    response: AnthropicResponse | None = None,
) -> MagicMock:
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
    thesis.instruments = []
    return thesis


def _make_macro_result() -> WorkflowResult:
    return WorkflowResult(
        workflow_name="MacroContextWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={
            "summary": "Fed is holding rates steady. Inflation cooling.",
            "series": {
                "T10Y2Y": {"latest_value": -0.05},
                "DFF": {"latest_value": 5.25},
                "CPIAUCSL": {"latest_value": 3.2},
            },
        },
        citations=[],
        agent_inferences=[],
        raw_output="{}",
    )


def _make_backtest_result() -> WorkflowResult:
    return WorkflowResult(
        workflow_name="BacktestWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={
            "label": "Historical Analog Analysis",
            "instrument": "TLT",
            "direction": "long",
            "analog_periods": [],
            "aggregate": {
                "avg_return": 0.08,
                "avg_max_drawdown": -0.12,
                "win_rate": 0.67,
                "n_periods": 3,
            },
            "benchmark_comparison": {},
            "analysis": "TLT performed well in similar rate environments.",
        },
        citations=[],
        agent_inferences=[],
        raw_output="{}",
    )


def _make_instrument_result() -> WorkflowResult:
    return WorkflowResult(
        workflow_name="InstrumentAnalysisWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={
            "instruments": {
                "TLT": {
                    "direction": "long",
                    "role": "primary",
                    "annualized_return": 0.05,
                    "annualized_vol": 0.15,
                    "max_drawdown": -0.20,
                }
            },
            "correlations": {},
            "analysis": "TLT is well-suited for rate-sensitive trades.",
        },
        citations=[],
        agent_inferences=[],
        raw_output="{}",
    )


def _make_web_result() -> WorkflowResult:
    return WorkflowResult(
        workflow_name="WebResearchWorkflow",
        status=WorkflowStatus.COMPLETED,
        structured_output={
            "search_queries": ["fed rate decision 2025"],
            "sources": [
                {
                    "title": "Fed holds rates at 5.25%",
                    "url": "https://example.com/fed",
                    "source_type": "web",
                    "annotation": "Fed signaled patience on rate cuts.",
                    "is_cited": True,
                    "rank": 1,
                },
            ],
        },
        citations=[],
        agent_inferences=[],
        raw_output="{}",
    )


def _make_context(
    with_macro: bool = True,
    with_backtest: bool = True,
    with_instrument: bool = True,
    with_web: bool = True,
) -> WorkflowContext:
    ctx = WorkflowContext(thesis=_make_thesis(), db=MagicMock())
    ctx.current_workflow_run_id = uuid.uuid4()
    if with_macro:
        ctx.prior_results.append(_make_macro_result())
    if with_backtest:
        ctx.prior_results.append(_make_backtest_result())
    if with_instrument:
        ctx.prior_results.append(_make_instrument_result())
    if with_web:
        ctx.prior_results.append(_make_web_result())
    return ctx


# ---------------------------------------------------------------------------
# TestFalsificationGenerationWorkflow — happy path
# ---------------------------------------------------------------------------


class TestFalsificationGenerationWorkflow:
    def test_generates_three_to_five_conditions(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        conditions = result.structured_output["conditions"]
        assert 3 <= len(conditions) <= 5

    def test_conditions_persisted_to_database(self):
        ctx = _make_context()
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        workflow.execute(_make_thesis(), ctx)
        # Verify db.add was called for each condition
        add_calls = ctx.db.add.call_args_list
        fc_calls = [c for c in add_calls if isinstance(c[0][0], FalsificationCondition)]
        assert len(fc_calls) == 4  # default response has 4 conditions

    def test_condition_type_assignment(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        conditions = result.structured_output["conditions"]
        state_conditions = [c for c in conditions if c["condition_type"] == "state"]
        event_conditions = [c for c in conditions if c["condition_type"] == "event"]
        assert len(state_conditions) == 3
        assert len(event_conditions) == 1

    def test_event_conditions_have_trigger_type(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        conditions = result.structured_output["conditions"]
        event_conditions = [c for c in conditions if c["condition_type"] == "event"]
        for ec in event_conditions:
            assert ec["trigger_type"] is not None
            assert ec["trigger_type"] in {
                "CPI_RELEASE",
                "FOMC_DECISION",
                "NFP_RELEASE",
                "PMI_RELEASE",
                "GDP_RELEASE",
                "PCE_RELEASE",
                "TARIFF_ANNOUNCEMENT",
                "FED_SPEECH",
                "GEOPOLITICAL_EVENT",
                "SURPRISE_RATE_MOVE",
            }

    def test_state_conditions_have_null_trigger_type(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        conditions = result.structured_output["conditions"]
        state_conditions = [c for c in conditions if c["condition_type"] == "state"]
        for sc in state_conditions:
            assert sc["trigger_type"] is None

    def test_measurable_proxy_translation(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        conditions = result.structured_output["conditions"]
        # Every condition must have a non-empty measurable_proxy
        for c in conditions:
            assert c["measurable_proxy"]
            assert len(c["measurable_proxy"]) > 0

    def test_uses_opus_model(self):
        anthropic = _make_anthropic_client()
        workflow = FalsificationGenerationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_context())
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-6"

    def test_handles_missing_prior_results(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        ctx = _make_context(
            with_macro=False,
            with_backtest=False,
            with_instrument=False,
            with_web=False,
        )
        result = workflow.execute(_make_thesis(), ctx)
        # Should still complete — prior results are optional
        assert result.status == WorkflowStatus.COMPLETED
        # Should flag missing results in agent_inferences
        combined = " ".join(result.agent_inferences)
        assert "MacroContextWorkflow" in combined
        assert "BacktestWorkflow" in combined
        assert "InstrumentAnalysisWorkflow" in combined
        assert "WebResearchWorkflow" in combined

    def test_chain_fields_null(self):
        ctx = _make_context()
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        workflow.execute(_make_thesis(), ctx)
        add_calls = ctx.db.add.call_args_list
        fc_calls = [c for c in add_calls if isinstance(c[0][0], FalsificationCondition)]
        for fc_call in fc_calls:
            condition = fc_call[0][0]
            assert condition.chain_operator is None
            assert condition.chain_group is None

    def test_partial_result_on_llm_parse_failure(self):
        bad_response = AnthropicResponse(
            content="not valid json at all",
            model="claude-opus-4-6",
            input_tokens=100,
            output_tokens=50,
            stop_reason="end_turn",
        )
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client(bad_response)
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.PARTIAL
        assert result.structured_output["conditions"] == []
        assert result.raw_output == "not valid json at all"


# ---------------------------------------------------------------------------
# TestFalsificationGenerationWorkflow — structured output
# ---------------------------------------------------------------------------


class TestFalsificationGenerationWorkflowOutput:
    def test_status_is_completed(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED

    def test_output_has_conditions_and_rationale(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert "conditions" in result.structured_output
        assert "rationale" in result.structured_output

    def test_workflow_name_is_correct(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.workflow_name == "FalsificationGenerationWorkflow"

    def test_raw_output_contains_llm_response(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert "conditions" in result.raw_output

    def test_rationale_is_populated(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert len(result.structured_output["rationale"]) > 0

    def test_each_condition_has_required_fields(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        for condition in result.structured_output["conditions"]:
            assert "id" in condition
            assert "description" in condition
            assert "condition_type" in condition
            assert "trigger_type" in condition
            assert "measurable_proxy" in condition
            assert "evaluation_logic" in condition

    def test_condition_ids_are_valid_uuids(self):
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        result = workflow.execute(_make_thesis(), _make_context())
        for condition in result.structured_output["conditions"]:
            # Should not raise
            uuid.UUID(condition["id"])


# ---------------------------------------------------------------------------
# TestFalsificationGenerationWorkflow — LLM call
# ---------------------------------------------------------------------------


class TestFalsificationGenerationWorkflowLLM:
    def test_llm_called_with_correct_task_type(self):
        anthropic = _make_anthropic_client()
        workflow = FalsificationGenerationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_context())
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["task_type"] == "falsification_generation"

    def test_workflow_run_id_passed_to_llm(self):
        anthropic = _make_anthropic_client()
        workflow = FalsificationGenerationWorkflow(anthropic_client=anthropic)
        ctx = _make_context()
        run_id = ctx.current_workflow_run_id
        workflow.execute(_make_thesis(), ctx)
        call_kwargs = anthropic.complete.call_args[1]
        assert call_kwargs["workflow_run_id"] == run_id

    def test_thesis_title_in_prompt(self):
        anthropic = _make_anthropic_client()
        workflow = FalsificationGenerationWorkflow(anthropic_client=anthropic)
        thesis = _make_thesis(title="Duration Steepener Play")
        workflow.execute(thesis, _make_context())
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "Duration Steepener Play" in content

    def test_macro_data_in_prompt_when_available(self):
        anthropic = _make_anthropic_client()
        workflow = FalsificationGenerationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_context())
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "MACRO CONTEXT" in content
        assert "T10Y2Y" in content

    def test_backtest_data_in_prompt_when_available(self):
        anthropic = _make_anthropic_client()
        workflow = FalsificationGenerationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_context())
        content = anthropic.complete.call_args[1]["messages"][0]["content"]
        assert "HISTORICAL ANALOG" in content

    def test_llm_called_exactly_once(self):
        anthropic = _make_anthropic_client()
        workflow = FalsificationGenerationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_context())
        assert anthropic.complete.call_count == 1

    def test_system_prompt_contains_trigger_types(self):
        anthropic = _make_anthropic_client()
        workflow = FalsificationGenerationWorkflow(anthropic_client=anthropic)
        workflow.execute(_make_thesis(), _make_context())
        call_kwargs = anthropic.complete.call_args[1]
        system = call_kwargs["system"]
        assert "FOMC_DECISION" in system
        assert "CPI_RELEASE" in system
        assert "TARIFF_ANNOUNCEMENT" in system


# ---------------------------------------------------------------------------
# TestFalsificationGenerationWorkflow — edge cases
# ---------------------------------------------------------------------------


class TestFalsificationGenerationWorkflowEdgeCases:
    def test_skips_conditions_missing_description(self):
        conditions = [
            {
                "description": "",
                "condition_type": "state",
                "trigger_type": None,
                "measurable_proxy": "FRED:T10Y2Y",
                "evaluation_logic": "> 0",
            },
            {
                "description": "Valid condition",
                "condition_type": "state",
                "trigger_type": None,
                "measurable_proxy": "FRED:DFF",
                "evaluation_logic": "< 4.5%",
            },
        ]
        response = _make_anthropic_response(conditions=conditions)
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client(response)
        )
        result = workflow.execute(_make_thesis(), _make_context())
        # Only the valid condition should appear
        assert len(result.structured_output["conditions"]) == 1

    def test_skips_conditions_with_invalid_condition_type(self):
        conditions = [
            {
                "description": "Bad type",
                "condition_type": "unknown_type",
                "trigger_type": None,
                "measurable_proxy": "FRED:DFF",
                "evaluation_logic": "< 4.5%",
            },
            {
                "description": "Valid condition",
                "condition_type": "state",
                "trigger_type": None,
                "measurable_proxy": "FRED:DFF",
                "evaluation_logic": "< 4.5%",
            },
        ]
        response = _make_anthropic_response(conditions=conditions)
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client(response)
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert len(result.structured_output["conditions"]) == 1

    def test_flags_fewer_than_three_conditions(self):
        conditions = [
            {
                "description": "Only one condition",
                "condition_type": "state",
                "trigger_type": None,
                "measurable_proxy": "FRED:DFF",
                "evaluation_logic": "< 4.5%",
            },
        ]
        response = _make_anthropic_response(conditions=conditions)
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client(response)
        )
        result = workflow.execute(_make_thesis(), _make_context())
        # Still COMPLETED, but flagged
        assert result.status == WorkflowStatus.COMPLETED
        combined = " ".join(result.agent_inferences)
        assert "1 condition" in combined

    def test_state_condition_trigger_type_forced_null(self):
        """Even if LLM returns a trigger_type for a state condition, it's nulled."""
        conditions = [
            {
                "description": "State with bad trigger",
                "condition_type": "state",
                "trigger_type": "FOMC_DECISION",
                "measurable_proxy": "FRED:DFF",
                "evaluation_logic": "< 4.5%",
            },
        ]
        response = _make_anthropic_response(conditions=conditions)
        ctx = _make_context()
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client(response)
        )
        workflow.execute(_make_thesis(), ctx)
        # Check persisted condition has null trigger_type
        add_calls = ctx.db.add.call_args_list
        fc_calls = [c for c in add_calls if isinstance(c[0][0], FalsificationCondition)]
        assert len(fc_calls) == 1
        condition = fc_calls[0][0][0]
        assert condition.trigger_type is None

    def test_event_condition_with_unsupported_trigger_type_flags_inference(self):
        conditions = [
            {
                "description": "Event with bad trigger",
                "condition_type": "event",
                "trigger_type": "ALIEN_INVASION",
                "measurable_proxy": "VIX",
                "evaluation_logic": "> 100",
            },
        ]
        response = _make_anthropic_response(conditions=conditions)
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client(response)
        )
        result = workflow.execute(_make_thesis(), _make_context())
        combined = " ".join(result.agent_inferences)
        assert "ALIEN_INVASION" in combined

    def test_db_flush_called_after_conditions_added(self):
        ctx = _make_context()
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        workflow.execute(_make_thesis(), ctx)
        ctx.db.flush.assert_called_once()

    def test_thesis_id_set_on_persisted_conditions(self):
        thesis = _make_thesis()
        ctx = _make_context()
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client()
        )
        workflow.execute(thesis, ctx)
        add_calls = ctx.db.add.call_args_list
        fc_calls = [c for c in add_calls if isinstance(c[0][0], FalsificationCondition)]
        for fc_call in fc_calls:
            condition = fc_call[0][0]
            assert condition.thesis_id == thesis.id

    def test_handles_markdown_fenced_json_response(self):
        """LLM sometimes wraps JSON in ```json ... ``` fences."""
        conditions = _make_conditions_response()
        raw_json = json.dumps(
            {
                "conditions": conditions,
                "rationale": "Strategy covers all assumptions.",
            }
        )
        fenced_content = f"```json\n{raw_json}\n```"
        response = AnthropicResponse(
            content=fenced_content,
            model="claude-opus-4-6",
            input_tokens=1000,
            output_tokens=800,
            stop_reason="end_turn",
        )
        workflow = FalsificationGenerationWorkflow(
            anthropic_client=_make_anthropic_client(response)
        )
        result = workflow.execute(_make_thesis(), _make_context())
        assert result.status == WorkflowStatus.COMPLETED
        assert len(result.structured_output["conditions"]) == 4
