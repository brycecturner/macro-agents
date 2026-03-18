"""Tests for BaseWorkflow, WorkflowResult, Citation, and WorkflowContext."""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock

import pytest

from app.workflows.base import (
    BaseWorkflow,
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thesis(title: str = "Test Thesis") -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.title = title
    return thesis


def _make_result(name: str = "SomeWorkflow") -> WorkflowResult:
    return WorkflowResult(
        workflow_name=name,
        status=WorkflowStatus.COMPLETED,
        structured_output={"key": "value"},
        citations=[],
        agent_inferences=[],
        raw_output="raw",
    )


class _ConcreteWorkflow(BaseWorkflow):
    """Minimal concrete subclass used in tests."""

    name = "ConcreteWorkflow"
    description = "A test workflow."
    required_inputs = ["title", "notes"]

    def execute(self, thesis, context: WorkflowContext) -> WorkflowResult:
        return WorkflowResult(
            workflow_name=self.name,
            status=WorkflowStatus.COMPLETED,
            structured_output={"thesis_title": thesis.title},
            citations=[],
            agent_inferences=["[Agent inference] This is a test."],
            raw_output="raw output",
        )


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class TestCitation:
    def test_fred_citation_fields(self):
        c = Citation(
            source_type=CitationSourceType.FRED,
            label="FRED:T10Y2Y, retrieved 2024-01-15",
            url=None,
            retrieval_date=date(2024, 1, 15),
        )
        assert c.source_type == CitationSourceType.FRED
        assert "T10Y2Y" in c.label
        assert c.url is None
        assert c.retrieval_date == date(2024, 1, 15)

    def test_web_citation_fields(self):
        c = Citation(
            source_type=CitationSourceType.WEB,
            label="https://example.com/article",
            url="https://example.com/article",
            retrieval_date=date(2024, 6, 1),
        )
        assert c.source_type == CitationSourceType.WEB
        assert c.url == "https://example.com/article"

    def test_ibkr_citation_fields(self):
        c = Citation(
            source_type=CitationSourceType.IBKR,
            label="IBKR:TLT price_history, 2024-01-15T16:00:00Z",
            url=None,
            retrieval_date=date(2024, 1, 15),
        )
        assert c.source_type == CitationSourceType.IBKR

    def test_agent_inference_citation(self):
        c = Citation(
            source_type=CitationSourceType.AGENT_INFERENCE,
            label="[Agent inference]",
            url=None,
            retrieval_date=date(2024, 1, 15),
        )
        assert c.source_type == CitationSourceType.AGENT_INFERENCE


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------


class TestWorkflowResult:
    def test_completed_status(self):
        result = _make_result()
        assert result.status == WorkflowStatus.COMPLETED

    def test_failed_status(self):
        result = WorkflowResult(
            workflow_name="FailingWorkflow",
            status=WorkflowStatus.FAILED,
            structured_output={},
            citations=[],
            agent_inferences=[],
            raw_output="error traceback here",
        )
        assert result.status == WorkflowStatus.FAILED

    def test_structured_output_accessible(self):
        result = WorkflowResult(
            workflow_name="SomeWorkflow",
            status=WorkflowStatus.COMPLETED,
            structured_output={"cpi": 3.2, "fed_funds": 5.25},
            citations=[],
            agent_inferences=[],
            raw_output="",
        )
        assert result.structured_output["cpi"] == 3.2

    def test_citations_list(self):
        citation = Citation(
            source_type=CitationSourceType.FRED,
            label="FRED:CPIAUCSL, retrieved 2024-01-15",
            url=None,
            retrieval_date=date(2024, 1, 15),
        )
        result = WorkflowResult(
            workflow_name="SomeWorkflow",
            status=WorkflowStatus.COMPLETED,
            structured_output={},
            citations=[citation],
            agent_inferences=[],
            raw_output="",
        )
        assert len(result.citations) == 1
        assert result.citations[0].source_type == CitationSourceType.FRED

    def test_agent_inferences_list(self):
        result = WorkflowResult(
            workflow_name="SomeWorkflow",
            status=WorkflowStatus.COMPLETED,
            structured_output={},
            citations=[],
            agent_inferences=["[Agent inference] Yield curve likely to steepen."],
            raw_output="",
        )
        assert len(result.agent_inferences) == 1
        assert "[Agent inference]" in result.agent_inferences[0]


# ---------------------------------------------------------------------------
# WorkflowContext
# ---------------------------------------------------------------------------


class TestWorkflowContext:
    def test_empty_prior_results_by_default(self):
        context = WorkflowContext(thesis=_make_thesis())
        assert context.prior_results == []

    def test_pod_settings_optional(self):
        context = WorkflowContext(thesis=_make_thesis())
        assert context.pod_settings is None

    def test_holds_thesis(self):
        thesis = _make_thesis("My Trade Idea")
        context = WorkflowContext(thesis=thesis)
        assert context.thesis.title == "My Trade Idea"

    def test_holds_prior_results(self):
        result = _make_result("MacroContextWorkflow")
        context = WorkflowContext(
            thesis=_make_thesis(),
            prior_results=[result],
        )
        assert len(context.prior_results) == 1

    def test_get_result_returns_matching_workflow(self):
        r1 = _make_result("MacroContextWorkflow")
        r2 = _make_result("HistoricalAnalogWorkflow")
        context = WorkflowContext(
            thesis=_make_thesis(),
            prior_results=[r1, r2],
        )
        assert context.get_result("MacroContextWorkflow") is r1
        assert context.get_result("HistoricalAnalogWorkflow") is r2

    def test_get_result_returns_none_when_not_found(self):
        context = WorkflowContext(thesis=_make_thesis(), prior_results=[])
        assert context.get_result("NonexistentWorkflow") is None

    def test_get_result_returns_most_recent_on_duplicate_names(self):
        first = _make_result("MacroContextWorkflow")
        first.structured_output = {"run": 1}
        second = _make_result("MacroContextWorkflow")
        second.structured_output = {"run": 2}
        context = WorkflowContext(
            thesis=_make_thesis(),
            prior_results=[first, second],
        )
        assert context.get_result("MacroContextWorkflow").structured_output["run"] == 2


# ---------------------------------------------------------------------------
# BaseWorkflow interface contract
# ---------------------------------------------------------------------------


class TestBaseWorkflow:
    def test_cannot_instantiate_base_directly(self):
        with pytest.raises(TypeError):
            BaseWorkflow()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        workflow = _ConcreteWorkflow()
        assert workflow.name == "ConcreteWorkflow"

    def test_concrete_subclass_has_required_class_attributes(self):
        workflow = _ConcreteWorkflow()
        assert hasattr(workflow, "name")
        assert hasattr(workflow, "description")
        assert hasattr(workflow, "required_inputs")

    def test_execute_returns_workflow_result(self):
        workflow = _ConcreteWorkflow()
        thesis = _make_thesis("Yield Curve Steepener")
        context = WorkflowContext(thesis=thesis)
        result = workflow.execute(thesis, context)
        assert isinstance(result, WorkflowResult)

    def test_execute_result_has_correct_workflow_name(self):
        workflow = _ConcreteWorkflow()
        thesis = _make_thesis()
        context = WorkflowContext(thesis=thesis)
        result = workflow.execute(thesis, context)
        assert result.workflow_name == "ConcreteWorkflow"

    def test_execute_result_status_is_completed(self):
        workflow = _ConcreteWorkflow()
        thesis = _make_thesis()
        context = WorkflowContext(thesis=thesis)
        result = workflow.execute(thesis, context)
        assert result.status == WorkflowStatus.COMPLETED

    def test_execute_uses_thesis_data(self):
        workflow = _ConcreteWorkflow()
        thesis = _make_thesis("Inflation Hedge")
        context = WorkflowContext(thesis=thesis)
        result = workflow.execute(thesis, context)
        assert result.structured_output["thesis_title"] == "Inflation Hedge"

    def test_subclass_missing_execute_raises_on_instantiation(self):
        class IncompleteWorkflow(BaseWorkflow):
            name = "Incomplete"
            description = "Missing execute."
            required_inputs = []

        with pytest.raises(TypeError):
            IncompleteWorkflow()  # type: ignore[abstract]
