"""Tests for WorkflowRunner and register_workflows."""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock, patch

from app.models.enums import WorkflowStatus as DBWorkflowStatus
from app.models.workflow import WorkflowRegistry, WorkflowRun
from app.workflows.base import (
    Citation,
    CitationSourceType,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)
from app.workflows.runner import (
    WorkflowRunner,
    _serialize_citations,
    register_workflows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thesis(title: str = "Test Thesis") -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.title = title
    return thesis


def _make_db() -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    return db


def _make_result(
    name: str = "SomeWorkflow",
    status: WorkflowStatus = WorkflowStatus.COMPLETED,
) -> WorkflowResult:
    return WorkflowResult(
        workflow_name=name,
        status=status,
        structured_output={"key": "value"},
        citations=[],
        agent_inferences=[],
        raw_output="raw",
    )


def _make_workflow_class(
    name: str = "TestWorkflow",
    result: WorkflowResult | None = None,
    raises: Exception | None = None,
):
    """Build a concrete BaseWorkflow subclass that returns result or raises.

    Uses type() so execute is defined in the class body, satisfying the ABC
    abstract method requirement at instantiation time.
    """
    from app.workflows.base import BaseWorkflow

    _result = result or _make_result(name)
    _raises = raises

    if _raises is not None:

        def _execute(self, thesis, context):
            raise _raises

    else:

        def _execute(self, thesis, context):
            return _result

    return type(
        "_W",
        (BaseWorkflow,),
        {
            "name": name,
            "description": f"Description of {name}",
            "required_inputs": [],
            "execute": _execute,
        },
    )


# ---------------------------------------------------------------------------
# _serialize_citations
# ---------------------------------------------------------------------------


class TestSerializeCitations:
    def test_empty_list(self):
        assert _serialize_citations([]) == []

    def test_serializes_source_type_as_string(self):
        c = Citation(
            source_type=CitationSourceType.FRED,
            label="FRED:T10Y2Y, retrieved 2024-01-15",
            url=None,
            retrieval_date=date(2024, 1, 15),
        )
        result = _serialize_citations([c])
        assert result[0]["source_type"] == "FRED"

    def test_serializes_retrieval_date_as_iso_string(self):
        c = Citation(
            source_type=CitationSourceType.WEB,
            label="https://example.com",
            url="https://example.com",
            retrieval_date=date(2024, 6, 15),
        )
        result = _serialize_citations([c])
        assert result[0]["retrieval_date"] == "2024-06-15"

    def test_null_url_preserved(self):
        c = Citation(
            source_type=CitationSourceType.FRED,
            label="FRED:CPIAUCSL",
            url=None,
            retrieval_date=date(2024, 1, 1),
        )
        result = _serialize_citations([c])
        assert result[0]["url"] is None

    def test_multiple_citations(self):
        citations = [
            Citation(
                source_type=CitationSourceType.FRED,
                label="FRED:T10Y2Y",
                url=None,
                retrieval_date=date(2024, 1, 1),
            ),
            Citation(
                source_type=CitationSourceType.IBKR,
                label="IBKR:TLT price_history",
                url=None,
                retrieval_date=date(2024, 1, 2),
            ),
        ]
        result = _serialize_citations(citations)
        assert len(result) == 2
        assert result[0]["source_type"] == "FRED"
        assert result[1]["source_type"] == "IBKR"


# ---------------------------------------------------------------------------
# register_workflows
# ---------------------------------------------------------------------------


class TestRegisterWorkflows:
    def test_inserts_new_workflow_when_not_in_registry(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        cls_a = _make_workflow_class("WorkflowA")
        with patch(
            "app.workflows.runner._discover_workflow_classes",
            return_value={"WorkflowA": cls_a},
        ):
            register_workflows(db)

        assert len(added) == 1
        assert isinstance(added[0], WorkflowRegistry)
        assert added[0].name == "WorkflowA"
        assert added[0].description == "Description of WorkflowA"

    def test_updates_existing_workflow_description(self):
        existing = MagicMock(spec=WorkflowRegistry)
        existing.name = "WorkflowA"
        existing.description = "Old description"

        db = _make_db()
        db.query.return_value.filter.return_value.first.return_value = existing

        cls_a = _make_workflow_class("WorkflowA")
        cls_a.description = "New description"

        with patch(
            "app.workflows.runner._discover_workflow_classes",
            return_value={"WorkflowA": cls_a},
        ):
            register_workflows(db)

        assert existing.description == "New description"
        db.add.assert_not_called()

    def test_commits_after_registration(self):
        db = _make_db()
        with patch("app.workflows.runner._discover_workflow_classes", return_value={}):
            register_workflows(db)
        db.commit.assert_called_once()

    def test_registers_multiple_workflows(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        classes = {
            "WorkflowA": _make_workflow_class("WorkflowA"),
            "WorkflowB": _make_workflow_class("WorkflowB"),
        }
        with patch(
            "app.workflows.runner._discover_workflow_classes", return_value=classes
        ):
            register_workflows(db)

        assert len(added) == 2
        names = {r.name for r in added}
        assert names == {"WorkflowA", "WorkflowB"}

    def test_new_registry_row_has_uuid(self):
        db = _make_db()
        added: list[WorkflowRegistry] = []
        db.add.side_effect = added.append

        cls_a = _make_workflow_class("WorkflowA")
        with patch(
            "app.workflows.runner._discover_workflow_classes",
            return_value={"WorkflowA": cls_a},
        ):
            register_workflows(db)

        assert isinstance(added[0].id, uuid.UUID)


# ---------------------------------------------------------------------------
# WorkflowRunner — sequential execution
# ---------------------------------------------------------------------------


class TestWorkflowRunnerSequentialExecution:
    def test_returns_results_for_each_workflow(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()

        cls_a = _make_workflow_class("WorkflowA", _make_result("WorkflowA"))
        cls_b = _make_workflow_class("WorkflowB", _make_result("WorkflowB"))

        results = runner.run(thesis, [cls_a, cls_b])

        assert len(results) == 2
        assert results[0].workflow_name == "WorkflowA"
        assert results[1].workflow_name == "WorkflowB"

    def test_executes_in_order(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()
        call_order: list[str] = []

        def _make_tracking_class(name: str):
            from app.workflows.base import BaseWorkflow

            _name = name

            def _execute(self, thesis, context):
                call_order.append(self.name)
                return _make_result(self.name)

            return type(
                "_W",
                (BaseWorkflow,),
                {
                    "name": _name,
                    "description": f"Description of {_name}",
                    "required_inputs": [],
                    "execute": _execute,
                },
            )

        cls_a = _make_tracking_class("Alpha")
        cls_b = _make_tracking_class("Beta")
        cls_c = _make_tracking_class("Gamma")

        runner.run(thesis, [cls_a, cls_b, cls_c])

        assert call_order == ["Alpha", "Beta", "Gamma"]

    def test_empty_workflow_list_returns_empty(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        results = runner.run(_make_thesis(), [])
        assert results == []


# ---------------------------------------------------------------------------
# WorkflowRunner — context accumulation
# ---------------------------------------------------------------------------


class TestWorkflowRunnerContextAccumulation:
    def test_prior_results_passed_to_subsequent_workflows(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()
        # Capture a snapshot at execute time — context is mutated after execute returns.
        snapshots: list[list] = []

        from app.workflows.base import BaseWorkflow

        class _TrackingWorkflow(BaseWorkflow):
            name = "TrackingWorkflow"
            description = "Tracks context."
            required_inputs = []

            def execute(self, thesis, context):
                snapshots.append(list(context.prior_results))
                return _make_result(self.name)

        cls_a = _make_workflow_class("First", _make_result("First"))
        runner.run(thesis, [cls_a, _TrackingWorkflow])

        assert len(snapshots) == 1
        assert len(snapshots[0]) == 1
        assert snapshots[0][0].workflow_name == "First"

    def test_context_accumulates_across_all_steps(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()
        snapshots: list[list] = []

        from app.workflows.base import BaseWorkflow

        class _LastWorkflow(BaseWorkflow):
            name = "LastWorkflow"
            description = "Last."
            required_inputs = []

            def execute(self, thesis, context):
                snapshots.append(list(context.prior_results))
                return _make_result(self.name)

        cls_a = _make_workflow_class("StepA", _make_result("StepA"))
        cls_b = _make_workflow_class("StepB", _make_result("StepB"))
        runner.run(thesis, [cls_a, cls_b, _LastWorkflow])

        # At _LastWorkflow execute time, prior_results has StepA + StepB (not itself)
        assert len(snapshots[0]) == 2

    def test_pod_settings_passed_through_context(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()
        pod_settings = MagicMock()
        received: list[WorkflowContext] = []

        from app.workflows.base import BaseWorkflow

        class _CheckSettings(BaseWorkflow):
            name = "CheckSettings"
            description = "Checks settings."
            required_inputs = []

            def execute(self, thesis, context):
                received.append(context)
                return _make_result(self.name)

        runner.run(thesis, [_CheckSettings], pod_settings=pod_settings)

        assert received[0].pod_settings is pod_settings


# ---------------------------------------------------------------------------
# WorkflowRunner — failure handling
# ---------------------------------------------------------------------------


class TestWorkflowRunnerFailureHandling:
    def test_continues_after_workflow_failure(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()

        cls_fail = _make_workflow_class("FailingWorkflow", raises=RuntimeError("boom"))
        cls_ok = _make_workflow_class("OkWorkflow", _make_result("OkWorkflow"))

        results = runner.run(thesis, [cls_fail, cls_ok])

        assert len(results) == 2
        assert results[1].workflow_name == "OkWorkflow"
        assert results[1].status == WorkflowStatus.COMPLETED

    def test_failed_result_has_failed_status(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()

        cls_fail = _make_workflow_class("FailingWorkflow", raises=ValueError("bad"))
        results = runner.run(thesis, [cls_fail])

        assert results[0].status == WorkflowStatus.FAILED

    def test_failed_result_raw_output_contains_error_message(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()

        cls_fail = _make_workflow_class(
            "FailingWorkflow", raises=RuntimeError("specific error")
        )
        results = runner.run(thesis, [cls_fail])

        assert "specific error" in results[0].raw_output

    def test_sets_has_partial_results_on_failure(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()
        received_contexts: list[WorkflowContext] = []

        from app.workflows.base import BaseWorkflow

        class _CheckPartial(BaseWorkflow):
            name = "CheckPartial"
            description = "Checks partial flag."
            required_inputs = []

            def execute(self, thesis, context):
                received_contexts.append(context)
                return _make_result(self.name)

        cls_fail = _make_workflow_class("Failing", raises=RuntimeError("err"))
        runner.run(thesis, [cls_fail, _CheckPartial])

        assert received_contexts[0].has_partial_results is True

    def test_has_partial_results_false_when_all_succeed(self):
        db = _make_db()
        runner = WorkflowRunner(db)
        thesis = _make_thesis()
        received_contexts: list[WorkflowContext] = []

        from app.workflows.base import BaseWorkflow

        class _CheckPartial(BaseWorkflow):
            name = "CheckPartial"
            description = "Checks partial flag."
            required_inputs = []

            def execute(self, thesis, context):
                received_contexts.append(context)
                return _make_result(self.name)

        cls_ok = _make_workflow_class("OkWorkflow", _make_result("OkWorkflow"))
        runner.run(thesis, [cls_ok, _CheckPartial])

        assert received_contexts[0].has_partial_results is False


# ---------------------------------------------------------------------------
# WorkflowRunner — DB logging
# ---------------------------------------------------------------------------


class TestWorkflowRunnerDBLogging:
    def test_writes_workflow_run_record_per_step(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        runner = WorkflowRunner(db)
        thesis = _make_thesis()

        cls_a = _make_workflow_class("WorkflowA", _make_result("WorkflowA"))
        cls_b = _make_workflow_class("WorkflowB", _make_result("WorkflowB"))
        runner.run(thesis, [cls_a, cls_b])

        run_records = [r for r in added if isinstance(r, WorkflowRun)]
        assert len(run_records) == 2

    def test_successful_run_record_has_completed_status(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        runner = WorkflowRunner(db)
        thesis = _make_thesis()
        cls_a = _make_workflow_class("WorkflowA", _make_result("WorkflowA"))
        runner.run(thesis, [cls_a])

        run_record = next(r for r in added if isinstance(r, WorkflowRun))
        assert run_record.status == DBWorkflowStatus.completed

    def test_failed_run_record_has_failed_status(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        runner = WorkflowRunner(db)
        thesis = _make_thesis()
        cls_fail = _make_workflow_class("FailingWorkflow", raises=RuntimeError("err"))
        runner.run(thesis, [cls_fail])

        run_record = next(r for r in added if isinstance(r, WorkflowRun))
        assert run_record.status == DBWorkflowStatus.failed

    def test_run_record_thesis_id_matches(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        runner = WorkflowRunner(db)
        thesis = _make_thesis()
        cls_a = _make_workflow_class("WorkflowA", _make_result("WorkflowA"))
        runner.run(thesis, [cls_a])

        run_record = next(r for r in added if isinstance(r, WorkflowRun))
        assert run_record.thesis_id == thesis.id

    def test_run_record_has_started_and_completed_at(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        runner = WorkflowRunner(db)
        cls_a = _make_workflow_class("WorkflowA", _make_result("WorkflowA"))
        runner.run(_make_thesis(), [cls_a])

        run_record = next(r for r in added if isinstance(r, WorkflowRun))
        assert run_record.started_at is not None
        assert run_record.completed_at is not None
        assert run_record.completed_at >= run_record.started_at

    def test_run_record_structured_output_stored(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        result = _make_result("WorkflowA")
        result.structured_output = {"cpi": 3.2}

        runner = WorkflowRunner(db)
        cls_a = _make_workflow_class("WorkflowA", result)
        runner.run(_make_thesis(), [cls_a])

        run_record = next(r for r in added if isinstance(r, WorkflowRun))
        assert run_record.structured_output == {"cpi": 3.2}

    def test_run_record_citations_serialized(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        result = _make_result("WorkflowA")
        result.citations = [
            Citation(
                source_type=CitationSourceType.FRED,
                label="FRED:T10Y2Y",
                url=None,
                retrieval_date=date(2024, 1, 1),
            )
        ]

        runner = WorkflowRunner(db)
        cls_a = _make_workflow_class("WorkflowA", result)
        runner.run(_make_thesis(), [cls_a])

        run_record = next(r for r in added if isinstance(r, WorkflowRun))
        assert isinstance(run_record.citations, list)
        assert run_record.citations[0]["source_type"] == "FRED"

    def test_commits_after_each_step(self):
        db = _make_db()
        runner = WorkflowRunner(db)

        cls_a = _make_workflow_class("WorkflowA", _make_result("WorkflowA"))
        cls_b = _make_workflow_class("WorkflowB", _make_result("WorkflowB"))
        runner.run(_make_thesis(), [cls_a, cls_b])

        assert db.commit.call_count == 2

    def test_failed_run_record_structured_output_is_none(self):
        db = _make_db()
        added: list[object] = []
        db.add.side_effect = added.append

        runner = WorkflowRunner(db)
        cls_fail = _make_workflow_class("FailingWorkflow", raises=RuntimeError("err"))
        runner.run(_make_thesis(), [cls_fail])

        run_record = next(r for r in added if isinstance(r, WorkflowRun))
        assert run_record.structured_output is None
