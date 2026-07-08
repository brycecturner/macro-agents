"""Tests for ResearchPipelineService and run_research_pipeline_for_thesis."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.integrations.email_client import EmailClientError
from app.models.enums import ThesisStatus
from app.models.log import AuditLog
from app.services.research_pipeline_service import (
    CORE_WORKFLOW_ORDER,
    ResearchPipelineService,
    _send_completion_email,
    run_research_pipeline_for_thesis,
)
from app.workflows.backtest import BacktestWorkflow
from app.workflows.base import WorkflowResult, WorkflowStatus
from app.workflows.falsification_generation import FalsificationGenerationWorkflow
from app.workflows.historical_analog import HistoricalAnalogWorkflow
from app.workflows.instrument_analysis import InstrumentAnalysisWorkflow
from app.workflows.macro_context import MacroContextWorkflow
from app.workflows.recommendation import RecommendationWorkflow
from app.workflows.web_research import WebResearchWorkflow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thesis(**kwargs) -> MagicMock:
    t = MagicMock()
    t.id = kwargs.get("id", uuid.uuid4())
    t.pod_id = kwargs.get("pod_id", uuid.uuid4())
    t.title = kwargs.get("title", "Yield Curve Steepener")
    t.status = kwargs.get("status", ThesisStatus.intake_sent)
    t.instruments = kwargs.get("instruments", [])
    return t


def _make_result(name: str) -> WorkflowResult:
    return WorkflowResult(
        workflow_name=name,
        status=WorkflowStatus.COMPLETED,
        structured_output={},
        citations=[],
        agent_inferences=[],
        raw_output="raw",
    )


def _make_settings(**overrides) -> MagicMock:
    s = MagicMock()
    s.smtp_host = overrides.get("smtp_host", "smtp.example.com")
    s.smtp_port = overrides.get("smtp_port", 587)
    s.smtp_user = overrides.get("smtp_user", "user")
    s.smtp_password = overrides.get("smtp_password", "pass")
    s.smtp_from = overrides.get("smtp_from", "noreply@example.com")
    s.alert_email = overrides.get("alert_email", "pm@example.com")
    s.app_host = overrides.get("app_host", "http://localhost:8000")
    return s


# ---------------------------------------------------------------------------
# CORE_WORKFLOW_ORDER
# ---------------------------------------------------------------------------


class TestCoreWorkflowOrder:
    def test_matches_prd_section_4_3_sequence(self):
        assert CORE_WORKFLOW_ORDER == [
            MacroContextWorkflow,
            HistoricalAnalogWorkflow,
            InstrumentAnalysisWorkflow,
            WebResearchWorkflow,
            BacktestWorkflow,
            FalsificationGenerationWorkflow,
            RecommendationWorkflow,
        ]


# ---------------------------------------------------------------------------
# ResearchPipelineService.run
# ---------------------------------------------------------------------------


class TestResearchPipelineServiceRun:
    @pytest.fixture(autouse=True)
    def _mock_store_brief(self):
        with patch("app.services.research_pipeline_service.store_brief") as mock:
            yield mock

    def test_runs_workflow_runner_with_core_order(self):
        thesis = _make_thesis()
        db = MagicMock()
        mock_runner = MagicMock()
        mock_runner.run.return_value = []
        with (
            patch(
                "app.services.research_pipeline_service.WorkflowRunner",
                return_value=mock_runner,
            ),
            patch("app.services.research_pipeline_service._send_completion_email"),
        ):
            ResearchPipelineService.run(thesis, db)

        mock_runner.run.assert_called_once_with(thesis, CORE_WORKFLOW_ORDER, None)

    def test_sets_status_to_researched(self):
        thesis = _make_thesis(status=ThesisStatus.intake_sent)
        db = MagicMock()
        with (
            patch("app.services.research_pipeline_service.WorkflowRunner") as cls,
            patch("app.services.research_pipeline_service._send_completion_email"),
        ):
            cls.return_value.run.return_value = []
            ResearchPipelineService.run(thesis, db)

        assert thesis.status == ThesisStatus.researched

    def test_writes_audit_log_entry(self):
        thesis = _make_thesis()
        db = MagicMock()
        added = []
        db.add.side_effect = added.append
        with (
            patch("app.services.research_pipeline_service.WorkflowRunner") as cls,
            patch("app.services.research_pipeline_service._send_completion_email"),
        ):
            cls.return_value.run.return_value = []
            ResearchPipelineService.run(thesis, db)

        audit_rows = [a for a in added if isinstance(a, AuditLog)]
        assert len(audit_rows) == 1
        assert audit_rows[0].entity_type == "thesis"
        assert audit_rows[0].entity_id == thesis.id
        assert audit_rows[0].action == "thesis_status_changed"
        assert audit_rows[0].new_value == {"status": "researched"}
        assert audit_rows[0].changed_by == "research_pipeline_agent"

    def test_audit_log_pod_id_matches_thesis(self):
        thesis = _make_thesis()
        db = MagicMock()
        added = []
        db.add.side_effect = added.append
        with (
            patch("app.services.research_pipeline_service.WorkflowRunner") as cls,
            patch("app.services.research_pipeline_service._send_completion_email"),
        ):
            cls.return_value.run.return_value = []
            ResearchPipelineService.run(thesis, db)

        audit_row = next(a for a in added if isinstance(a, AuditLog))
        assert audit_row.pod_id == thesis.pod_id

    def test_commits_after_status_update(self):
        thesis = _make_thesis()
        db = MagicMock()
        with (
            patch("app.services.research_pipeline_service.WorkflowRunner") as cls,
            patch("app.services.research_pipeline_service._send_completion_email"),
        ):
            cls.return_value.run.return_value = []
            ResearchPipelineService.run(thesis, db)

        db.commit.assert_called_once()

    def test_returns_workflow_results(self):
        thesis = _make_thesis()
        db = MagicMock()
        results = [
            _make_result("MacroContextWorkflow"),
            _make_result("RecommendationWorkflow"),
        ]
        with (
            patch("app.services.research_pipeline_service.WorkflowRunner") as cls,
            patch("app.services.research_pipeline_service._send_completion_email"),
        ):
            cls.return_value.run.return_value = results
            returned = ResearchPipelineService.run(thesis, db)

        assert returned == results

    def test_passes_pod_settings_through(self):
        thesis = _make_thesis()
        db = MagicMock()
        pod_settings = MagicMock()
        mock_runner = MagicMock()
        mock_runner.run.return_value = []
        with (
            patch(
                "app.services.research_pipeline_service.WorkflowRunner",
                return_value=mock_runner,
            ),
            patch("app.services.research_pipeline_service._send_completion_email"),
        ):
            ResearchPipelineService.run(thesis, db, pod_settings)

        mock_runner.run.assert_called_once_with(
            thesis, CORE_WORKFLOW_ORDER, pod_settings
        )

    def test_sends_completion_email(self):
        thesis = _make_thesis()
        db = MagicMock()
        with (
            patch("app.services.research_pipeline_service.WorkflowRunner") as cls,
            patch(
                "app.services.research_pipeline_service._send_completion_email"
            ) as mock_send,
        ):
            cls.return_value.run.return_value = []
            ResearchPipelineService.run(thesis, db)

        mock_send.assert_called_once_with(thesis)

    def test_assembles_and_stores_brief(self, _mock_store_brief: MagicMock):
        thesis = _make_thesis()
        db = MagicMock()
        with (
            patch("app.services.research_pipeline_service.WorkflowRunner") as cls,
            patch("app.services.research_pipeline_service._send_completion_email"),
        ):
            cls.return_value.run.return_value = []
            ResearchPipelineService.run(thesis, db)

        _mock_store_brief.assert_called_once_with(thesis, db)

    def test_brief_stored_before_status_commit(self, _mock_store_brief: MagicMock):
        thesis = _make_thesis()
        db = MagicMock()
        call_order: list[str] = []
        _mock_store_brief.side_effect = lambda *a, **k: call_order.append("store_brief")
        db.commit.side_effect = lambda: call_order.append("commit")
        with (
            patch("app.services.research_pipeline_service.WorkflowRunner") as cls,
            patch("app.services.research_pipeline_service._send_completion_email"),
        ):
            cls.return_value.run.return_value = []
            ResearchPipelineService.run(thesis, db)

        assert call_order == ["store_brief", "commit"]


# ---------------------------------------------------------------------------
# _send_completion_email
# ---------------------------------------------------------------------------


class TestSendCompletionEmail:
    def test_skips_when_smtp_host_missing(self):
        thesis = _make_thesis()
        settings = _make_settings(smtp_host=None)
        with (
            patch(
                "app.services.research_pipeline_service.get_settings",
                return_value=settings,
            ),
            patch("app.services.research_pipeline_service.EmailClient") as mock_cls,
        ):
            _send_completion_email(thesis)
        mock_cls.assert_not_called()

    def test_skips_when_alert_email_missing(self):
        thesis = _make_thesis()
        settings = _make_settings(alert_email=None)
        with (
            patch(
                "app.services.research_pipeline_service.get_settings",
                return_value=settings,
            ),
            patch("app.services.research_pipeline_service.EmailClient") as mock_cls,
        ):
            _send_completion_email(thesis)
        mock_cls.assert_not_called()

    def test_sends_to_alert_email(self):
        thesis = _make_thesis()
        settings = _make_settings()
        mock_instance = MagicMock()
        with (
            patch(
                "app.services.research_pipeline_service.get_settings",
                return_value=settings,
            ),
            patch(
                "app.services.research_pipeline_service.EmailClient",
                return_value=mock_instance,
            ),
        ):
            _send_completion_email(thesis)

        mock_instance.send.assert_called_once()
        _, kwargs = mock_instance.send.call_args
        assert kwargs["to"] == "pm@example.com"

    def test_email_body_contains_brief_link(self):
        thesis = _make_thesis()
        settings = _make_settings(app_host="http://localhost:8000")
        mock_instance = MagicMock()
        with (
            patch(
                "app.services.research_pipeline_service.get_settings",
                return_value=settings,
            ),
            patch(
                "app.services.research_pipeline_service.EmailClient",
                return_value=mock_instance,
            ),
        ):
            _send_completion_email(thesis)

        _, kwargs = mock_instance.send.call_args
        assert f"http://localhost:8000/theses/{thesis.id}" in kwargs["body"]

    def test_email_subject_includes_thesis_title(self):
        thesis = _make_thesis(title="Inflation Breakout")
        settings = _make_settings()
        mock_instance = MagicMock()
        with (
            patch(
                "app.services.research_pipeline_service.get_settings",
                return_value=settings,
            ),
            patch(
                "app.services.research_pipeline_service.EmailClient",
                return_value=mock_instance,
            ),
        ):
            _send_completion_email(thesis)

        _, kwargs = mock_instance.send.call_args
        assert "Inflation Breakout" in kwargs["subject"]

    def test_swallows_email_client_error(self):
        thesis = _make_thesis()
        settings = _make_settings()
        mock_instance = MagicMock()
        mock_instance.send.side_effect = EmailClientError("smtp down")
        with (
            patch(
                "app.services.research_pipeline_service.get_settings",
                return_value=settings,
            ),
            patch(
                "app.services.research_pipeline_service.EmailClient",
                return_value=mock_instance,
            ),
        ):
            _send_completion_email(thesis)  # must not raise


# ---------------------------------------------------------------------------
# run_research_pipeline_for_thesis
# ---------------------------------------------------------------------------


class TestRunResearchPipelineForThesis:
    def test_loads_thesis_and_runs_pipeline(self):
        thesis_id = uuid.uuid4()
        thesis = _make_thesis(id=thesis_id)
        db = MagicMock()
        db.query.return_value.filter.return_value.first.side_effect = [
            thesis,
            MagicMock(),
        ]
        factory = MagicMock(return_value=db)
        with (
            patch(
                "app.services.research_pipeline_service.get_session_factory",
                return_value=factory,
            ),
            patch("app.services.research_pipeline_service.PodSettings"),
            patch(
                "app.services.research_pipeline_service.ResearchPipelineService.run"
            ) as mock_run,
        ):
            run_research_pipeline_for_thesis(thesis_id)

        mock_run.assert_called_once()
        args, _kwargs = mock_run.call_args
        assert args[0] is thesis

    def test_logs_and_returns_when_thesis_not_found(self):
        thesis_id = uuid.uuid4()
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        factory = MagicMock(return_value=db)
        with (
            patch(
                "app.services.research_pipeline_service.get_session_factory",
                return_value=factory,
            ),
            patch(
                "app.services.research_pipeline_service.ResearchPipelineService.run"
            ) as mock_run,
        ):
            run_research_pipeline_for_thesis(thesis_id)

        mock_run.assert_not_called()

    def test_closes_session_on_success(self):
        thesis_id = uuid.uuid4()
        thesis = _make_thesis(id=thesis_id)
        db = MagicMock()
        db.query.return_value.filter.return_value.first.side_effect = [
            thesis,
            MagicMock(),
        ]
        factory = MagicMock(return_value=db)
        with (
            patch(
                "app.services.research_pipeline_service.get_session_factory",
                return_value=factory,
            ),
            patch("app.services.research_pipeline_service.PodSettings"),
            patch("app.services.research_pipeline_service.ResearchPipelineService.run"),
        ):
            run_research_pipeline_for_thesis(thesis_id)

        db.close.assert_called_once()

    def test_closes_session_even_on_failure(self):
        thesis_id = uuid.uuid4()
        thesis = _make_thesis(id=thesis_id)
        db = MagicMock()
        db.query.return_value.filter.return_value.first.side_effect = [
            thesis,
            MagicMock(),
        ]
        factory = MagicMock(return_value=db)
        with (
            patch(
                "app.services.research_pipeline_service.get_session_factory",
                return_value=factory,
            ),
            patch("app.services.research_pipeline_service.PodSettings"),
            patch(
                "app.services.research_pipeline_service.ResearchPipelineService.run",
                side_effect=RuntimeError("boom"),
            ),
        ):
            run_research_pipeline_for_thesis(thesis_id)  # must not raise

        db.close.assert_called_once()

    def test_loads_pod_settings_from_pod_config(self):
        thesis_id = uuid.uuid4()
        thesis = _make_thesis(id=thesis_id)
        pod_config = MagicMock()
        db = MagicMock()
        db.query.return_value.filter.return_value.first.side_effect = [
            thesis,
            pod_config,
        ]
        factory = MagicMock(return_value=db)

        with (
            patch(
                "app.services.research_pipeline_service.get_session_factory",
                return_value=factory,
            ),
            patch(
                "app.services.research_pipeline_service.PodSettings"
            ) as mock_pod_settings_cls,
            patch("app.services.research_pipeline_service.ResearchPipelineService.run"),
        ):
            run_research_pipeline_for_thesis(thesis_id)

        mock_pod_settings_cls.from_orm.assert_called_once_with(pod_config)

    def test_pod_settings_none_when_no_pod_config(self):
        thesis_id = uuid.uuid4()
        thesis = _make_thesis(id=thesis_id)
        db = MagicMock()
        db.query.return_value.filter.return_value.first.side_effect = [thesis, None]
        factory = MagicMock(return_value=db)

        with (
            patch(
                "app.services.research_pipeline_service.get_session_factory",
                return_value=factory,
            ),
            patch(
                "app.services.research_pipeline_service.ResearchPipelineService.run"
            ) as mock_run,
        ):
            run_research_pipeline_for_thesis(thesis_id)

        _args, kwargs = mock_run.call_args
        assert mock_run.call_args[0][2] is None
