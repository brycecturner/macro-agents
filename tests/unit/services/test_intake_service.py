"""Tests for IntakeService."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from app.integrations.anthropic_client import AnthropicResponse
from app.models.enums import ThesisStatus
from app.models.log import AuditLog
from app.services.intake_service import IntakeService


def _make_thesis(**kwargs) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.pod_id = uuid.uuid4()
    t.title = kwargs.get("title", "Yield Curve Steepener")
    t.direction.value = kwargs.get("direction", "long")
    t.time_horizon = kwargs.get("time_horizon", "6 months")
    t.notes = kwargs.get("notes", "Long TLT as the curve steepens.")
    t.status = kwargs.get("status", ThesisStatus.intake_sent)
    t.thesis_confirmed = kwargs.get("thesis_confirmed", True)
    t.intake_sent_at = kwargs.get("intake_sent_at", None)
    t.intake_responded_at = kwargs.get("intake_responded_at", None)
    return t


def _make_anthropic_response(
    content: str = "## Thesis as I Understood It\nLong TLT.",
) -> AnthropicResponse:
    return AnthropicResponse(
        content=content,
        model="claude-opus-4-6",
        input_tokens=100,
        output_tokens=50,
        stop_reason="end_turn",
    )


# ---------------------------------------------------------------------------
# generate_intake_message
# ---------------------------------------------------------------------------


class TestGenerateIntakeMessage:
    @pytest.fixture
    def mock_client(self, monkeypatch) -> MagicMock:
        client = MagicMock()
        client.complete.return_value = _make_anthropic_response()
        monkeypatch.setattr(
            "app.services.intake_service.AnthropicClient",
            lambda api_key, db: client,
        )
        return client

    def test_stores_intake_message(self, mock_client: MagicMock) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().generate_intake_message(thesis, db, "test-key")
        assert thesis.intake_message == "## Thesis as I Understood It\nLong TLT."

    def test_sets_status_to_intake_sent(self, mock_client: MagicMock) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().generate_intake_message(thesis, db, "test-key")
        assert thesis.status == ThesisStatus.intake_sent

    def test_sets_intake_sent_at(self, mock_client: MagicMock) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().generate_intake_message(thesis, db, "test-key")
        assert thesis.intake_sent_at is not None

    def test_commits_db(self, mock_client: MagicMock) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().generate_intake_message(thesis, db, "test-key")
        db.commit.assert_called_once()

    def test_calls_anthropic_with_opus_model(self, mock_client: MagicMock) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().generate_intake_message(thesis, db, "test-key")
        _, kwargs = mock_client.complete.call_args
        assert kwargs["model"] == "claude-opus-4-6"

    def test_calls_anthropic_with_intake_task_type(
        self, mock_client: MagicMock
    ) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().generate_intake_message(thesis, db, "test-key")
        _, kwargs = mock_client.complete.call_args
        assert kwargs["task_type"] == "intake"

    def test_passes_thesis_id_for_cost_tracking(self, mock_client: MagicMock) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().generate_intake_message(thesis, db, "test-key")
        _, kwargs = mock_client.complete.call_args
        assert kwargs["thesis_id"] == thesis.id

    def test_thesis_notes_included_in_prompt(self, mock_client: MagicMock) -> None:
        thesis = _make_thesis(notes="I think TLT goes up when the curve steepens.")
        db = MagicMock()
        IntakeService().generate_intake_message(thesis, db, "test-key")
        prompt = mock_client.complete.call_args.kwargs["messages"][0]["content"]
        assert "I think TLT goes up" in prompt


# ---------------------------------------------------------------------------
# handle_intake_response
# ---------------------------------------------------------------------------


class TestHandleIntakeResponse:
    def test_stores_user_response(self) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().handle_intake_response(thesis, "Confirmed, use TLT.", db)
        assert thesis.intake_user_response == "Confirmed, use TLT."

    def test_strips_whitespace_from_response(self) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().handle_intake_response(thesis, "  Yes, correct.  ", db)
        assert thesis.intake_user_response == "Yes, correct."

    def test_empty_response_stored_as_none(self) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().handle_intake_response(thesis, "   ", db)
        assert thesis.intake_user_response is None

    def test_sets_intake_responded_at(self) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().handle_intake_response(thesis, "OK", db)
        assert thesis.intake_responded_at is not None

    def test_commits_db(self) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService().handle_intake_response(thesis, "OK", db)
        db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# process_timeout
# ---------------------------------------------------------------------------


class TestProcessTimeout:
    @pytest.fixture(autouse=True)
    def mock_pipeline(self, monkeypatch) -> MagicMock:
        mock = MagicMock()
        monkeypatch.setattr(
            "app.services.intake_service.run_research_pipeline_for_thesis", mock
        )
        return mock

    def test_sets_thesis_confirmed_false(self) -> None:
        thesis = _make_thesis(thesis_confirmed=True)
        db = MagicMock()
        IntakeService.process_timeout(thesis, db)
        assert thesis.thesis_confirmed is False

    def test_commits_db(self) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService.process_timeout(thesis, db)
        db.commit.assert_called_once()

    def test_triggers_research_pipeline(self, mock_pipeline: MagicMock) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService.process_timeout(thesis, db)
        mock_pipeline.assert_called_once_with(thesis.id)


# ---------------------------------------------------------------------------
# acknowledge_intake
# ---------------------------------------------------------------------------


class TestAcknowledgeIntake:
    def test_sets_thesis_confirmed_true(self) -> None:
        thesis = _make_thesis(thesis_confirmed=False)
        db = MagicMock()
        IntakeService.acknowledge_intake(thesis, db)
        assert thesis.thesis_confirmed is True

    def test_commits_db(self) -> None:
        thesis = _make_thesis()
        db = MagicMock()
        IntakeService.acknowledge_intake(thesis, db)
        db.commit.assert_called_once()

    def test_writes_audit_log_entry(self) -> None:
        thesis = _make_thesis(thesis_confirmed=False)
        db = MagicMock()
        added = []
        db.add.side_effect = added.append
        IntakeService.acknowledge_intake(thesis, db)

        audit_rows = [a for a in added if isinstance(a, AuditLog)]
        assert len(audit_rows) == 1
        assert audit_rows[0].entity_type == "thesis"
        assert audit_rows[0].entity_id == thesis.id
        assert audit_rows[0].pod_id == thesis.pod_id
        assert audit_rows[0].action == "intake_acknowledged"
        assert audit_rows[0].previous_value == {"thesis_confirmed": False}
        assert audit_rows[0].new_value == {"thesis_confirmed": True}
        assert audit_rows[0].changed_by == "user"


# ---------------------------------------------------------------------------
# check_and_process_timeouts
# ---------------------------------------------------------------------------


class TestCheckAndProcessTimeouts:
    @pytest.fixture(autouse=True)
    def mock_pipeline(self, monkeypatch) -> MagicMock:
        mock = MagicMock()
        monkeypatch.setattr(
            "app.services.intake_service.run_research_pipeline_for_thesis", mock
        )
        return mock

    def _timed_out_thesis(self) -> MagicMock:
        return _make_thesis(
            thesis_confirmed=True,
            intake_sent_at=datetime.now(UTC) - timedelta(hours=25),
            intake_responded_at=None,
        )

    def test_returns_count_of_processed_theses(self) -> None:
        thesis1 = self._timed_out_thesis()
        thesis2 = self._timed_out_thesis()
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [thesis1, thesis2]

        count = IntakeService.check_and_process_timeouts(db, intake_timeout_hours=24)

        assert count == 2

    def test_sets_thesis_confirmed_false_on_each(self) -> None:
        thesis = self._timed_out_thesis()
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [thesis]

        IntakeService.check_and_process_timeouts(db, intake_timeout_hours=24)

        assert thesis.thesis_confirmed is False

    def test_returns_zero_when_nothing_timed_out(self) -> None:
        db = MagicMock()
        db.query.return_value.filter.return_value.all.return_value = []

        count = IntakeService.check_and_process_timeouts(db, intake_timeout_hours=24)

        assert count == 0
