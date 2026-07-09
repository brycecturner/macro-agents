"""Tests for DeepDiveService — reconstructing context and running a single
user-triggered deep dive workflow."""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from app.models.enums import WorkflowStatus as DBWorkflowStatus
from app.services.deep_dive_service import (
    DEEP_DIVE_WORKFLOWS,
    UnknownDeepDiveError,
    _deserialize_citations,
    _load_prior_results,
    run_deep_dive,
)
from app.workflows.base import CitationSourceType, WorkflowResult, WorkflowStatus
from app.workflows.historical_analog_detail import HistoricalAnalogDetailWorkflow
from app.workflows.portfolio_correlation import PortfolioCorrelationWorkflow
from app.workflows.regime_stress_test import RegimeStressTestWorkflow
from app.workflows.sensitivity_analysis import SensitivityAnalysisWorkflow

# ---------------------------------------------------------------------------
# DEEP_DIVE_WORKFLOWS
# ---------------------------------------------------------------------------


class TestDeepDiveWorkflowsRegistry:
    def test_contains_all_four_deep_dives(self) -> None:
        assert set(DEEP_DIVE_WORKFLOWS.keys()) == {
            "SensitivityAnalysisWorkflow",
            "RegimeStressTestWorkflow",
            "PortfolioCorrelationWorkflow",
            "HistoricalAnalogDetailWorkflow",
        }

    def test_maps_to_correct_classes(self) -> None:
        assert DEEP_DIVE_WORKFLOWS["SensitivityAnalysisWorkflow"] is (
            SensitivityAnalysisWorkflow
        )
        assert (
            DEEP_DIVE_WORKFLOWS["RegimeStressTestWorkflow"] is RegimeStressTestWorkflow
        )
        assert DEEP_DIVE_WORKFLOWS["PortfolioCorrelationWorkflow"] is (
            PortfolioCorrelationWorkflow
        )
        assert DEEP_DIVE_WORKFLOWS["HistoricalAnalogDetailWorkflow"] is (
            HistoricalAnalogDetailWorkflow
        )


# ---------------------------------------------------------------------------
# _deserialize_citations
# ---------------------------------------------------------------------------


class TestDeserializeCitations:
    def test_empty_list_for_none(self) -> None:
        assert _deserialize_citations(None) == []

    def test_empty_list_for_empty(self) -> None:
        assert _deserialize_citations([]) == []

    def test_reconstructs_source_type_and_label(self) -> None:
        raw = [
            {
                "source_type": "FRED",
                "label": "FRED:T10Y2Y, retrieved 2024-01-15",
                "url": None,
                "retrieval_date": "2024-01-15",
            }
        ]
        result = _deserialize_citations(raw)
        assert result[0].source_type == CitationSourceType.FRED
        assert result[0].label == "FRED:T10Y2Y, retrieved 2024-01-15"
        assert result[0].retrieval_date == date(2024, 1, 15)

    def test_null_url_preserved(self) -> None:
        raw = [
            {
                "source_type": "IBKR",
                "label": "IBKR:TLT price_history",
                "url": None,
                "retrieval_date": "2024-01-01",
            }
        ]
        result = _deserialize_citations(raw)
        assert result[0].url is None

    def test_missing_retrieval_date_defaults_to_today(self) -> None:
        raw = [
            {
                "source_type": "web",
                "label": "https://example.com",
                "url": "https://example.com",
                "retrieval_date": None,
            }
        ]
        result = _deserialize_citations(raw)
        assert result[0].retrieval_date == date.today()

    def test_multiple_citations_preserved_in_order(self) -> None:
        raw = [
            {
                "source_type": "FRED",
                "label": "a",
                "url": None,
                "retrieval_date": "2024-01-01",
            },
            {
                "source_type": "IBKR",
                "label": "b",
                "url": None,
                "retrieval_date": "2024-01-02",
            },
        ]
        result = _deserialize_citations(raw)
        assert [c.label for c in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# _load_prior_results
# ---------------------------------------------------------------------------


def _make_run_row(
    workflow_name: str,
    status: DBWorkflowStatus = DBWorkflowStatus.completed,
    structured_output: dict | None = None,
    citations: list[dict] | None = None,
    agent_inferences: list[str] | None = None,
    raw_output: str | None = "raw",
) -> MagicMock:
    row = MagicMock()
    row.workflow_name = workflow_name
    row.status = status
    row.structured_output = structured_output
    row.citations = citations
    row.agent_inferences = agent_inferences
    row.raw_output = raw_output
    return row


def _make_db_with_runs(runs: list[MagicMock]) -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = (
        runs
    )
    return db


class TestLoadPriorResults:
    def test_returns_empty_list_when_no_runs(self) -> None:
        db = _make_db_with_runs([])
        result = _load_prior_results(uuid.uuid4(), db)
        assert result == []

    def test_reconstructs_workflow_result_per_row(self) -> None:
        row = _make_run_row("MacroContextWorkflow", structured_output={"summary": "x"})
        db = _make_db_with_runs([row])
        result = _load_prior_results(uuid.uuid4(), db)
        assert len(result) == 1
        assert isinstance(result[0], WorkflowResult)
        assert result[0].workflow_name == "MacroContextWorkflow"
        assert result[0].structured_output == {"summary": "x"}

    def test_status_mapped_to_base_workflow_status(self) -> None:
        row = _make_run_row("A", status=DBWorkflowStatus.failed)
        db = _make_db_with_runs([row])
        result = _load_prior_results(uuid.uuid4(), db)
        assert result[0].status == WorkflowStatus.FAILED

    def test_null_structured_output_becomes_empty_dict(self) -> None:
        row = _make_run_row("A", structured_output=None)
        db = _make_db_with_runs([row])
        result = _load_prior_results(uuid.uuid4(), db)
        assert result[0].structured_output == {}

    def test_null_agent_inferences_becomes_empty_list(self) -> None:
        row = _make_run_row("A", agent_inferences=None)
        db = _make_db_with_runs([row])
        result = _load_prior_results(uuid.uuid4(), db)
        assert result[0].agent_inferences == []

    def test_preserves_order_of_multiple_runs(self) -> None:
        rows = [_make_run_row("First"), _make_run_row("Second")]
        db = _make_db_with_runs(rows)
        result = _load_prior_results(uuid.uuid4(), db)
        assert [r.workflow_name for r in result] == ["First", "Second"]


# ---------------------------------------------------------------------------
# run_deep_dive
# ---------------------------------------------------------------------------


class TestRunDeepDive:
    def test_raises_for_unknown_workflow_name(self) -> None:
        db = _make_db_with_runs([])
        thesis = MagicMock()
        with pytest.raises(UnknownDeepDiveError):
            run_deep_dive(thesis, "NotARealWorkflow", db)

    def test_calls_run_single_with_correct_workflow_class(self) -> None:
        db = _make_db_with_runs([])
        thesis = MagicMock()
        thesis.id = uuid.uuid4()

        with patch("app.services.deep_dive_service.WorkflowRunner") as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_single.return_value = MagicMock(spec=WorkflowResult)
            run_deep_dive(thesis, "SensitivityAnalysisWorkflow", db)

        mock_runner.run_single.assert_called_once()
        call_args = mock_runner.run_single.call_args
        assert call_args[0][0] is thesis
        assert call_args[0][1] is SensitivityAnalysisWorkflow

    def test_passes_reconstructed_prior_results(self) -> None:
        row = _make_run_row(
            "HistoricalAnalogWorkflow", structured_output={"analogs": []}
        )
        db = _make_db_with_runs([row])
        thesis = MagicMock()
        thesis.id = uuid.uuid4()

        with patch("app.services.deep_dive_service.WorkflowRunner") as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_single.return_value = MagicMock(spec=WorkflowResult)
            run_deep_dive(thesis, "RegimeStressTestWorkflow", db)

        call_args = mock_runner.run_single.call_args
        prior_results = call_args[0][2]
        assert len(prior_results) == 1
        assert prior_results[0].workflow_name == "HistoricalAnalogWorkflow"

    def test_passes_pod_settings_through(self) -> None:
        db = _make_db_with_runs([])
        thesis = MagicMock()
        thesis.id = uuid.uuid4()
        pod_settings = MagicMock()

        with patch("app.services.deep_dive_service.WorkflowRunner") as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_single.return_value = MagicMock(spec=WorkflowResult)
            run_deep_dive(
                thesis,
                "PortfolioCorrelationWorkflow",
                db,
                pod_settings=pod_settings,
            )

        assert mock_runner.run_single.call_args.kwargs["pod_settings"] is pod_settings

    def test_returns_result_from_run_single(self) -> None:
        db = _make_db_with_runs([])
        thesis = MagicMock()
        thesis.id = uuid.uuid4()
        expected = MagicMock(spec=WorkflowResult)

        with patch("app.services.deep_dive_service.WorkflowRunner") as mock_runner_cls:
            mock_runner = mock_runner_cls.return_value
            mock_runner.run_single.return_value = expected
            result = run_deep_dive(thesis, "HistoricalAnalogDetailWorkflow", db)

        assert result is expected
