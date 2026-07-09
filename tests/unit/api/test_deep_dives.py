"""Tests for POST /theses/{thesis_id}/deep-dives/{workflow_name}/run."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.models.pod import PodConfig
from app.models.thesis import Thesis
from app.workflows.base import (
    Citation,
    CitationSourceType,
    WorkflowResult,
    WorkflowStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thesis() -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.pod_id = uuid.uuid4()
    return thesis


def _configure_db(
    mock_db: MagicMock, *, thesis: MagicMock | None = None, pod_config=None
) -> None:
    def _query(model: type) -> MagicMock:
        q = MagicMock()
        if model is Thesis:
            q.filter.return_value.first.return_value = thesis
        elif model is PodConfig:
            q.filter.return_value.first.return_value = pod_config
        return q

    mock_db.query.side_effect = _query


def _make_result(
    workflow_name: str = "SensitivityAnalysisWorkflow",
    status: WorkflowStatus = WorkflowStatus.COMPLETED,
    structured_output: dict | None = None,
    citations: list[Citation] | None = None,
    agent_inferences: list[str] | None = None,
) -> WorkflowResult:
    return WorkflowResult(
        workflow_name=workflow_name,
        status=status,
        structured_output=structured_output or {},
        citations=citations or [],
        agent_inferences=agent_inferences or [],
        raw_output="raw",
    )


class TestRunDeepDiveRoute:
    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.post(
            f"/theses/{uuid.uuid4()}/deep-dives/SensitivityAnalysisWorkflow/run"
        )
        assert response.status_code == 404

    def test_returns_404_for_unknown_workflow_name(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)
        response = client.post(f"/theses/{thesis.id}/deep-dives/NotARealWorkflow/run")
        assert response.status_code == 404

    def test_returns_200_and_html_on_success(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result()
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/SensitivityAnalysisWorkflow/run"
            )

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_result_status_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result(status=WorkflowStatus.COMPLETED)
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/SensitivityAnalysisWorkflow/run"
            )

        assert "completed" in response.text

    def test_agent_inferences_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result(
                agent_inferences=["[Agent inference] Sample size is small."]
            )
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/SensitivityAnalysisWorkflow/run"
            )

        assert "[Agent inference] Sample size is small." in response.text

    def test_citations_rendered(self, client: TestClient, mock_db: MagicMock) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        citation = Citation(
            source_type=CitationSourceType.IBKR,
            label="IBKR:TLT price_history, 2025-01-15",
            url=None,
            retrieval_date=None,
        )
        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result(citations=[citation])
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/SensitivityAnalysisWorkflow/run"
            )

        assert "IBKR:TLT price_history, 2025-01-15" in response.text

    def test_sensitivity_analysis_offsets_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result(
                workflow_name="SensitivityAnalysisWorkflow",
                structured_output={
                    "instrument": "TLT",
                    "direction": "long",
                    "offsets": [
                        {
                            "offset_months": 0,
                            "aggregate": {
                                "avg_return": 0.05,
                                "win_rate": 0.67,
                                "n_periods": 2,
                            },
                        }
                    ],
                    "analysis": "Robust to entry timing.",
                },
            )
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/SensitivityAnalysisWorkflow/run"
            )

        assert "5.0%" in response.text
        assert "Robust to entry timing." in response.text

    def test_regime_stress_test_regimes_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result(
                workflow_name="RegimeStressTestWorkflow",
                structured_output={
                    "instrument": "TLT",
                    "direction": "long",
                    "regimes": {
                        "hiking_cycle": {
                            "n_runs": 1,
                            "runs": [],
                            "aggregate": {
                                "avg_return": -0.1,
                                "win_rate": 0.0,
                                "n_periods": 1,
                            },
                        }
                    },
                    "analysis": "Underperforms during hiking cycles.",
                },
            )
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/RegimeStressTestWorkflow/run"
            )

        assert "hiking cycle" in response.text
        assert "Underperforms during hiking cycles." in response.text

    def test_portfolio_correlation_table_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result(
                workflow_name="PortfolioCorrelationWorkflow",
                structured_output={
                    "instrument": "TLT",
                    "correlations": {
                        "GLD": {"correlation": 0.42, "n_overlapping_days": 250}
                    },
                    "analysis": "Moderate correlation with GLD.",
                },
            )
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/PortfolioCorrelationWorkflow/run"
            )

        assert "GLD" in response.text
        assert "0.42" in response.text

    def test_portfolio_correlation_empty_state_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result(
                workflow_name="PortfolioCorrelationWorkflow",
                status=WorkflowStatus.PARTIAL,
                structured_output={
                    "instrument": "TLT",
                    "correlations": {},
                    "analysis": "",
                },
                agent_inferences=[
                    "[Agent inference] No other active positions were found."
                ],
            )
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/PortfolioCorrelationWorkflow/run"
            )

        assert "No other positions found in this pod" in response.text

    def test_historical_analog_detail_periods_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result(
                workflow_name="HistoricalAnalogDetailWorkflow",
                structured_output={
                    "instrument": "TLT",
                    "direction": "long",
                    "periods": [
                        {
                            "label": "2018 rate-hike peak",
                            "start_date": "2018-10",
                            "end_date": "2019-01",
                            "duration_months": 4,
                            "macro_conditions": {},
                            "similarity_rationale": "",
                            "outcome_summary": "",
                            "total_return": 0.03,
                            "max_drawdown": -0.02,
                            "volatility": 0.1,
                            "directionally_correct": True,
                            "n_trading_days": 60,
                            "likely_catalysts": ["[Agent inference] Fed pause."],
                        }
                    ],
                },
            )
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/HistoricalAnalogDetailWorkflow/run"
            )

        assert "2018 rate-hike peak" in response.text
        assert "[Agent inference] Fed pause." in response.text

    def test_failed_status_shows_failure_message(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)

        with patch("app.api.deep_dives.run_deep_dive") as mock_run:
            mock_run.return_value = _make_result(status=WorkflowStatus.FAILED)
            response = client.post(
                f"/theses/{thesis.id}/deep-dives/SensitivityAnalysisWorkflow/run"
            )

        assert "failed to run" in response.text
