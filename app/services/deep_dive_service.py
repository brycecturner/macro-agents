"""DeepDiveService — runs a single user-triggered Tier 2 deep dive workflow
against an existing thesis.

Per PRD Section 4.4, deep dives are never generated automatically — the user
triggers one named workflow at a time from the brief page. Unlike the core
research chain, a deep dive doesn't get its prior_results handed to it by an
in-memory WorkflowRunner.run() call — it's invoked independently, potentially
long after the core pipeline finished. This module reconstructs prior_results
from persisted workflow_runs rows so deep dives can still consume core
workflow outputs (e.g. HistoricalAnalogWorkflow's analog periods) exactly as
BacktestWorkflow does within the core chain.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from app.models.thesis import Thesis
from app.models.workflow import WorkflowRun
from app.workflows.base import (
    BaseWorkflow,
    Citation,
    CitationSourceType,
    WorkflowResult,
    WorkflowStatus,
)
from app.workflows.historical_analog_detail import HistoricalAnalogDetailWorkflow
from app.workflows.portfolio_correlation import PortfolioCorrelationWorkflow
from app.workflows.regime_stress_test import RegimeStressTestWorkflow
from app.workflows.runner import WorkflowRunner
from app.workflows.sensitivity_analysis import SensitivityAnalysisWorkflow

DEEP_DIVE_WORKFLOWS: dict[str, type[BaseWorkflow]] = {
    cls.name: cls
    for cls in (
        SensitivityAnalysisWorkflow,
        RegimeStressTestWorkflow,
        PortfolioCorrelationWorkflow,
        HistoricalAnalogDetailWorkflow,
    )
}

# Display labels per PRD Section 4.4's initial deep dive library naming.
DEEP_DIVE_DISPLAY_LABELS: dict[str, str] = {
    "SensitivityAnalysisWorkflow": "Sensitivity Analysis",
    "RegimeStressTestWorkflow": "Regime Stress Test",
    "PortfolioCorrelationWorkflow": "Portfolio Correlation Check",
    "HistoricalAnalogDetailWorkflow": "Historical Analog Detail",
}


class UnknownDeepDiveError(Exception):
    """Raised when a requested deep dive name doesn't match a registered
    deep dive workflow."""


def _deserialize_citations(raw: list[dict] | None) -> list[Citation]:
    citations: list[Citation] = []
    for c in raw or []:
        retrieval_date_str = c.get("retrieval_date")
        citations.append(
            Citation(
                source_type=CitationSourceType(c["source_type"]),
                label=c["label"],
                url=c.get("url"),
                retrieval_date=(
                    date.fromisoformat(retrieval_date_str)
                    if retrieval_date_str
                    else date.today()
                ),
            )
        )
    return citations


def _load_prior_results(thesis_id, db: Session) -> list[WorkflowResult]:
    """Reconstruct WorkflowResult objects from persisted workflow_runs rows.

    Includes every prior run for the thesis (core workflows and any deep
    dives already run) so a deep dive can consume whichever it needs via
    WorkflowContext.get_result(), same as within the sequential core chain.
    """
    runs = (
        db.query(WorkflowRun)
        .filter(WorkflowRun.thesis_id == thesis_id)
        .order_by(WorkflowRun.started_at)
        .all()
    )
    return [
        WorkflowResult(
            workflow_name=run.workflow_name,
            status=WorkflowStatus(run.status.value),
            structured_output=run.structured_output or {},
            citations=_deserialize_citations(run.citations),
            agent_inferences=run.agent_inferences or [],
            raw_output=run.raw_output or "",
        )
        for run in runs
    ]


def run_deep_dive(
    thesis: Thesis,
    workflow_name: str,
    db: Session,
    pod_settings=None,
) -> WorkflowResult:
    """Run a single named deep dive workflow against thesis and persist the run.

    Raises:
        UnknownDeepDiveError: If workflow_name isn't a registered deep dive.
    """
    workflow_cls = DEEP_DIVE_WORKFLOWS.get(workflow_name)
    if workflow_cls is None:
        raise UnknownDeepDiveError(f"Unknown deep dive workflow: {workflow_name!r}")

    prior_results = _load_prior_results(thesis.id, db)
    return WorkflowRunner(db).run_single(
        thesis, workflow_cls, prior_results, pod_settings=pod_settings
    )
