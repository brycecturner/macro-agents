"""ResearchPipelineService — orchestrates the 7 core research workflows.

Triggered automatically after intake confirmation (user response) or intake
timeout — never manually. Runs all core workflows sequentially via
WorkflowRunner in the order defined in PRD Section 4.3, marks the thesis
'researched', and sends the single brief-completion email for the idea
pipeline flow.
"""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.orm import Session

from app.core.database import get_session_factory
from app.core.pod_settings import PodSettings
from app.core.settings import get_settings
from app.integrations.email_client import EmailClient, EmailClientError
from app.models.enums import ThesisStatus
from app.models.log import AuditLog
from app.models.pod import PodConfig
from app.models.thesis import Thesis
from app.workflows.backtest import BacktestWorkflow
from app.workflows.base import BaseWorkflow, WorkflowResult
from app.workflows.falsification_generation import FalsificationGenerationWorkflow
from app.workflows.historical_analog import HistoricalAnalogWorkflow
from app.workflows.instrument_analysis import InstrumentAnalysisWorkflow
from app.workflows.macro_context import MacroContextWorkflow
from app.workflows.recommendation import RecommendationWorkflow
from app.workflows.runner import WorkflowRunner
from app.workflows.web_research import WebResearchWorkflow

logger = logging.getLogger(__name__)

# Execution order per PRD Section 4.3.
CORE_WORKFLOW_ORDER: list[type[BaseWorkflow]] = [
    MacroContextWorkflow,
    HistoricalAnalogWorkflow,
    InstrumentAnalysisWorkflow,
    WebResearchWorkflow,
    BacktestWorkflow,
    FalsificationGenerationWorkflow,
    RecommendationWorkflow,
]


class ResearchPipelineService:
    """Runs the full research pipeline for a thesis and notifies the user."""

    @staticmethod
    def run(
        thesis: Thesis,
        db: Session,
        pod_settings: PodSettings | None = None,
    ) -> list[WorkflowResult]:
        """Execute all core workflows, mark the thesis researched, email the user.

        A failed workflow step does not stop the pipeline — WorkflowRunner
        continues so the brief is assembled from whatever results are
        available (per PRD Section 7: workflow failures are partial, not total).
        """
        results = WorkflowRunner(db).run(thesis, CORE_WORKFLOW_ORDER, pod_settings)

        previous_status = (
            thesis.status.value if hasattr(thesis.status, "value") else thesis.status
        )
        thesis.status = ThesisStatus.researched
        db.add(
            AuditLog(
                id=uuid.uuid4(),
                pod_id=thesis.pod_id,
                entity_id=thesis.id,
                entity_type="thesis",
                action="thesis_status_changed",
                previous_value={"status": previous_status},
                new_value={"status": ThesisStatus.researched.value},
                changed_by="research_pipeline_agent",
            )
        )
        db.commit()
        logger.info("Research pipeline completed for thesis %s", thesis.id)

        _send_completion_email(thesis)

        return results


def _send_completion_email(thesis: Thesis) -> None:
    """Send the single brief-completion email for the idea pipeline flow.

    Per PRD Section 4.3, this is the first and only email in the idea
    pipeline flow — outbound-only, no reply handling. Failures are logged
    and swallowed; a missed notification email does not roll back the
    completed research pipeline.
    """
    settings = get_settings()
    if not settings.smtp_host or not settings.alert_email or not settings.smtp_from:
        logger.warning(
            "SMTP not configured — skipping brief completion email for thesis %s",
            thesis.id,
        )
        return

    brief_url = f"{settings.app_host}/theses/{thesis.id}"
    body = (
        f'Your research brief for "{thesis.title}" is ready.\n\n'
        f"View the brief: {brief_url}\n"
    )

    try:
        EmailClient(
            host=settings.smtp_host,
            port=settings.smtp_port,
            username=settings.smtp_user,
            password=settings.smtp_password,
            from_addr=settings.smtp_from,
        ).send(
            to=settings.alert_email,
            subject=f'Research brief ready: "{thesis.title}"',
            body=body,
        )
    except EmailClientError:
        logger.exception(
            "Failed to send brief completion email for thesis %s", thesis.id
        )


def run_research_pipeline_for_thesis(thesis_id: uuid.UUID) -> None:
    """Entry point for background execution — opens its own DB session.

    Used both as a FastAPI BackgroundTasks target (after an intake response)
    and as a direct call from the intake timeout job, which already runs in
    its own background thread.
    """
    db = get_session_factory()()
    try:
        thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
        if thesis is None:
            logger.error("Thesis %s not found for research pipeline", thesis_id)
            return
        config = db.query(PodConfig).filter(PodConfig.pod_id == thesis.pod_id).first()
        pod_settings = PodSettings.from_orm(config) if config else None
        ResearchPipelineService.run(thesis, db, pod_settings)
    except Exception:
        logger.exception("Research pipeline failed for thesis %s", thesis_id)
    finally:
        db.close()
