"""Workflow registry population and sequential workflow runner.

`register_workflows` is called at app startup and upserts every discovered
BaseWorkflow subclass into the workflow_registry table.

`WorkflowRunner` executes a list of workflow classes sequentially, accumulates
context, and persists a WorkflowRun record per step.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import uuid
from datetime import UTC, datetime

import app.workflows
from app.models.enums import WorkflowStatus as DBWorkflowStatus
from app.models.workflow import WorkflowRegistry, WorkflowRun
from app.workflows.base import (
    BaseWorkflow,
    Citation,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_citations(citations: list[Citation]) -> list[dict]:
    """Convert Citation dataclasses to plain dicts for JSONB storage."""
    return [
        {
            "source_type": str(c.source_type),
            "label": c.label,
            "url": c.url,
            "retrieval_date": c.retrieval_date.isoformat(),
        }
        for c in citations
    ]


# ---------------------------------------------------------------------------
# Workflow discovery
# ---------------------------------------------------------------------------


def _discover_workflow_classes() -> dict[str, type[BaseWorkflow]]:
    """Scan the app.workflows package and return all BaseWorkflow subclasses.

    Excludes base.py and runner.py. Returns a dict keyed by workflow name.
    """
    discovered: dict[str, type[BaseWorkflow]] = {}

    for _, module_name, _ in pkgutil.iter_modules(app.workflows.__path__):
        if module_name in ("base", "runner"):
            continue
        try:
            module = importlib.import_module(f"app.workflows.{module_name}")
        except Exception:
            logger.exception("Failed to import workflow module: %s", module_name)
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseWorkflow)
                and obj is not BaseWorkflow
                and hasattr(obj, "name")
                and hasattr(obj, "description")
            ):
                discovered[obj.name] = obj

    return discovered


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------


def register_workflows(db) -> None:
    """Upsert all discovered BaseWorkflow subclasses into workflow_registry.

    Safe to call on every startup — existing rows are updated in place if
    the description has changed; new rows are inserted.
    """
    classes = _discover_workflow_classes()

    for cls_name, cls in classes.items():
        existing = (
            db.query(WorkflowRegistry).filter(WorkflowRegistry.name == cls_name).first()
        )
        if existing:
            existing.description = cls.description
        else:
            db.add(
                WorkflowRegistry(
                    id=uuid.uuid4(),
                    name=cls_name,
                    description=cls.description,
                )
            )

    db.commit()
    logger.info("Registered %d workflow(s): %s", len(classes), list(classes.keys()))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class WorkflowRunner:
    """Executes a list of workflow classes sequentially for a given thesis.

    Each execution is persisted to workflow_runs. On failure the runner logs
    the error, marks the run as failed, sets context.has_partial_results, and
    continues with the remaining steps.
    """

    def __init__(self, db) -> None:
        self._db = db

    def run(
        self,
        thesis,
        workflow_classes: list[type[BaseWorkflow]],
        pod_settings=None,
    ) -> list[WorkflowResult]:
        """Execute workflow_classes in order, accumulating context.

        Returns the list of WorkflowResult objects (one per class, including
        failed ones so callers can inspect the full run history).
        """
        context = WorkflowContext(thesis=thesis, pod_settings=pod_settings, db=self._db)

        for workflow_cls in workflow_classes:
            workflow = workflow_cls()
            run_id = uuid.uuid4()
            started_at = datetime.now(UTC)

            # Create the run record before execute() so AnthropicClient can
            # log the correct workflow_run_id FK in llm_usage_log.
            run_record = WorkflowRun(
                id=run_id,
                thesis_id=thesis.id,
                workflow_name=workflow.name,
                status=DBWorkflowStatus.failed,  # safe default; updated on success
                started_at=started_at,
            )
            self._db.add(run_record)
            self._db.flush()
            context.current_workflow_run_id = run_id

            try:
                result = workflow.execute(thesis, context)
                run_record.status = DBWorkflowStatus.completed
                run_record.structured_output = result.structured_output
                run_record.citations = _serialize_citations(result.citations)
                run_record.agent_inferences = result.agent_inferences
                run_record.raw_output = result.raw_output
                run_record.completed_at = datetime.now(UTC)
            except Exception as exc:
                logger.exception(
                    "Workflow %s failed for thesis %s", workflow.name, thesis.id
                )
                result = WorkflowResult(
                    workflow_name=workflow.name,
                    status=WorkflowStatus.FAILED,
                    structured_output={},
                    citations=[],
                    agent_inferences=[],
                    raw_output=str(exc),
                )
                run_record.raw_output = str(exc)
                run_record.completed_at = datetime.now(UTC)
                context.has_partial_results = True

            self._db.commit()
            context.current_workflow_run_id = None
            context.prior_results.append(result)

        return context.prior_results
