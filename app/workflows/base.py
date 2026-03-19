"""BaseWorkflow, WorkflowResult, Citation, and WorkflowContext.

All research workflows in this system subclass BaseWorkflow and return
a WorkflowResult. WorkflowContext is passed into every execute() call so
workflows can consume prior results without reaching into global state.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from app.core.pod_settings import PodSettings
    from app.models.thesis import Thesis


class WorkflowStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class CitationSourceType(StrEnum):
    FRED = "FRED"
    IBKR = "IBKR"
    WEB = "web"
    OECD = "OECD"
    AGENT_INFERENCE = "agent_inference"


@dataclass
class Citation:
    """A traceable source reference attached to a workflow output.

    Label format by source type (from PRD citation table):
      FRED  — "FRED:{series_id}, retrieved {date}"
      IBKR  — "IBKR:{instrument} {data_type}, {timestamp}"
      WEB   — full URL string
      AGENT — "[Agent inference]"
    """

    source_type: CitationSourceType
    label: str
    url: str | None
    retrieval_date: date


@dataclass
class WorkflowResult:
    """Structured output from a single workflow execution.

    structured_output is typed per workflow — each workflow defines the
    expected keys. Only structured_output is passed forward to subsequent
    workflows; raw_output is stored for debugging only.
    """

    workflow_name: str
    status: WorkflowStatus
    structured_output: dict
    citations: list[Citation]
    agent_inferences: list[str]
    raw_output: str


@dataclass
class WorkflowContext:
    """Execution context passed to every workflow's execute() method.

    prior_results accumulates WorkflowResult objects as the sequential
    research chain progresses. Workflows downstream in the chain can
    consume earlier results via this list.
    """

    thesis: Thesis
    prior_results: list[WorkflowResult] = field(default_factory=list)
    pod_settings: PodSettings | None = None
    # Set to True when any workflow in the chain fails; subsequent workflows
    # can inspect this to adjust their behaviour.
    has_partial_results: bool = False
    # SQLAlchemy Session — threaded through by WorkflowRunner so workflows
    # can pass it to AnthropicClient for llm_usage_log writes. Any is
    # unavoidable: importing Session would couple base.py to SQLAlchemy.
    db: Session | None = None
    # Set by WorkflowRunner before each execute() call so workflows can pass
    # the correct FK when logging Anthropic API calls to llm_usage_log.
    current_workflow_run_id: uuid.UUID | None = None

    def get_result(self, workflow_name: str) -> WorkflowResult | None:
        """Return the most recent result for a named workflow, or None."""
        for result in reversed(self.prior_results):
            if result.workflow_name == workflow_name:
                return result
        return None


class BaseWorkflow(ABC):
    """Abstract base class for all research workflows.

    Subclasses must define name, description, required_inputs, and
    implement execute(). Attempting to instantiate this class directly
    raises NotImplementedError via the abstract method enforcement.
    """

    name: str
    description: str
    required_inputs: list[str]

    @abstractmethod
    def execute(self, thesis: Thesis, context: WorkflowContext) -> WorkflowResult:
        """Run the workflow and return a structured result.

        Args:
            thesis: The thesis object this workflow is researching.
            context: Accumulated prior results and pod settings.

        Returns:
            A WorkflowResult with status, structured_output, citations,
            agent_inferences, and raw_output populated.
        """
        ...
