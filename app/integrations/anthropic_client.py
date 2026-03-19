"""AnthropicClient — the single gateway for all Anthropic API calls.

Every LLM call in this system goes through this client. No workflow, service,
or route may call the Anthropic SDK directly. This client:
  - Makes the API call
  - Extracts token usage from the response
  - Computes estimated cost from MODEL_PRICING constants
  - Writes a row to llm_usage_log (even on failure, with zero counts)
  - Returns an AnthropicResponse dataclass
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import anthropic

from app.core.constants import MODEL_PRICING
from app.models.log import LLMUsageLog

logger = logging.getLogger(__name__)


class AnthropicClientError(Exception):
    """Raised when the Anthropic API call fails."""


@dataclass
class AnthropicResponse:
    """Structured response from a completed Anthropic API call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute estimated USD cost from MODEL_PRICING constants.

    Uses the per-1k-token rates defined in app/core/constants.py.
    Falls back to 0.0 for unrecognised models rather than raising.
    """
    pricing = MODEL_PRICING.get(model, {"input_per_1k": 0.0, "output_per_1k": 0.0})
    return (input_tokens / 1000.0) * pricing["input_per_1k"] + (
        output_tokens / 1000.0
    ) * pricing["output_per_1k"]


def _extract_text(content: list) -> str:
    """Return the concatenated text from all text blocks in an API response."""
    return "".join(
        block.text for block in content if getattr(block, "type", None) == "text"
    )


class AnthropicClient:
    """Gateway for all Anthropic API calls with mandatory usage logging.

    Instantiate once per workflow run (or per request for non-workflow uses).
    Every call to :meth:`complete` writes a row to ``llm_usage_log`` — even
    on failure, so cost tracking is always complete.
    """

    def __init__(self, api_key: str, db) -> None:
        if not api_key:
            raise AnthropicClientError(
                "ANTHROPIC_API_KEY is not configured. "
                "Set it in .env before making LLM calls."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._db = db

    def complete(
        self,
        messages: list[dict],
        model: str,
        task_type: str,
        workflow_run_id: uuid.UUID | None = None,
        thesis_id: uuid.UUID | None = None,
        pod_id: uuid.UUID | None = None,
        **kwargs,
    ) -> AnthropicResponse:
        """Make an Anthropic messages API call and log usage to llm_usage_log.

        Args:
            messages: List of message dicts (role + content) for the API call.
            model: Exact model string, e.g. ``"claude-sonnet-4-6"``.
            task_type: Human-readable label, e.g. ``"macro_context"``.
            workflow_run_id: FK to workflow_runs — nullable for non-workflow calls.
            thesis_id: FK to theses — nullable where not applicable.
            pod_id: FK to pods — nullable; pass when known at the call site.
            **kwargs: Passed through to ``messages.create()`` (e.g. ``max_tokens``,
                ``system``, ``tools``).

        Returns:
            :class:`AnthropicResponse` with extracted text content and usage stats.

        Raises:
            AnthropicClientError: If the API call fails. Usage is still logged
                with zero token counts before the error is re-raised.
        """
        try:
            response = self._client.messages.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            content = _extract_text(response.content)
            stop_reason = getattr(response, "stop_reason", None)

        except Exception as exc:
            self._log_usage(
                model=model,
                task_type=task_type,
                input_tokens=0,
                output_tokens=0,
                workflow_run_id=workflow_run_id,
                thesis_id=thesis_id,
                pod_id=pod_id,
            )
            raise AnthropicClientError(
                f"Anthropic API call failed (task_type={task_type!r}, "
                f"model={model!r}): {type(exc).__name__}: {exc}"
            ) from exc

        self._log_usage(
            model=model,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            workflow_run_id=workflow_run_id,
            thesis_id=thesis_id,
            pod_id=pod_id,
        )

        logger.debug(
            "Anthropic call complete: task_type=%r model=%r "
            "input_tokens=%d output_tokens=%d",
            task_type,
            model,
            input_tokens,
            output_tokens,
        )

        return AnthropicResponse(
            content=content,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_usage(
        self,
        *,
        model: str,
        task_type: str,
        input_tokens: int,
        output_tokens: int,
        workflow_run_id: uuid.UUID | None,
        thesis_id: uuid.UUID | None,
        pod_id: uuid.UUID | None,
    ) -> None:
        """Write one row to llm_usage_log. Never raises — log and continue."""
        try:
            cost = _compute_cost(model, input_tokens, output_tokens)
            log_row = LLMUsageLog(
                id=uuid.uuid4(),
                pod_id=pod_id,
                thesis_id=thesis_id,
                workflow_run_id=workflow_run_id,
                model=model,
                task_type=task_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                called_at=datetime.now(UTC),
            )
            self._db.add(log_row)
            self._db.commit()
        except Exception:
            logger.exception(
                "Failed to write llm_usage_log for task_type=%r", task_type
            )
