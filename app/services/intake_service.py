"""IntakeService — one-volley intake agent for new thesis submissions.

On thesis creation the agent restates the thesis as it understood it and
asks any clarifying questions needed before the full research pipeline runs.
The user has one chance to respond with corrections or answers. After the
configurable timeout the pipeline proceeds with the original interpretation
and thesis_confirmed is set to False, showing a non-dismissible warning
banner until the user explicitly acknowledges it.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.integrations.anthropic_client import AnthropicClient
from app.models.enums import ThesisStatus
from app.models.thesis import Thesis

logger = logging.getLogger(__name__)

_MODEL = "claude-opus-4-6"
_TASK_TYPE = "intake"
_MAX_TOKENS = 1024

_SYSTEM = (
    "You are doing a pre-flight check on a new macro trade thesis submission. "
    "Your job is lightweight: restate the thesis as you understood it and ask "
    "only the questions you genuinely need answered before a full research "
    "pipeline runs. Be brief and direct."
)

_PROMPT = """\
Thesis submitted:
- Title: {title}
- Direction: {direction}
- Time Horizon: {time_horizon}
- Notes:
{notes}

Write a short intake message with these two parts:

## Thesis as I Understood It
In 2–3 sentences, restate: the primary instrument (or your best inference if \
not specified), the direction, the time horizon, and the core macro mechanism. \
Be specific enough that the PM can immediately catch any misinterpretation.

## Questions (if any)
Ask only questions you genuinely need answered before research can run \
effectively — for example, if the instrument is ambiguous between plausible \
alternatives, the time horizon conflicts with the stated mechanism, or a key \
assumption is missing. If the thesis is unambiguous, omit this section entirely.\
"""


class IntakeService:
    """Generates and processes the one-volley intake message for a new thesis."""

    def generate_intake_message(
        self, thesis: Thesis, db: Session, api_key: str
    ) -> None:
        """Call the LLM, persist the intake message, and set status to intake_sent."""
        client = AnthropicClient(api_key=api_key, db=db)
        prompt = _PROMPT.format(
            title=thesis.title,
            direction=thesis.direction.value,
            time_horizon=thesis.time_horizon,
            notes=thesis.notes or "",
        )
        response = client.complete(
            messages=[{"role": "user", "content": prompt}],
            model=_MODEL,
            task_type=_TASK_TYPE,
            thesis_id=thesis.id,
            pod_id=thesis.pod_id,
            system=_SYSTEM,
            max_tokens=_MAX_TOKENS,
        )
        thesis.intake_message = response.content
        thesis.intake_sent_at = datetime.now(UTC)
        thesis.status = ThesisStatus.intake_sent
        db.commit()
        logger.info("Intake message generated for thesis %s", thesis.id)

    def handle_intake_response(
        self, thesis: Thesis, user_response: str, db: Session
    ) -> None:
        """Store the user's response/corrections and queue the research pipeline."""
        stripped = user_response.strip()
        thesis.intake_user_response = stripped if stripped else None
        thesis.intake_responded_at = datetime.now(UTC)
        db.commit()
        logger.info("Intake response recorded for thesis %s", thesis.id)
        _schedule_research(thesis.id)

    @staticmethod
    def process_timeout(thesis: Thesis, db: Session) -> None:
        """Set thesis_confirmed=False and proceed to research anyway."""
        thesis.thesis_confirmed = False
        db.commit()
        logger.warning(
            "Intake timed out for thesis %s — proceeding with assumed interpretation",
            thesis.id,
        )
        _schedule_research(thesis.id)

    @staticmethod
    def acknowledge_intake(thesis: Thesis, db: Session) -> None:
        """Restore thesis_confirmed=True after the user explicitly acknowledges it."""
        thesis.thesis_confirmed = True
        db.commit()

    @staticmethod
    def check_and_process_timeouts(db: Session, intake_timeout_hours: int) -> int:
        """Find intake_sent theses past the timeout window and process each one.

        Returns the number of theses processed.
        """
        cutoff = datetime.now(UTC) - timedelta(hours=intake_timeout_hours)
        timed_out = (
            db.query(Thesis)
            .filter(
                Thesis.status == ThesisStatus.intake_sent,
                Thesis.thesis_confirmed.is_(True),
                Thesis.intake_responded_at.is_(None),
                Thesis.intake_sent_at.isnot(None),
                Thesis.intake_sent_at <= cutoff,
            )
            .all()
        )
        for thesis in timed_out:
            IntakeService.process_timeout(thesis, db)
        return len(timed_out)


def _schedule_research(thesis_id: uuid.UUID) -> None:
    """Queue the full research pipeline for this thesis.

    Stub — TICKET 019 connects this to WorkflowRunner.
    """
    logger.info("Research pipeline queued for thesis %s (TICKET 019)", thesis_id)
