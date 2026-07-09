"""IntakeService — one-volley intake agent for new thesis submissions.

On thesis creation the agent restates the thesis as it understood it, asks
any clarifying questions needed before the full research pipeline runs, and
extracts the instrument(s) the thesis trades into thesis_instruments. The
user has one chance to respond with corrections or answers. After the
configurable timeout the pipeline proceeds with the original interpretation
and thesis_confirmed is set to False, showing a non-dismissible warning
banner until the user explicitly acknowledges it.

Instrument extraction (TICKET-018b) piggybacks on the existing one-volley
LLM call rather than a separate call or a dedicated pipeline workflow: the
thesis-level `direction` field is already captured at submission
(TICKET-017), so only the ticker symbol(s), role, and (for non-primary
legs) direction need inference. A thesis may describe more than one
instrument (e.g. a primary position plus an explicit hedge) — extraction is
not hardcoded to a single instrument, matching the thesis_instruments
schema's existing support for multi-leg theses (PRD Section 6.6). Any
ambiguity in the extracted instrument(s) — including a primary instrument
whose inferred direction disagrees with the thesis-level field — is raised
as one of intake's existing clarifying questions rather than silently
resolved.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.integrations.anthropic_client import AnthropicClient
from app.models.enums import Direction, InstrumentRole, ThesisStatus
from app.models.log import AuditLog
from app.models.thesis import Thesis, ThesisInstrument
from app.services.research_pipeline_service import run_research_pipeline_for_thesis

logger = logging.getLogger(__name__)

_MODEL = "claude-opus-4-6"
_TASK_TYPE = "intake"
_MAX_TOKENS = 1024

_SYSTEM = (
    "You are doing a pre-flight check on a new macro trade thesis submission. "
    "Your job is lightweight: restate the thesis as you understood it, ask "
    "only the questions you genuinely need answered before a full research "
    "pipeline runs, and extract the instrument(s) the thesis trades. Be brief "
    "and direct."
)

_PROMPT = """\
Thesis submitted:
- Title: {title}
- Direction: {direction}
- Time Horizon: {time_horizon}
- Notes:
{notes}

Respond with a JSON object with exactly two keys:

- "intake_message": a short markdown message with these two parts:

  ## Thesis as I Understood It
  In 2–3 sentences, restate: the primary instrument (or your best inference \
if not specified), the direction, the time horizon, and the core macro \
mechanism. Be specific enough that the PM can immediately catch any \
misinterpretation.

  ## Questions (if any)
  Ask only questions you genuinely need answered before research can run \
effectively — for example, if an instrument is ambiguous between plausible \
alternatives, the time horizon conflicts with the stated mechanism, a key \
assumption is missing, or an instrument's role or direction (see below) is \
uncertain. If the thesis is unambiguous, omit this section entirely.

- "instruments": a list of one or more objects describing every ETF \
instrument this thesis trades. Most theses describe exactly one, but some \
describe an explicit hedge or secondary leg (e.g. "long TLT, hedged with \
short IEF") — do not assume a hedge or secondary leg shares the primary \
instrument's direction; infer each instrument's direction independently \
from the notes. Each object has:
  - "instrument": the ETF ticker symbol
  - "role": one of "primary", "hedge", "secondary" — exactly one instrument \
must be "primary"
  - "direction": "long" or "short"

  The primary instrument's direction should match the thesis-level \
Direction given above ({direction}). If your inference disagrees, note \
that disagreement as a clarifying question in "intake_message" rather than \
silently picking one.

Respond only with the JSON object. No markdown fences, no preamble.\
"""


def _parse_intake_response(
    content: str, default_direction: Direction
) -> tuple[str, list[dict]]:
    """Parse the JSON intake response into (intake_message, instruments).

    Falls back to treating the raw content as the intake message with no
    extracted instruments if the response isn't valid JSON — a malformed
    response still shows the user *something* rather than nothing, matching
    the graceful-degradation pattern used by the research workflows.

    Each returned instrument dict has "instrument" (str), "role"
    (InstrumentRole), and "direction" (Direction) — invalid/missing role
    defaults to primary; invalid/missing direction defaults to
    default_direction (the thesis-level direction).
    """
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logger.warning("IntakeService: LLM response was not valid JSON")
        return content, []

    intake_message = parsed.get("intake_message") or content
    raw_instruments = parsed.get("instruments") or []

    instruments: list[dict] = []
    for entry in raw_instruments:
        symbol = entry.get("instrument")
        if not symbol:
            continue
        try:
            role = InstrumentRole(entry.get("role", "primary"))
        except ValueError:
            role = InstrumentRole.primary
        try:
            direction = Direction(entry.get("direction"))
        except ValueError:
            direction = default_direction
        instruments.append({"instrument": symbol, "role": role, "direction": direction})

    return intake_message, instruments


class IntakeService:
    """Generates and processes the one-volley intake message for a new thesis."""

    def generate_intake_message(
        self, thesis: Thesis, db: Session, api_key: str
    ) -> None:
        """Call the LLM, persist the intake message and extracted instruments,
        and set status to intake_sent."""
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

        intake_message, instruments = _parse_intake_response(
            response.content, thesis.direction
        )

        thesis.intake_message = intake_message
        thesis.intake_sent_at = datetime.now(UTC)
        thesis.status = ThesisStatus.intake_sent

        for entry in instruments:
            db.add(
                ThesisInstrument(
                    id=uuid.uuid4(),
                    thesis_id=thesis.id,
                    instrument=entry["instrument"],
                    direction=entry["direction"],
                    role=entry["role"],
                )
            )

        db.commit()
        logger.info(
            "Intake message generated for thesis %s (%d instrument(s) extracted)",
            thesis.id,
            len(instruments),
        )

    def handle_intake_response(
        self, thesis: Thesis, user_response: str, db: Session
    ) -> None:
        """Store the user's response/corrections.

        Does not trigger the research pipeline itself — the caller (the
        intake-response route) schedules it as a background task, since the
        pipeline runs multiple sequential LLM calls and should not block the
        HTTP response.
        """
        stripped = user_response.strip()
        thesis.intake_user_response = stripped if stripped else None
        thesis.intake_responded_at = datetime.now(UTC)
        db.commit()
        logger.info("Intake response recorded for thesis %s", thesis.id)

    @staticmethod
    def process_timeout(thesis: Thesis, db: Session) -> None:
        """Set thesis_confirmed=False and proceed to research anyway.

        Runs synchronously — this is called from the hourly intake-timeout
        APScheduler job, which already executes off the request thread.
        """
        thesis.thesis_confirmed = False
        db.commit()
        logger.warning(
            "Intake timed out for thesis %s — proceeding with assumed interpretation",
            thesis.id,
        )
        run_research_pipeline_for_thesis(thesis.id)

    @staticmethod
    def acknowledge_intake(thesis: Thesis, db: Session) -> None:
        """Restore thesis_confirmed=True after the user explicitly acknowledges it.

        Logs the acknowledgment to audit_log — this banner persists until
        explicitly dismissed, so the dismissal itself is a tracked action.
        """
        thesis.thesis_confirmed = True
        db.add(
            AuditLog(
                id=uuid.uuid4(),
                pod_id=thesis.pod_id,
                entity_id=thesis.id,
                entity_type="thesis",
                action="intake_acknowledged",
                previous_value={"thesis_confirmed": False},
                new_value={"thesis_confirmed": True},
                changed_by="user",
            )
        )
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
