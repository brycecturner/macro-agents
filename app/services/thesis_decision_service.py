"""ThesisDecisionService — records the human Go/No-Go/Hold decision.

Per PRD Section 4.4, the Tier 1 brief exposes a human decision field with
three options. Go approves the thesis, No-Go rejects it, and Hold defers the
decision without changing status — the thesis remains 'researched' so the
buttons stay available for a later decision.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models.enums import ThesisStatus
from app.models.log import AuditLog
from app.models.thesis import Thesis

_VALID_DECISIONS = frozenset({"go", "no_go", "hold"})

# "hold" maps to None — it defers the decision, leaving thesis.status unchanged.
_STATUS_FOR_DECISION: dict[str, ThesisStatus | None] = {
    "go": ThesisStatus.approved,
    "no_go": ThesisStatus.rejected,
    "hold": None,
}


class InvalidDecisionError(ValueError):
    """Raised when an unrecognised decision value is submitted."""


def _status_value(status: ThesisStatus | str) -> str:
    return status.value if hasattr(status, "value") else str(status)


def record_decision(thesis: Thesis, decision: str, db: Session) -> None:
    """Apply the human decision and write an audit_log entry.

    Caller is responsible for confirming the thesis is eligible for a
    decision (status 'researched') and for committing the session.

    Raises:
        InvalidDecisionError: If decision is not one of go/no_go/hold.
    """
    if decision not in _VALID_DECISIONS:
        raise InvalidDecisionError(f"Unrecognised decision: {decision!r}")

    previous_status = _status_value(thesis.status)
    new_status = _STATUS_FOR_DECISION[decision]
    if new_status is not None:
        thesis.status = new_status

    db.add(
        AuditLog(
            id=uuid.uuid4(),
            pod_id=thesis.pod_id,
            entity_id=thesis.id,
            entity_type="thesis",
            action="thesis_decision",
            previous_value={"status": previous_status},
            new_value={"status": _status_value(thesis.status), "decision": decision},
            changed_by="user",
        )
    )
