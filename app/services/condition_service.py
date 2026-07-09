"""ConditionService — enforces the falsification condition edit lock and
provides an on-demand, read-only Test Now evaluation.

Per PRD Section 5.7, falsification conditions are fully editable only while
a thesis is 'approved' — the window between research completing and the
thesis becoming active. Once a thesis is 'active' (or any other status),
create/update/delete are rejected with a typed ConditionLockedError. The
only path to editing a condition on an active thesis is close-and-reopen —
closing unlocks the conditions again.

Test Now is exempt from the lock: it is a read-only, on-demand evaluation
available at any thesis status, including 'active'.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.models.enums import ThesisStatus
from app.models.thesis import FalsificationCondition, Thesis
from app.schemas.condition import (
    FalsificationConditionCreate,
    FalsificationConditionUpdate,
)

# STUB: the real evaluator ships with the State Condition Evaluator and Event
# Condition Evaluator tickets. This return value keeps the Test Now button,
# HTMX endpoint, and read-only guarantee in place ahead of that work landing.
# See FUTURE_IMPROVEMENTS.md.
_TEST_NOW_NOT_IMPLEMENTED_MESSAGE = (
    "Condition evaluation is not yet implemented — this ships with the "
    "State and Event Condition Evaluator tickets. This button is wired and "
    "will run the real evaluator once available."
)


class ConditionLockedError(Exception):
    """Raised when a condition create/update/delete is attempted while the
    thesis is not 'approved'. Conditions are only editable in that window;
    close-and-reopen is the only way to unlock an active thesis's conditions.
    """


def _ensure_editable(thesis: Thesis) -> None:
    if thesis.status != ThesisStatus.approved:
        raise ConditionLockedError(
            f"Falsification conditions cannot be edited while thesis status "
            f"is '{thesis.status.value}'. Conditions are only editable when "
            f"the thesis is 'approved'."
        )


def create_condition(
    thesis: Thesis,
    data: FalsificationConditionCreate,
    db: Session,
) -> FalsificationCondition:
    """Create a new falsification condition. Caller commits.

    Raises:
        ConditionLockedError: If thesis.status is not 'approved'.
    """
    _ensure_editable(thesis)
    condition = FalsificationCondition(
        id=uuid.uuid4(),
        thesis_id=thesis.id,
        description=data.description,
        condition_type=data.condition_type,
        trigger_type=data.trigger_type,
        measurable_proxy=data.measurable_proxy,
        evaluation_logic=data.evaluation_logic,
    )
    db.add(condition)
    return condition


def update_condition(
    thesis: Thesis,
    condition: FalsificationCondition,
    data: FalsificationConditionUpdate,
) -> FalsificationCondition:
    """Replace an existing condition's fields in place. Caller commits.

    Raises:
        ConditionLockedError: If thesis.status is not 'approved'.
    """
    _ensure_editable(thesis)
    condition.description = data.description
    condition.condition_type = data.condition_type
    condition.trigger_type = data.trigger_type
    condition.measurable_proxy = data.measurable_proxy
    condition.evaluation_logic = data.evaluation_logic
    return condition


def delete_condition(
    thesis: Thesis, condition: FalsificationCondition, db: Session
) -> None:
    """Delete a condition. Caller commits.

    Raises:
        ConditionLockedError: If thesis.status is not 'approved'.
    """
    _ensure_editable(thesis)
    db.delete(condition)


def test_now(condition: FalsificationCondition) -> dict:
    """Run an on-demand, read-only evaluation of a single condition.

    Never modifies condition or thesis state, and is available regardless
    of thesis status, including 'active' (PRD Section 5.7).
    """
    return {
        "status": "not_implemented",
        "message": _TEST_NOW_NOT_IMPLEMENTED_MESSAGE,
        "data_value": None,
        "threshold": None,
        "citation": None,
    }
