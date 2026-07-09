"""Tests for ConditionService — falsification condition edit lock enforcement
and the read-only Test Now evaluation."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from app.models.enums import ConditionType, ThesisStatus
from app.schemas.condition import (
    FalsificationConditionCreate,
    FalsificationConditionUpdate,
)
from app.services.condition_service import (
    ConditionLockedError,
    create_condition,
    delete_condition,
    update_condition,
)
from app.services.condition_service import test_now as run_test_now

_LOCKED_STATUSES = [
    ThesisStatus.draft,
    ThesisStatus.intake_sent,
    ThesisStatus.researched,
    ThesisStatus.active,
    ThesisStatus.closed,
    ThesisStatus.rejected,
]


def _make_thesis(status: ThesisStatus = ThesisStatus.approved) -> MagicMock:
    thesis = MagicMock()
    thesis.id = uuid.uuid4()
    thesis.status = status
    return thesis


def _make_condition(**overrides) -> MagicMock:
    condition = MagicMock()
    condition.id = overrides.get("id", uuid.uuid4())
    condition.thesis_id = overrides.get("thesis_id", uuid.uuid4())
    condition.description = overrides.get("description", "Old description")
    condition.condition_type = overrides.get("condition_type", ConditionType.state)
    condition.trigger_type = overrides.get("trigger_type", None)
    condition.measurable_proxy = overrides.get("measurable_proxy", "FRED:OLD")
    condition.evaluation_logic = overrides.get("evaluation_logic", "< 1.0")
    return condition


def _make_data(**overrides) -> FalsificationConditionCreate:
    fields = {
        "description": "10Y yield falls below 4.0%",
        "condition_type": ConditionType.state,
        "trigger_type": None,
        "measurable_proxy": "FRED:DGS10",
        "evaluation_logic": "< 4.0",
    }
    fields.update(overrides)
    return FalsificationConditionCreate(**fields)


class TestCreateCondition:
    @pytest.mark.parametrize("status", _LOCKED_STATUSES)
    def test_raises_locked_error_when_not_approved(self, status: ThesisStatus) -> None:
        thesis = _make_thesis(status=status)
        db = MagicMock()
        with pytest.raises(ConditionLockedError):
            create_condition(thesis, _make_data(), db)

    def test_does_not_add_to_db_when_locked(self) -> None:
        thesis = _make_thesis(status=ThesisStatus.active)
        db = MagicMock()
        with pytest.raises(ConditionLockedError):
            create_condition(thesis, _make_data(), db)
        db.add.assert_not_called()

    def test_creates_condition_when_approved(self) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        db = MagicMock()
        condition = create_condition(thesis, _make_data(), db)
        db.add.assert_called_once_with(condition)

    def test_condition_fields_set_from_data(self) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        db = MagicMock()
        data = _make_data(
            description="CPI surprises to the upside",
            condition_type=ConditionType.event,
            trigger_type="CPI_RELEASE",
            measurable_proxy="FRED:CPIAUCSL",
            evaluation_logic="> consensus",
        )
        condition = create_condition(thesis, data, db)
        assert condition.description == "CPI surprises to the upside"
        assert condition.condition_type == ConditionType.event
        assert condition.trigger_type == "CPI_RELEASE"
        assert condition.thesis_id == thesis.id

    def test_condition_id_is_uuid(self) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        db = MagicMock()
        condition = create_condition(thesis, _make_data(), db)
        assert isinstance(condition.id, uuid.UUID)


class TestUpdateCondition:
    @pytest.mark.parametrize("status", _LOCKED_STATUSES)
    def test_raises_locked_error_when_not_approved(self, status: ThesisStatus) -> None:
        thesis = _make_thesis(status=status)
        condition = _make_condition()
        update_data = FalsificationConditionUpdate(**_make_data().model_dump())
        with pytest.raises(ConditionLockedError):
            update_condition(thesis, condition, update_data)

    def test_condition_unchanged_when_locked(self) -> None:
        thesis = _make_thesis(status=ThesisStatus.active)
        condition = _make_condition(description="Original")
        update_data = FalsificationConditionUpdate(
            **_make_data(description="Changed").model_dump()
        )
        with pytest.raises(ConditionLockedError):
            update_condition(thesis, condition, update_data)
        assert condition.description == "Original"

    def test_updates_fields_when_approved(self) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        condition = _make_condition(description="Original")
        update_data = FalsificationConditionUpdate(
            **_make_data(description="Changed", evaluation_logic="< 3.5").model_dump()
        )
        updated = update_condition(thesis, condition, update_data)
        assert updated.description == "Changed"
        assert updated.evaluation_logic == "< 3.5"


class TestDeleteCondition:
    @pytest.mark.parametrize("status", _LOCKED_STATUSES)
    def test_raises_locked_error_when_not_approved(self, status: ThesisStatus) -> None:
        thesis = _make_thesis(status=status)
        condition = _make_condition()
        db = MagicMock()
        with pytest.raises(ConditionLockedError):
            delete_condition(thesis, condition, db)
        db.delete.assert_not_called()

    def test_deletes_when_approved(self) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        condition = _make_condition()
        db = MagicMock()
        delete_condition(thesis, condition, db)
        db.delete.assert_called_once_with(condition)


class TestTestNow:
    def test_takes_no_thesis_argument(self) -> None:
        # test_now() is deliberately decoupled from thesis status — it is
        # never subject to the edit lock, so it has nothing to check.
        condition = _make_condition()
        result = run_test_now(condition)
        assert result["status"] == "not_implemented"

    def test_returns_stub_result_shape(self) -> None:
        condition = _make_condition()
        result = run_test_now(condition)
        assert set(result.keys()) == {
            "status",
            "message",
            "data_value",
            "threshold",
            "citation",
        }

    def test_never_touches_condition_object(self) -> None:
        condition = _make_condition(description="Untouched")
        run_test_now(condition)
        assert condition.description == "Untouched"
