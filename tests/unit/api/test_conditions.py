"""Tests for falsification condition routes — POST /theses/{id}/conditions,
POST /theses/{id}/conditions/{id}/update, .../delete, and .../test-now."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.models.enums import ConditionType, ThesisStatus
from app.models.thesis import FalsificationCondition, Thesis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_db(
    mock_db: MagicMock,
    *,
    thesis: MagicMock | None = None,
    condition: MagicMock | None = None,
) -> None:
    def _query(model: type) -> MagicMock:
        q = MagicMock()
        if model is Thesis:
            q.filter.return_value.first.return_value = thesis
        elif model is FalsificationCondition:
            q.filter.return_value.first.return_value = condition
        return q

    mock_db.query.side_effect = _query


def _make_thesis(status: ThesisStatus = ThesisStatus.approved) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.status = status
    return t


def _make_condition(
    *, thesis_id: uuid.UUID | None = None, **overrides: object
) -> MagicMock:
    c = MagicMock()
    c.id = overrides.get("id", uuid.uuid4())
    c.thesis_id = thesis_id or uuid.uuid4()
    c.description = overrides.get("description", "Old description")
    c.condition_type = overrides.get("condition_type", ConditionType.state)
    c.trigger_type = overrides.get("trigger_type", None)
    c.measurable_proxy = overrides.get("measurable_proxy", "FRED:OLD")
    c.evaluation_logic = overrides.get("evaluation_logic", "< 1.0")
    return c


_VALID_FORM = {
    "description": "10Y yield falls below 4.0%",
    "condition_type": "state",
    "trigger_type": "",
    "measurable_proxy": "FRED:DGS10",
    "evaluation_logic": "< 4.0",
}


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/conditions
# ---------------------------------------------------------------------------


class TestCreateConditionRoute:
    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.post(f"/theses/{uuid.uuid4()}/conditions", data=_VALID_FORM)
        assert response.status_code == 404

    def test_returns_409_when_thesis_active(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.active)
        _configure_db(mock_db, thesis=thesis)
        response = client.post(f"/theses/{thesis.id}/conditions", data=_VALID_FORM)
        assert response.status_code == 409

    @pytest.mark.parametrize(
        "status",
        [
            ThesisStatus.draft,
            ThesisStatus.intake_sent,
            ThesisStatus.researched,
            ThesisStatus.closed,
            ThesisStatus.rejected,
        ],
    )
    def test_returns_409_for_any_non_approved_status(
        self, status: ThesisStatus, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=status)
        _configure_db(mock_db, thesis=thesis)
        response = client.post(f"/theses/{thesis.id}/conditions", data=_VALID_FORM)
        assert response.status_code == 409

    def test_returns_422_for_empty_description(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        _configure_db(mock_db, thesis=thesis)
        data = {**_VALID_FORM, "description": ""}
        response = client.post(f"/theses/{thesis.id}/conditions", data=data)
        assert response.status_code == 422

    def test_returns_422_for_invalid_condition_type(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        _configure_db(mock_db, thesis=thesis)
        data = {**_VALID_FORM, "condition_type": "sideways"}
        response = client.post(f"/theses/{thesis.id}/conditions", data=data)
        assert response.status_code == 422

    def test_creates_condition_when_approved(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        _configure_db(mock_db, thesis=thesis)
        response = client.post(
            f"/theses/{thesis.id}/conditions",
            data=_VALID_FORM,
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == f"/theses/{thesis.id}"

        saved = mock_db.add.call_args[0][0]
        assert saved.description == "10Y yield falls below 4.0%"
        assert saved.thesis_id == thesis.id
        mock_db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/conditions/{condition_id}/update
# ---------------------------------------------------------------------------


class TestUpdateConditionRoute:
    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.post(
            f"/theses/{uuid.uuid4()}/conditions/{uuid.uuid4()}/update",
            data=_VALID_FORM,
        )
        assert response.status_code == 404

    def test_returns_404_for_missing_condition(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        _configure_db(mock_db, thesis=thesis, condition=None)
        response = client.post(
            f"/theses/{thesis.id}/conditions/{uuid.uuid4()}/update",
            data=_VALID_FORM,
        )
        assert response.status_code == 404

    def test_returns_409_when_thesis_active(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.active)
        condition = _make_condition(thesis_id=thesis.id)
        _configure_db(mock_db, thesis=thesis, condition=condition)
        response = client.post(
            f"/theses/{thesis.id}/conditions/{condition.id}/update",
            data=_VALID_FORM,
        )
        assert response.status_code == 409

    def test_condition_unchanged_when_locked(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.active)
        condition = _make_condition(thesis_id=thesis.id, description="Original")
        _configure_db(mock_db, thesis=thesis, condition=condition)
        client.post(
            f"/theses/{thesis.id}/conditions/{condition.id}/update",
            data={**_VALID_FORM, "description": "Changed"},
        )
        assert condition.description == "Original"

    def test_returns_422_for_empty_description(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        condition = _make_condition(thesis_id=thesis.id)
        _configure_db(mock_db, thesis=thesis, condition=condition)
        data = {**_VALID_FORM, "description": ""}
        response = client.post(
            f"/theses/{thesis.id}/conditions/{condition.id}/update", data=data
        )
        assert response.status_code == 422

    def test_updates_condition_when_approved(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        condition = _make_condition(thesis_id=thesis.id, description="Original")
        _configure_db(mock_db, thesis=thesis, condition=condition)
        data = {**_VALID_FORM, "description": "Updated description"}
        response = client.post(
            f"/theses/{thesis.id}/conditions/{condition.id}/update",
            data=data,
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert condition.description == "Updated description"
        mock_db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/conditions/{condition_id}/delete
# ---------------------------------------------------------------------------


class TestDeleteConditionRoute:
    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.post(
            f"/theses/{uuid.uuid4()}/conditions/{uuid.uuid4()}/delete"
        )
        assert response.status_code == 404

    def test_returns_404_for_missing_condition(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        _configure_db(mock_db, thesis=thesis, condition=None)
        response = client.post(f"/theses/{thesis.id}/conditions/{uuid.uuid4()}/delete")
        assert response.status_code == 404

    def test_returns_409_when_thesis_active(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.active)
        condition = _make_condition(thesis_id=thesis.id)
        _configure_db(mock_db, thesis=thesis, condition=condition)
        response = client.post(f"/theses/{thesis.id}/conditions/{condition.id}/delete")
        assert response.status_code == 409
        mock_db.delete.assert_not_called()

    def test_deletes_when_approved(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        condition = _make_condition(thesis_id=thesis.id)
        _configure_db(mock_db, thesis=thesis, condition=condition)
        response = client.post(
            f"/theses/{thesis.id}/conditions/{condition.id}/delete",
            follow_redirects=False,
        )
        assert response.status_code == 303
        mock_db.delete.assert_called_once_with(condition)
        mock_db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/conditions/{condition_id}/test-now
# ---------------------------------------------------------------------------


class TestTestNowRoute:
    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.post(
            f"/theses/{uuid.uuid4()}/conditions/{uuid.uuid4()}/test-now"
        )
        assert response.status_code == 404

    def test_returns_404_for_missing_condition(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        _configure_db(mock_db, thesis=thesis, condition=None)
        response = client.post(
            f"/theses/{thesis.id}/conditions/{uuid.uuid4()}/test-now"
        )
        assert response.status_code == 404

    def test_returns_200_when_approved(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        condition = _make_condition(thesis_id=thesis.id)
        _configure_db(mock_db, thesis=thesis, condition=condition)
        response = client.post(
            f"/theses/{thesis.id}/conditions/{condition.id}/test-now"
        )
        assert response.status_code == 200

    def test_returns_200_when_active(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        # Test Now is exempt from the edit lock — it must work even when
        # the thesis is active, unlike create/update/delete above.
        thesis = _make_thesis(status=ThesisStatus.active)
        condition = _make_condition(thesis_id=thesis.id)
        _configure_db(mock_db, thesis=thesis, condition=condition)
        response = client.post(
            f"/theses/{thesis.id}/conditions/{condition.id}/test-now"
        )
        assert response.status_code == 200

    def test_response_contains_stub_message(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        condition = _make_condition(thesis_id=thesis.id)
        _configure_db(mock_db, thesis=thesis, condition=condition)
        response = client.post(
            f"/theses/{thesis.id}/conditions/{condition.id}/test-now"
        )
        assert "not yet implemented" in response.text

    def test_never_commits_or_writes(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved)
        condition = _make_condition(thesis_id=thesis.id)
        _configure_db(mock_db, thesis=thesis, condition=condition)
        client.post(f"/theses/{thesis.id}/conditions/{condition.id}/test-now")
        mock_db.commit.assert_not_called()
        mock_db.add.assert_not_called()
        mock_db.delete.assert_not_called()
