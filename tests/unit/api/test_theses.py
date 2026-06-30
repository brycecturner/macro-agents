"""Tests for thesis routes — GET /theses/new, POST /theses, GET /theses/{id}."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.models.enums import Direction, KillAuthority, ThesisStatus, TradingMode
from app.models.pod import Pod, PodConfig
from app.models.thesis import Thesis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_db(
    mock_db: MagicMock,
    *,
    thesis: MagicMock | None = None,
    no_pod: bool = False,
) -> MagicMock:
    """Wire mock_db.query() to return sensible objects per model class."""
    mock_pod = MagicMock()
    mock_pod.id = uuid.uuid4()

    mock_config = MagicMock()
    mock_config.kill_authority_default = KillAuthority.alert_only
    mock_config.trading_mode = TradingMode.paper

    def _query(model: type) -> MagicMock:
        q = MagicMock()
        if model is Pod:
            q.first.return_value = None if no_pod else mock_pod
        elif model is PodConfig:
            q.filter.return_value.first.return_value = mock_config
        elif model is Thesis:
            q.filter.return_value.first.return_value = thesis
        return q

    mock_db.query.side_effect = _query
    return mock_pod


def _make_thesis(
    *,
    title: str = "Yield Curve Steepener",
    direction: Direction = Direction.long,
    time_horizon: str = "6 months",
    notes: str = "Long TLT as curve steepens.",
    status: ThesisStatus = ThesisStatus.draft,
) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.title = title
    t.direction = direction
    t.time_horizon = time_horizon
    t.notes = notes
    t.status = status
    return t


_VALID_FORM = {
    "thesis_title": "Yield Curve Steepener",
    "time_horizon": "6 months",
    "direction": "long",
    "notes": "Long TLT as curve steepens.",
}


# ---------------------------------------------------------------------------
# GET /theses/new
# ---------------------------------------------------------------------------


class TestNewThesisForm:
    @pytest.fixture(autouse=True)
    def _setup_db(self, mock_db: MagicMock) -> None:
        _configure_db(mock_db)

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/theses/new")
        assert response.status_code == 200

    def test_response_is_html(self, client: TestClient) -> None:
        response = client.get("/theses/new")
        assert "text/html" in response.headers["content-type"]

    def test_form_fields_present(self, client: TestClient) -> None:
        response = client.get("/theses/new")
        html = response.text
        assert 'name="thesis_title"' in html
        assert 'name="time_horizon"' in html
        assert 'name="direction"' in html
        assert 'name="notes"' in html

    def test_form_posts_to_theses(self, client: TestClient) -> None:
        response = client.get("/theses/new")
        assert 'action="/theses"' in response.text

    def test_no_errors_on_blank_form(self, client: TestClient) -> None:
        response = client.get("/theses/new")
        assert '<p class="error-msg">' not in response.text

    def test_pod_name_in_nav(self, client: TestClient, mock_db: MagicMock) -> None:
        mock_db.query.side_effect = None
        pod = MagicMock()
        pod.name = "Alpha Pod"
        pod.id = uuid.uuid4()
        pod_config = MagicMock()
        pod_config.trading_mode = TradingMode.paper

        def _query(model: type) -> MagicMock:
            q = MagicMock()
            if model is Pod:
                q.first.return_value = pod
            elif model is PodConfig:
                q.filter.return_value.first.return_value = pod_config
            return q

        mock_db.query.side_effect = _query
        response = client.get("/theses/new")
        assert "Alpha Pod" in response.text

    def test_paper_mode_badge_shown(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        response = client.get("/theses/new")
        assert "PAPER" in response.text

    def test_live_mode_badge_shown_for_real_trading(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        mock_db.query.side_effect = None
        pod = MagicMock()
        pod.name = "Live Pod"
        pod.id = uuid.uuid4()
        pod_config = MagicMock()
        pod_config.trading_mode = TradingMode.real

        def _query(model: type) -> MagicMock:
            q = MagicMock()
            if model is Pod:
                q.first.return_value = pod
            elif model is PodConfig:
                q.filter.return_value.first.return_value = pod_config
            return q

        mock_db.query.side_effect = _query
        response = client.get("/theses/new")
        assert "LIVE TRADING" in response.text


# ---------------------------------------------------------------------------
# POST /theses — validation
# ---------------------------------------------------------------------------


class TestCreateThesisValidation:
    def test_missing_title_returns_422(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        data = {**_VALID_FORM, "thesis_title": ""}
        response = client.post("/theses", data=data)
        assert response.status_code == 422
        assert "Title is required" in response.text

    def test_whitespace_title_returns_422(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        data = {**_VALID_FORM, "thesis_title": "   "}
        response = client.post("/theses", data=data)
        assert response.status_code == 422

    def test_missing_time_horizon_returns_422(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        data = {**_VALID_FORM, "time_horizon": ""}
        response = client.post("/theses", data=data)
        assert response.status_code == 422
        assert "Time horizon is required" in response.text

    def test_missing_direction_returns_422(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        data = {**_VALID_FORM, "direction": ""}
        response = client.post("/theses", data=data)
        assert response.status_code == 422
        assert "Direction is required" in response.text

    def test_invalid_direction_returns_422(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        data = {**_VALID_FORM, "direction": "sideways"}
        response = client.post("/theses", data=data)
        assert response.status_code == 422

    def test_missing_notes_returns_422(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        data = {**_VALID_FORM, "notes": ""}
        response = client.post("/theses", data=data)
        assert response.status_code == 422
        assert "Notes are required" in response.text

    def test_validation_error_re_renders_form(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        data = {**_VALID_FORM, "thesis_title": ""}
        response = client.post("/theses", data=data)
        assert 'name="thesis_title"' in response.text

    def test_submitted_values_preserved_on_error(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        data = {**_VALID_FORM, "thesis_title": "", "notes": "My thesis notes."}
        response = client.post("/theses", data=data)
        assert "My thesis notes." in response.text


# ---------------------------------------------------------------------------
# POST /theses — successful creation
# ---------------------------------------------------------------------------


class TestCreateThesisSuccess:
    def test_valid_submission_redirects(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db)
        response = client.post("/theses", data=_VALID_FORM, follow_redirects=False)
        assert response.status_code == 303

    def test_redirect_location_is_thesis_detail(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db)
        response = client.post("/theses", data=_VALID_FORM, follow_redirects=False)
        assert response.headers["location"].startswith("/theses/")

    def test_thesis_saved_with_draft_status(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db)
        client.post("/theses", data=_VALID_FORM, follow_redirects=False)

        mock_db.add.assert_called_once()
        saved = mock_db.add.call_args[0][0]
        assert saved.status == ThesisStatus.draft

    def test_thesis_pod_id_from_default_pod(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        mock_pod = _configure_db(mock_db)
        client.post("/theses", data=_VALID_FORM, follow_redirects=False)

        saved = mock_db.add.call_args[0][0]
        assert saved.pod_id == mock_pod.id

    def test_thesis_kill_authority_from_pod_config(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db)
        client.post("/theses", data=_VALID_FORM, follow_redirects=False)

        saved = mock_db.add.call_args[0][0]
        assert saved.kill_authority == KillAuthority.alert_only

    def test_thesis_direction_stored_correctly(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db)
        data = {**_VALID_FORM, "direction": "short"}
        client.post("/theses", data=data, follow_redirects=False)

        saved = mock_db.add.call_args[0][0]
        assert saved.direction == Direction.short

    def test_thesis_id_is_uuid(self, client: TestClient, mock_db: MagicMock) -> None:
        _configure_db(mock_db)
        client.post("/theses", data=_VALID_FORM, follow_redirects=False)

        saved = mock_db.add.call_args[0][0]
        assert isinstance(saved.id, uuid.UUID)

    def test_db_committed(self, client: TestClient, mock_db: MagicMock) -> None:
        _configure_db(mock_db)
        client.post("/theses", data=_VALID_FORM, follow_redirects=False)
        mock_db.commit.assert_called_once()

    def test_title_stripped_of_whitespace(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db)
        data = {**_VALID_FORM, "thesis_title": "  Yield Curve  "}
        client.post("/theses", data=data, follow_redirects=False)

        saved = mock_db.add.call_args[0][0]
        assert saved.title == "Yield Curve"


# ---------------------------------------------------------------------------
# GET /theses/{thesis_id}
# ---------------------------------------------------------------------------


class TestThesisDetail:
    def test_returns_200_for_existing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis()
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert response.status_code == 200

    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.get(f"/theses/{uuid.uuid4()}")
        assert response.status_code == 404

    def test_displays_thesis_title(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(title="Inflation Breakout")
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Inflation Breakout" in response.text

    def test_displays_thesis_status(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.draft)
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "draft" in response.text

    def test_displays_direction(self, client: TestClient, mock_db: MagicMock) -> None:
        thesis = _make_thesis(direction=Direction.short)
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Short" in response.text

    def test_displays_time_horizon(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(time_horizon="9 months")
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "9 months" in response.text

    def test_invalid_uuid_returns_422(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        response = client.get("/theses/not-a-uuid")
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "status",
        [
            ThesisStatus.draft,
            ThesisStatus.intake_sent,
            ThesisStatus.researched,
            ThesisStatus.approved,
            ThesisStatus.active,
            ThesisStatus.closed,
            ThesisStatus.rejected,
        ],
    )
    def test_status_badge_rendered_for_all_states(
        self,
        status: ThesisStatus,
        client: TestClient,
        mock_db: MagicMock,
    ) -> None:
        thesis = _make_thesis(status=status)
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert f"badge-{status.value}" in response.text
