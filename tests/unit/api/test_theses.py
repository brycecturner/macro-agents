"""Tests for thesis routes — GET /theses/new, POST /theses, GET /theses/{id},
POST /theses/{id}/intake-response, POST /theses/{id}/acknowledge-intake."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.models.enums import Direction, KillAuthority, ThesisStatus, TradingMode
from app.models.log import AuditLog
from app.models.pod import Pod, PodConfig
from app.models.thesis import Thesis
from app.models.workflow import FurtherReading

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_db(
    mock_db: MagicMock,
    *,
    thesis: MagicMock | None = None,
    no_pod: bool = False,
    further_reading: list | None = None,
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
        elif model is FurtherReading:
            q.filter.return_value.order_by.return_value.all.return_value = (
                further_reading or []
            )
        return q

    mock_db.query.side_effect = _query
    return mock_pod


def _make_further_reading(
    *,
    title: str = "Fed Signals Patience on Rate Cuts",
    url: str = "https://www.federalreserve.gov/newsevents/example",
    source_type: str = "web",
    annotation: str = "Directly informs the thesis mechanism around Fed policy.",
    rank: int = 1,
    is_cited: bool = True,
) -> MagicMock:
    entry = MagicMock()
    entry.id = uuid.uuid4()
    entry.title = title
    entry.url = url
    entry.source_type = source_type
    entry.annotation = annotation
    entry.rank = rank
    entry.is_cited = is_cited
    return entry


def _make_thesis(
    *,
    title: str = "Yield Curve Steepener",
    direction: Direction = Direction.long,
    time_horizon: str = "6 months",
    notes: str = "Long TLT as curve steepens.",
    status: ThesisStatus = ThesisStatus.draft,
    thesis_confirmed: bool = True,
    intake_message: str | None = None,
    intake_user_response: str | None = None,
    brief: dict | None = None,
) -> MagicMock:
    t = MagicMock()
    t.id = uuid.uuid4()
    t.pod_id = uuid.uuid4()
    t.title = title
    t.direction = direction
    t.time_horizon = time_horizon
    t.notes = notes
    t.status = status
    t.thesis_confirmed = thesis_confirmed
    t.intake_message = intake_message
    t.intake_user_response = intake_user_response
    t.brief = brief
    return t


_VALID_FORM = {
    "thesis_title": "Yield Curve Steepener",
    "time_horizon": "6 months",
    "direction": "long",
    "notes": "Long TLT as curve steepens.",
}


def _make_brief(**overrides) -> dict:
    brief = {
        "thesis_id": str(uuid.uuid4()),
        "title": "Yield Curve Steepener",
        "summary": "The macro backdrop favors curve steepening.",
        "instrument": "TLT",
        "direction": "long",
        "time_horizon": "6 months",
        "backtest_stats": {
            "label": "Historical Analog Analysis",
            "n_periods": 3,
            "avg_return": 0.05,
            "worst_return": -0.02,
            "best_return": 0.12,
            "win_rate": 0.67,
            "avg_max_drawdown": -0.04,
            "statistical_limitation_note": "Small sample size.",
            "benchmark_comparison": {
                "spy": {"avg_return": 0.03},
                "60_40": {"avg_return": 0.02},
            },
            "analysis": "The instrument rallied in prior analogs.",
        },
        "assumptions": ["Fed cuts continue through year-end"],
        "falsification_conditions": [
            {
                "id": str(uuid.uuid4()),
                "description": "10Y yield falls below 4.0%",
                "condition_type": "state",
                "trigger_type": None,
                "measurable_proxy": "FRED:DGS10",
                "evaluation_logic": "< 4.0",
            },
            {
                "id": str(uuid.uuid4()),
                "description": "CPI surprises to the upside",
                "condition_type": "event",
                "trigger_type": "CPI_RELEASE",
                "measurable_proxy": "FRED:CPIAUCSL",
                "evaluation_logic": "> consensus",
            },
        ],
        "recommendation": {
            "recommendation": "go",
            "rationale": "Evidence is supportive across workflows.",
            "confidence_level": "high",
        },
        "source_index": [
            {
                "source_type": "FRED",
                "label": "FRED:T10Y2Y, retrieved 2026-07-01",
                "url": None,
                "retrieval_date": "2026-07-01",
            },
            {
                "source_type": "web",
                "label": "https://www.federalreserve.gov/example",
                "url": "https://www.federalreserve.gov/example",
                "retrieval_date": "2026-07-01",
            },
        ],
    }
    brief.update(overrides)
    return brief


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
    @pytest.fixture(autouse=True)
    def _no_intake_bg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.api.theses._run_intake_generation", lambda _: None)

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

    def test_thesis_confirmed_true_on_creation(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db)
        client.post("/theses", data=_VALID_FORM, follow_redirects=False)

        saved = mock_db.add.call_args[0][0]
        assert saved.thesis_confirmed is True

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

    def test_intake_message_displayed_when_present(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(
            status=ThesisStatus.intake_sent,
            intake_message="## Thesis as I Understood It\nLong TLT.",
        )
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Long TLT." in response.text

    def test_response_form_shown_when_awaiting_intake(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(
            status=ThesisStatus.intake_sent,
            intake_message="Some message.",
            intake_user_response=None,
        )
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert 'name="user_response"' in response.text

    def test_response_form_hidden_after_user_responds(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(
            status=ThesisStatus.intake_sent,
            intake_message="Some message.",
            intake_user_response="Confirmed.",
        )
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert 'name="user_response"' not in response.text
        assert "Confirmed." in response.text

    def test_loading_state_shown_when_intake_pending(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.intake_sent, intake_message=None)
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Generating intake review" in response.text

    def test_unconfirmed_banner_shown_when_thesis_not_confirmed(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(thesis_confirmed=False)
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Intake not confirmed" in response.text

    def test_unconfirmed_banner_hidden_when_confirmed(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(thesis_confirmed=True)
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Intake not confirmed" not in response.text

    def test_acknowledge_button_present_when_not_confirmed(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(thesis_confirmed=False)
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "acknowledge-intake" in response.text

    def test_no_brief_section_when_brief_absent(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.intake_sent, brief=None)
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Historical Analog Analysis" not in response.text

    def test_brief_summary_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(
            status=ThesisStatus.researched,
            brief=_make_brief(summary="The macro backdrop favors curve steepening."),
        )
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "The macro backdrop favors curve steepening." in response.text

    def test_backtest_stats_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Historical Analog Analysis" in response.text
        assert "5.0%" in response.text  # avg_return 0.05

    def test_assumptions_rendered(self, client: TestClient, mock_db: MagicMock) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Fed cuts continue through year-end" in response.text

    def test_falsification_conditions_rendered_with_type_label(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "10Y yield falls below 4.0%" in response.text
        assert "state" in response.text
        assert "CPI surprises to the upside" in response.text
        assert "event" in response.text
        assert "CPI_RELEASE" in response.text

    def test_recommendation_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "Evidence is supportive across workflows." in response.text
        assert "high" in response.text

    def test_source_index_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert "FRED:T10Y2Y, retrieved 2026-07-01" in response.text
        assert "https://www.federalreserve.gov/example" in response.text

    def test_decision_buttons_shown_when_researched(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert f"/theses/{thesis.id}/decision" in response.text
        assert 'value="go"' in response.text
        assert 'value="no_go"' in response.text
        assert 'value="hold"' in response.text

    def test_decision_buttons_hidden_when_approved(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}")
        assert f"/theses/{thesis.id}/decision" not in response.text


# ---------------------------------------------------------------------------
# GET /theses/{thesis_id} — Further Reading (Tier 3)
# ---------------------------------------------------------------------------


class TestFurtherReadingSection:
    def test_no_section_when_no_entries(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis, further_reading=[])
        response = client.get(f"/theses/{thesis.id}")
        assert "Further Reading" not in response.text

    def test_entry_fields_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        entry = _make_further_reading(
            title="Fed Signals Patience on Rate Cuts",
            url="https://www.federalreserve.gov/newsevents/example",
            source_type="web",
            annotation="Directly informs the thesis mechanism around Fed policy.",
        )
        _configure_db(mock_db, thesis=thesis, further_reading=[entry])
        response = client.get(f"/theses/{thesis.id}")
        assert "Fed Signals Patience on Rate Cuts" in response.text
        assert 'href="https://www.federalreserve.gov/newsevents/example"' in (
            response.text
        )
        assert "Directly informs the thesis mechanism around Fed policy." in (
            response.text
        )
        assert "Further Reading" in response.text

    def test_source_type_badge_rendered(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        entry = _make_further_reading(source_type="academic paper")
        _configure_db(mock_db, thesis=thesis, further_reading=[entry])
        response = client.get(f"/theses/{thesis.id}")
        assert "academic paper" in response.text

    def test_entries_rendered_in_rank_order(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        cited = _make_further_reading(
            title="Cited Primary Source", rank=1, is_cited=True
        )
        additional = _make_further_reading(
            title="Additional Context Source", rank=2, is_cited=False
        )
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis, further_reading=[cited, additional])
        response = client.get(f"/theses/{thesis.id}")
        assert response.text.index("Cited Primary Source") < response.text.index(
            "Additional Context Source"
        )

    def test_section_rendered_even_without_brief(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.intake_sent, brief=None)
        entry = _make_further_reading()
        _configure_db(mock_db, thesis=thesis, further_reading=[entry])
        response = client.get(f"/theses/{thesis.id}")
        assert "Further Reading" in response.text


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/intake-response
# ---------------------------------------------------------------------------


class TestIntakeResponse:
    @pytest.fixture(autouse=True)
    def mock_research_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        mock = MagicMock()
        monkeypatch.setattr("app.api.theses.run_research_pipeline_for_thesis", mock)
        return mock

    def test_redirects_to_detail_on_success(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.intake_sent)
        _configure_db(mock_db, thesis=thesis)
        response = client.post(
            f"/theses/{thesis.id}/intake-response",
            data={"user_response": "Confirmed."},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == f"/theses/{thesis.id}"

    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.post(
            f"/theses/{uuid.uuid4()}/intake-response",
            data={"user_response": "OK"},
        )
        assert response.status_code == 404

    def test_returns_409_when_not_intake_sent(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.draft)
        _configure_db(mock_db, thesis=thesis)
        response = client.post(
            f"/theses/{thesis.id}/intake-response",
            data={"user_response": "OK"},
        )
        assert response.status_code == 409

    def test_schedules_research_pipeline_background_task(
        self,
        client: TestClient,
        mock_db: MagicMock,
        mock_research_pipeline: MagicMock,
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.intake_sent)
        _configure_db(mock_db, thesis=thesis)
        client.post(
            f"/theses/{thesis.id}/intake-response",
            data={"user_response": "Confirmed."},
            follow_redirects=False,
        )
        mock_research_pipeline.assert_called_once_with(thesis.id)

    def test_does_not_schedule_pipeline_when_thesis_not_found(
        self,
        client: TestClient,
        mock_db: MagicMock,
        mock_research_pipeline: MagicMock,
    ) -> None:
        _configure_db(mock_db, thesis=None)
        client.post(
            f"/theses/{uuid.uuid4()}/intake-response",
            data={"user_response": "OK"},
        )
        mock_research_pipeline.assert_not_called()


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/acknowledge-intake
# ---------------------------------------------------------------------------


class TestAcknowledgeIntake:
    def test_redirects_to_detail_on_success(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(thesis_confirmed=False)
        _configure_db(mock_db, thesis=thesis)
        response = client.post(
            f"/theses/{thesis.id}/acknowledge-intake",
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == f"/theses/{thesis.id}"

    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.post(f"/theses/{uuid.uuid4()}/acknowledge-intake")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /theses/{thesis_id}/brief
# ---------------------------------------------------------------------------


class TestGetThesisBrief:
    def test_returns_200_when_brief_exists(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched)
        thesis.brief = {"thesis_id": str(thesis.id), "title": thesis.title}
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}/brief")
        assert response.status_code == 200

    def test_response_is_json(self, client: TestClient, mock_db: MagicMock) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched)
        thesis.brief = {"thesis_id": str(thesis.id)}
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}/brief")
        assert "application/json" in response.headers["content-type"]

    def test_returns_stored_brief_body(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched)
        thesis.brief = {
            "thesis_id": str(thesis.id),
            "summary": "Macro backdrop.",
            "instrument": "TLT",
            "direction": "long",
            "time_horizon": "6 months",
            "backtest_stats": {"label": "Historical Analog Analysis"},
            "assumptions": ["Fed cuts continue"],
            "falsification_conditions": [],
            "recommendation": {"recommendation": "go"},
            "source_index": [],
        }
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}/brief")
        assert response.json() == thesis.brief

    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.get(f"/theses/{uuid.uuid4()}/brief")
        assert response.status_code == 404

    def test_returns_404_when_brief_not_yet_generated(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.intake_sent)
        thesis.brief = None
        _configure_db(mock_db, thesis=thesis)
        response = client.get(f"/theses/{thesis.id}/brief")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/decision
# ---------------------------------------------------------------------------


class TestSubmitThesisDecision:
    def test_go_redirects_to_detail(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.post(
            f"/theses/{thesis.id}/decision",
            data={"decision": "go"},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert response.headers["location"] == f"/theses/{thesis.id}"

    def test_go_sets_status_approved(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        client.post(
            f"/theses/{thesis.id}/decision",
            data={"decision": "go"},
            follow_redirects=False,
        )
        assert thesis.status == ThesisStatus.approved

    def test_no_go_sets_status_rejected(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        client.post(
            f"/theses/{thesis.id}/decision",
            data={"decision": "no_go"},
            follow_redirects=False,
        )
        assert thesis.status == ThesisStatus.rejected

    def test_hold_leaves_status_researched(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        client.post(
            f"/theses/{thesis.id}/decision",
            data={"decision": "hold"},
            follow_redirects=False,
        )
        assert thesis.status == ThesisStatus.researched

    def test_writes_audit_log_entry(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        added = []
        mock_db.add.side_effect = added.append
        client.post(
            f"/theses/{thesis.id}/decision",
            data={"decision": "go"},
            follow_redirects=False,
        )

        audit_rows = [a for a in added if isinstance(a, AuditLog)]
        assert len(audit_rows) == 1
        assert audit_rows[0].action == "thesis_decision"

    def test_commits_db(self, client: TestClient, mock_db: MagicMock) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        client.post(
            f"/theses/{thesis.id}/decision",
            data={"decision": "go"},
            follow_redirects=False,
        )
        mock_db.commit.assert_called_once()

    def test_returns_404_for_missing_thesis(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        _configure_db(mock_db, thesis=None)
        response = client.post(
            f"/theses/{uuid.uuid4()}/decision",
            data={"decision": "go"},
        )
        assert response.status_code == 404

    def test_returns_409_when_not_researched(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.draft)
        _configure_db(mock_db, thesis=thesis)
        response = client.post(
            f"/theses/{thesis.id}/decision",
            data={"decision": "go"},
        )
        assert response.status_code == 409

    def test_returns_409_when_already_approved(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.approved, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.post(
            f"/theses/{thesis.id}/decision",
            data={"decision": "go"},
        )
        assert response.status_code == 409

    def test_returns_422_for_invalid_decision(
        self, client: TestClient, mock_db: MagicMock
    ) -> None:
        thesis = _make_thesis(status=ThesisStatus.researched, brief=_make_brief())
        _configure_db(mock_db, thesis=thesis)
        response = client.post(
            f"/theses/{thesis.id}/decision",
            data={"decision": "maybe"},
        )
        assert response.status_code == 422
