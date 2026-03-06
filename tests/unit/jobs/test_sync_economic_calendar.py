from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

from app.core.constants import FRED_RELEASE_IDS
from app.integrations.fred_client import FREDClientError, FREDReleaseDatesResult
from app.jobs.sync_economic_calendar import sync_economic_calendar
from app.models.monitoring import EconomicCalendar


def _make_fred_result(release_id: int, dates: list[date]) -> FREDReleaseDatesResult:
    return FREDReleaseDatesResult(
        release_id=release_id,
        dates=dates,
        retrieved_at=datetime.now(tz=UTC),
    )


def _make_db(existing: EconomicCalendar | None = None) -> MagicMock:
    """Mock session that returns `existing` from any filter_by().first() call."""
    db = MagicMock()
    db.query.return_value.filter_by.return_value.first.return_value = existing
    return db


def _make_fred_client(dates_by_release_id: dict[int, list[date]]) -> MagicMock:
    client = MagicMock()

    def _get_release_dates(release_id: int) -> FREDReleaseDatesResult:
        dates = dates_by_release_id.get(release_id, [])
        return _make_fred_result(release_id, dates)

    client.get_release_dates.side_effect = _get_release_dates
    return client


# Use a fixed "today" across all tests so window calculations are deterministic
TODAY = date(2025, 6, 15)
PAST_DATE = date(2025, 3, 1)
FUTURE_DATE = date(2025, 9, 1)
OUT_OF_WINDOW_PAST = date(2020, 1, 1)
OUT_OF_WINDOW_FUTURE = date(2030, 1, 1)


class TestSyncEconomicCalendarInsertion:
    def test_inserts_new_past_release(self):
        fred_client = _make_fred_client({FRED_RELEASE_IDS["CPI_RELEASE"]: [PAST_DATE]})
        db = _make_db(existing=None)
        added = []
        db.add.side_effect = added.append

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            sync_economic_calendar(db, fred_client)

        assert len(added) == 1
        row = added[0]
        assert isinstance(row, EconomicCalendar)
        assert row.release_type == "CPI_RELEASE"
        assert row.scheduled_date == PAST_DATE
        assert row.actual_date == PAST_DATE

    def test_inserts_new_future_release_with_null_actual_date(self):
        fred_client = _make_fred_client(
            {FRED_RELEASE_IDS["CPI_RELEASE"]: [FUTURE_DATE]}
        )
        db = _make_db(existing=None)
        added = []
        db.add.side_effect = added.append

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            sync_economic_calendar(db, fred_client)

        assert len(added) == 1
        assert added[0].actual_date is None

    def test_inserted_row_has_uuid_id(self):
        fred_client = _make_fred_client({FRED_RELEASE_IDS["NFP_RELEASE"]: [PAST_DATE]})
        db = _make_db(existing=None)
        added = []
        db.add.side_effect = added.append

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            sync_economic_calendar(db, fred_client)

        import uuid

        assert isinstance(added[0].id, uuid.UUID)

    def test_returns_correct_inserted_count(self):
        fred_client = _make_fred_client(
            {FRED_RELEASE_IDS["CPI_RELEASE"]: [PAST_DATE, FUTURE_DATE]}
        )
        db = _make_db(existing=None)

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            counts = sync_economic_calendar(db, fred_client)

        assert counts["inserted"] == 2
        assert counts["updated"] == 0
        assert counts["skipped"] == 0


class TestSyncEconomicCalendarIdempotency:
    def test_skips_existing_row_with_matching_actual_date(self):
        existing = MagicMock(spec=EconomicCalendar)
        existing.actual_date = PAST_DATE
        fred_client = _make_fred_client({FRED_RELEASE_IDS["CPI_RELEASE"]: [PAST_DATE]})
        db = _make_db(existing=existing)

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            counts = sync_economic_calendar(db, fred_client)

        db.add.assert_not_called()
        assert counts["skipped"] >= 1

    def test_updates_actual_date_when_future_becomes_past(self):
        # Row exists with actual_date=None (was future), now the date has passed
        existing = MagicMock(spec=EconomicCalendar)
        existing.actual_date = None
        fred_client = _make_fred_client({FRED_RELEASE_IDS["CPI_RELEASE"]: [PAST_DATE]})
        db = _make_db(existing=existing)

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            counts = sync_economic_calendar(db, fred_client)

        assert existing.actual_date == PAST_DATE
        db.add.assert_not_called()
        assert counts["updated"] >= 1

    def test_running_twice_does_not_double_insert(self):
        """Second run finds existing rows and skips them."""
        fred_client = _make_fred_client({FRED_RELEASE_IDS["GDP_RELEASE"]: [PAST_DATE]})
        added = []

        # First run: no existing row → insert
        db1 = _make_db(existing=None)
        db1.add.side_effect = added.append

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            sync_economic_calendar(db1, fred_client)

        assert len(added) == 1

        # Second run: existing row found → skip
        existing = MagicMock(spec=EconomicCalendar)
        existing.actual_date = PAST_DATE
        db2 = _make_db(existing=existing)
        db2.add.side_effect = added.append

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            counts = sync_economic_calendar(db2, fred_client)

        assert len(added) == 1  # Nothing new added on second run
        assert counts["skipped"] >= 1


class TestSyncEconomicCalendarWindowFiltering:
    def test_excludes_dates_before_lookback_window(self):
        fred_client = _make_fred_client(
            {FRED_RELEASE_IDS["CPI_RELEASE"]: [OUT_OF_WINDOW_PAST]}
        )
        db = _make_db(existing=None)

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            counts = sync_economic_calendar(db, fred_client)

        db.add.assert_not_called()
        assert counts["inserted"] == 0

    def test_excludes_dates_after_lookahead_window(self):
        fred_client = _make_fred_client(
            {FRED_RELEASE_IDS["CPI_RELEASE"]: [OUT_OF_WINDOW_FUTURE]}
        )
        db = _make_db(existing=None)

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            counts = sync_economic_calendar(db, fred_client)

        db.add.assert_not_called()
        assert counts["inserted"] == 0

    def test_includes_dates_on_window_boundary(self):
        from datetime import timedelta

        from app.core.constants import SYNC_LOOKAHEAD_DAYS, SYNC_LOOKBACK_DAYS

        boundary_past = TODAY - timedelta(days=SYNC_LOOKBACK_DAYS)
        boundary_future = TODAY + timedelta(days=SYNC_LOOKAHEAD_DAYS)

        fred_client = _make_fred_client(
            {FRED_RELEASE_IDS["CPI_RELEASE"]: [boundary_past, boundary_future]}
        )
        db = _make_db(existing=None)
        added = []
        db.add.side_effect = added.append

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            sync_economic_calendar(db, fred_client)

        assert len(added) == 2


class TestSyncEconomicCalendarErrorHandling:
    def test_continues_after_fred_client_error(self):
        """A failure fetching one release type should not abort the others."""
        client = MagicMock()
        cpi_id = FRED_RELEASE_IDS["CPI_RELEASE"]
        nfp_id = FRED_RELEASE_IDS["NFP_RELEASE"]

        def _side_effect(release_id: int) -> FREDReleaseDatesResult:
            if release_id == cpi_id:
                raise FREDClientError("timeout")
            if release_id == nfp_id:
                return _make_fred_result(nfp_id, [PAST_DATE])
            return _make_fred_result(release_id, [])

        client.get_release_dates.side_effect = _side_effect
        db = _make_db(existing=None)
        added = []
        db.add.side_effect = added.append

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            sync_economic_calendar(db, fred_client=client)

        # NFP should still have been processed despite CPI failing
        nfp_rows = [r for r in added if r.release_type == "NFP_RELEASE"]
        assert len(nfp_rows) == 1

    def test_flushes_session_on_completion(self):
        fred_client = _make_fred_client({})
        db = _make_db(existing=None)

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            sync_economic_calendar(db, fred_client)

        db.flush.assert_called_once()


class TestSyncEconomicCalendarCoversTriggerTypes:
    def test_fetches_all_six_trigger_types(self):
        client = MagicMock()
        client.get_release_dates.return_value = _make_fred_result(0, [])
        db = _make_db(existing=None)

        with patch("app.jobs.sync_economic_calendar.date") as mock_date:
            mock_date.today.return_value = TODAY
            mock_date.side_effect = lambda *a, **kw: date(*a, **kw)
            sync_economic_calendar(db, client)

        called_ids = {c.args[0] for c in client.get_release_dates.call_args_list}
        expected_ids = set(FRED_RELEASE_IDS.values())
        assert called_ids == expected_ids
