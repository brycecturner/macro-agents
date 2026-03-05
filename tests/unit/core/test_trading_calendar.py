from datetime import date

import pytest

from app.core.trading_calendar import TradingCalendar


class TestTradingCalendar:
    """Tests for TradingCalendar date resolution against NYSE rules.

    Each test uses a concrete, verified date so failures are immediately
    debuggable. Holiday dates are cross-referenced against NYSE official
    calendar.
    """

    @pytest.fixture
    def cal(self) -> TradingCalendar:
        return TradingCalendar()

    # ── Weekend resolution ────────────────────────────────────────────────

    def test_saturday_returns_preceding_friday(self, cal: TradingCalendar) -> None:
        # 2024-03-02 is a Saturday
        assert cal.most_recent_trading_day(date(2024, 3, 2)) == date(2024, 3, 1)

    def test_sunday_returns_preceding_friday(self, cal: TradingCalendar) -> None:
        # 2024-03-03 is a Sunday
        assert cal.most_recent_trading_day(date(2024, 3, 3)) == date(2024, 3, 1)

    def test_weekday_non_holiday_returns_self(self, cal: TradingCalendar) -> None:
        # 2024-03-04 is a regular Monday
        assert cal.most_recent_trading_day(date(2024, 3, 4)) == date(2024, 3, 4)

    # ── MLK Day ───────────────────────────────────────────────────────────

    def test_mlk_day_2025_skipped(self, cal: TradingCalendar) -> None:
        # MLK Day 2025: January 20 (3rd Monday)
        assert not cal.is_trading_day(date(2025, 1, 20))
        assert cal.most_recent_trading_day(date(2025, 1, 20)) == date(2025, 1, 17)

    # ── Presidents' Day ───────────────────────────────────────────────────

    def test_presidents_day_2025_skipped(self, cal: TradingCalendar) -> None:
        # Presidents' Day 2025: February 17 (3rd Monday)
        assert not cal.is_trading_day(date(2025, 2, 17))
        assert cal.most_recent_trading_day(date(2025, 2, 17)) == date(2025, 2, 14)

    # ── Good Friday ───────────────────────────────────────────────────────

    def test_good_friday_2025_skipped(self, cal: TradingCalendar) -> None:
        # Easter 2025: April 20; Good Friday: April 18
        assert not cal.is_trading_day(date(2025, 4, 18))
        assert cal.most_recent_trading_day(date(2025, 4, 18)) == date(2025, 4, 17)

    def test_good_friday_2024_skipped(self, cal: TradingCalendar) -> None:
        # Easter 2024: March 31; Good Friday: March 29
        assert not cal.is_trading_day(date(2024, 3, 29))

    # ── Memorial Day ──────────────────────────────────────────────────────

    def test_memorial_day_2025_skipped(self, cal: TradingCalendar) -> None:
        # Memorial Day 2025: May 26 (last Monday in May)
        assert not cal.is_trading_day(date(2025, 5, 26))
        assert cal.most_recent_trading_day(date(2025, 5, 26)) == date(2025, 5, 23)

    # ── Juneteenth ────────────────────────────────────────────────────────

    def test_juneteenth_2024_skipped(self, cal: TradingCalendar) -> None:
        # Juneteenth 2024: June 19 is a Wednesday
        assert not cal.is_trading_day(date(2024, 6, 19))

    def test_juneteenth_observed_2022_sunday_to_monday(
        self, cal: TradingCalendar
    ) -> None:
        # June 19, 2022 falls on a Sunday → observed Monday June 20
        assert not cal.is_trading_day(date(2022, 6, 20))
        # June 19 itself is a Sunday (not a trading day for that reason)
        assert not cal.is_trading_day(date(2022, 6, 19))

    def test_juneteenth_not_holiday_before_2022(self, cal: TradingCalendar) -> None:
        # June 19, 2021 is a Saturday — but Juneteenth wasn't an NYSE holiday yet
        # June 18, 2021 (Friday before) should be a trading day
        assert cal.is_trading_day(date(2021, 6, 18))

    # ── Independence Day ──────────────────────────────────────────────────

    def test_independence_day_2025_skipped(self, cal: TradingCalendar) -> None:
        # July 4, 2025 is a Friday
        assert not cal.is_trading_day(date(2025, 7, 4))

    def test_independence_day_observed_friday_2020(self, cal: TradingCalendar) -> None:
        # July 4, 2020 is a Saturday → observed Friday July 3
        assert not cal.is_trading_day(date(2020, 7, 3))
        assert cal.is_trading_day(date(2020, 7, 6))  # Monday is trading day

    # ── Labor Day ─────────────────────────────────────────────────────────

    def test_labor_day_2025_skipped(self, cal: TradingCalendar) -> None:
        # Labor Day 2025: September 1 (1st Monday)
        assert not cal.is_trading_day(date(2025, 9, 1))
        assert cal.most_recent_trading_day(date(2025, 9, 1)) == date(2025, 8, 29)

    # ── Thanksgiving ──────────────────────────────────────────────────────

    def test_thanksgiving_2025_skipped(self, cal: TradingCalendar) -> None:
        # Thanksgiving 2025: November 27 (4th Thursday)
        assert not cal.is_trading_day(date(2025, 11, 27))
        assert cal.most_recent_trading_day(date(2025, 11, 27)) == date(2025, 11, 26)

    # ── Christmas ─────────────────────────────────────────────────────────

    def test_christmas_2024_skipped(self, cal: TradingCalendar) -> None:
        # December 25, 2024 is a Wednesday
        assert not cal.is_trading_day(date(2024, 12, 25))

    def test_christmas_observed_friday_2021(self, cal: TradingCalendar) -> None:
        # December 25, 2021 is a Saturday → observed Friday December 24
        assert not cal.is_trading_day(date(2021, 12, 24))
        assert cal.is_trading_day(date(2021, 12, 23))  # Thursday is trading

    def test_christmas_observed_monday_2022(self, cal: TradingCalendar) -> None:
        # December 25, 2022 is a Sunday → observed Monday December 26
        assert not cal.is_trading_day(date(2022, 12, 26))
        assert cal.is_trading_day(date(2022, 12, 23))  # Friday is trading

    # ── New Year's Day ────────────────────────────────────────────────────

    def test_new_years_day_2024_skipped(self, cal: TradingCalendar) -> None:
        # January 1, 2024 is a Monday
        assert not cal.is_trading_day(date(2024, 1, 1))

    def test_new_years_observed_monday_2023(self, cal: TradingCalendar) -> None:
        # January 1, 2023 is a Sunday → observed Monday January 2
        assert not cal.is_trading_day(date(2023, 1, 2))
        # December 30, 2022 is a Friday and should be a trading day
        assert cal.is_trading_day(date(2022, 12, 30))

    # ── Easter calculation ────────────────────────────────────────────────

    def test_easter_known_dates(self, cal: TradingCalendar) -> None:
        assert TradingCalendar._easter(2024) == date(2024, 3, 31)
        assert TradingCalendar._easter(2025) == date(2025, 4, 20)
        assert TradingCalendar._easter(2023) == date(2023, 4, 9)
        assert TradingCalendar._easter(2000) == date(2000, 4, 23)
