from datetime import date, timedelta


class TradingCalendar:
    """US NYSE trading calendar utility.

    Used consistently by all workflows, the daily monitoring job, and the
    position sizer to resolve trading dates. Never use inline date logic
    elsewhere — always call this class.
    """

    def most_recent_trading_day(self, as_of: date) -> date:
        """Return the most recent US trading day on or before as_of.

        If as_of is a trading day, returns as_of. Otherwise walks backwards
        until a trading day is found (e.g. Saturday → Friday, holiday → day before).
        """
        d = as_of
        while not self.is_trading_day(d):
            d -= timedelta(days=1)
        return d

    def is_trading_day(self, d: date) -> bool:
        """Return True if d is a US NYSE trading day (weekday, not a holiday)."""
        if d.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        return d not in self._nyse_holidays(d.year)

    def _nyse_holidays(self, year: int) -> frozenset[date]:
        """Compute all NYSE holidays for the given year.

        Holidays observed on Monday when they fall on Sunday, and on Friday
        when they fall on Saturday — matching NYSE rules.
        """
        holidays: list[date] = []

        # New Year's Day — January 1
        holidays.append(_observed(date(year, 1, 1)))

        # Martin Luther King Jr. Day — 3rd Monday in January
        holidays.append(_nth_weekday(year, 1, 0, 3))

        # Presidents' Day — 3rd Monday in February
        holidays.append(_nth_weekday(year, 2, 0, 3))

        # Good Friday — 2 days before Easter Sunday
        holidays.append(self._easter(year) - timedelta(days=2))

        # Memorial Day — last Monday in May
        holidays.append(_last_weekday(year, 5, 0))

        # Juneteenth — June 19 (observed), added by NYSE from 2022 onwards
        if year >= 2022:
            holidays.append(_observed(date(year, 6, 19)))

        # Independence Day — July 4
        holidays.append(_observed(date(year, 7, 4)))

        # Labor Day — 1st Monday in September
        holidays.append(_nth_weekday(year, 9, 0, 1))

        # Thanksgiving — 4th Thursday in November
        holidays.append(_nth_weekday(year, 11, 3, 4))

        # Christmas — December 25
        holidays.append(_observed(date(year, 12, 25)))

        return frozenset(holidays)

    @staticmethod
    def _easter(year: int) -> date:
        """Compute Easter Sunday for the given year using Butcher's algorithm."""
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        ll = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * ll) // 451
        month = (h + ll - 7 * m + 114) // 31
        day = ((h + ll - 7 * m + 114) % 31) + 1
        return date(year, month, day)


def _observed(d: date) -> date:
    """Return the observed date for a fixed-date holiday.

    Saturday → preceding Friday; Sunday → following Monday.
    """
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the nth occurrence of weekday (0=Mon … 6=Sun) in the given month/year."""
    first = date(year, month, 1)
    # Days until the first occurrence of the target weekday
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of weekday (0=Mon … 6=Sun) in the given month/year."""
    # Start from the last day of the month and walk backwards
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)
