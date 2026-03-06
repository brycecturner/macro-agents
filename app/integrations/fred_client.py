import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)


class FREDClientError(Exception):
    """Raised when the FRED API returns an error or is unreachable."""


@dataclass
class FREDSeriesResult:
    """Result of a FRED series fetch, with retrieval timestamp for citations.

    Citation format: ``FRED:{series_id}, retrieved {retrieved_at.date()}``
    """

    series_id: str
    data: pd.Series  # DatetimeIndex → float values as returned by fredapi
    retrieved_at: datetime


@dataclass
class FREDReleaseDatesResult:
    """Result of a FRED release dates fetch, with retrieval timestamp for citations."""

    release_id: int
    dates: list[date]
    retrieved_at: datetime


class FREDClient:
    """Wraps the fredapi library with caching, typed errors, and citation timestamps.

    Instantiate once per workflow run. All results are cached per instance so
    repeated requests for the same series within a single workflow execution do
    not make redundant API calls.
    """

    def __init__(self, api_key: str) -> None:
        self._fred = Fred(api_key=api_key)
        self._series_cache: dict[
            tuple[str, date | None, date | None], FREDSeriesResult
        ] = {}
        self._release_dates_cache: dict[int, FREDReleaseDatesResult] = {}

    def get_series(
        self,
        series_id: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> FREDSeriesResult:
        """Fetch a FRED time series.

        Args:
            series_id: FRED series identifier (e.g. ``"T10Y2Y"``, ``"CPIAUCSL"``).
            start_date: First observation date, inclusive. ``None`` means no lower
                bound.
            end_date: Last observation date, inclusive. ``None`` means no upper bound.

        Returns:
            :class:`FREDSeriesResult` containing the series data and the UTC
            timestamp at which the data was retrieved.

        Raises:
            FREDClientError: If the FRED API call fails for any reason.
        """
        cache_key = (series_id, start_date, end_date)
        if cache_key in self._series_cache:
            logger.debug("FRED series cache hit: %s", series_id)
            return self._series_cache[cache_key]

        try:
            data: pd.Series = self._fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
        except Exception as exc:
            raise FREDClientError(
                f"Failed to fetch FRED series '{series_id}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        result = FREDSeriesResult(
            series_id=series_id,
            data=data,
            retrieved_at=datetime.now(tz=UTC),
        )
        self._series_cache[cache_key] = result
        logger.debug("FRED series fetched and cached: %s", series_id)
        return result

    def get_release_dates(self, release_id: int) -> FREDReleaseDatesResult:
        """Fetch all release dates for a FRED release series.

        Args:
            release_id: FRED release identifier (e.g. ``10`` for CPI).

        Returns:
            :class:`FREDReleaseDatesResult` containing a list of release dates
            and the UTC timestamp at which the data was retrieved.

        Raises:
            FREDClientError: If the FRED API call fails for any reason.
        """
        if release_id in self._release_dates_cache:
            logger.debug("FRED release dates cache hit: release_id=%d", release_id)
            return self._release_dates_cache[release_id]

        try:
            df: pd.DataFrame = self._fred.get_release_dates(release_id)
        except Exception as exc:
            raise FREDClientError(
                f"Failed to fetch FRED release dates for release_id={release_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            dates: list[date] = [pd.Timestamp(d).date() for d in df["date"]]
        except (KeyError, Exception) as exc:
            raise FREDClientError(
                f"Unexpected response format for FRED release dates "
                f"(release_id={release_id}): {type(exc).__name__}: {exc}"
            ) from exc

        result = FREDReleaseDatesResult(
            release_id=release_id,
            dates=dates,
            retrieved_at=datetime.now(tz=UTC),
        )
        self._release_dates_cache[release_id] = result
        logger.debug(
            "FRED release dates fetched and cached: release_id=%d, count=%d",
            release_id,
            len(dates),
        )
        return result
