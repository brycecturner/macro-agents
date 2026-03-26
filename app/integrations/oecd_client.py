"""OECD Data Explorer client using the new sdmx.oecd.org REST API.

The old stats.oecd.org/SDMX-JSON endpoint was shut down on 2024-07-01 and
permanently redirects. All series now live on the OECD Data Explorer:
    https://sdmx.oecd.org/public/rest/data/

URL format:
    {BASE}/{agency},{dataflow},{version}/{dimension_key}?format=csvfile&...

To find the correct agency/dataflow/version/dimension_key for any series:
    1. Browse https://data-explorer.oecd.org
    2. Find your indicator
    3. Click the "Developer API" button above the data table
    4. Copy the generated URL and read off the path components
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from io import StringIO

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

OECD_BASE_URL = "https://sdmx.oecd.org/public/rest/data"


class OECDClientError(Exception):
    """Raised when the OECD API returns an error or is unreachable."""


@dataclass
class OECDSeriesSpec:
    """Identifies a single OECD time series in the new sdmx.oecd.org API.

    All four path components come from the Developer API button on
    https://data-explorer.oecd.org — find your indicator, click the button,
    and read off the components from the generated URL.

    Example (OECD Composite Leading Indicator):
        agency        = "OECD.SDD.STES"
        dataflow      = "DSD_STES@DF_CLI"
        version       = "4.1"
        dimension_key = "OECD.M.CCICP.IX._Z.AA.IX._Z.H"
        label         = "OECD Composite Leading Indicator"
    """

    agency: str
    dataflow: str
    version: str
    dimension_key: str
    label: str


@dataclass
class OECDSeriesResult:
    """Result of an OECD series fetch, with retrieval timestamp for citations.

    Citation format:
        ``OECD:{spec.dataflow}/{spec.dimension_key}, retrieved {retrieved_at.date()}``
    """

    spec: OECDSeriesSpec
    data: pd.Series  # DatetimeIndex → float values
    retrieved_at: datetime


def _normalize_time_period(tp: str) -> str:
    """Convert an OECD TIME_PERIOD value to a pandas-parseable date string.

    Handles:
    - Monthly:   "2024-01"  → "2024-01"  (pass through)
    - Annual:    "2024"     → "2024"     (pass through)
    - Quarterly: "2024-Q1"  → "2024-01"  (first month of quarter)
    """
    if "-Q" in tp:
        year, q = tp.split("-Q")
        month = {"1": "01", "2": "04", "3": "07", "4": "10"}[q]
        return f"{year}-{month}"
    return tp


class OECDClient:
    """Wraps the OECD sdmx.oecd.org REST API with caching, typed errors, and citations.

    Rate limit: 20 requests per hour per IP. Per-instance caching prevents
    redundant calls within a single workflow run.

    Instantiate once per workflow run and inject explicitly in tests.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._http = httpx.Client(timeout=timeout, follow_redirects=True)
        self._cache: dict[tuple, OECDSeriesResult] = {}

    def get_series(
        self,
        spec: OECDSeriesSpec,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> OECDSeriesResult:
        """Fetch an OECD time series.

        Args:
            spec: Series specification — agency, dataflow, version, dimension_key.
            start_date: First observation date, inclusive. Formatted as YYYY-MM.
            end_date: Last observation date, inclusive. Formatted as YYYY-MM.

        Returns:
            OECDSeriesResult containing a DatetimeIndex pandas Series and
            the UTC timestamp at which the data was retrieved.

        Raises:
            OECDClientError: If the API call fails or returns an unparseable response.
        """
        cache_key = (
            spec.agency,
            spec.dataflow,
            spec.version,
            spec.dimension_key,
            start_date,
            end_date,
        )
        if cache_key in self._cache:
            logger.debug("OECD cache hit: %s", spec.label)
            return self._cache[cache_key]

        url = (
            f"{OECD_BASE_URL}/{spec.agency},{spec.dataflow},{spec.version}"
            f"/{spec.dimension_key}"
        )
        params: dict[str, str] = {
            "format": "csvfile",
            "dimensionAtObservation": "AllDimensions",
        }
        if start_date is not None:
            params["startPeriod"] = start_date.strftime("%Y-%m")
        if end_date is not None:
            params["endPeriod"] = end_date.strftime("%Y-%m")

        try:
            response = self._http.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OECDClientError(
                f"OECD API returned HTTP {exc.response.status_code} for "
                f"{spec.label!r} ({spec.agency},{spec.dataflow},{spec.version}): {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise OECDClientError(
                f"Failed to reach OECD API for {spec.label!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            data = self._parse_csv(response.text)
        except OECDClientError:
            raise
        except Exception as exc:
            raise OECDClientError(
                f"Failed to parse OECD response for {spec.label!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        result = OECDSeriesResult(
            spec=spec,
            data=data,
            retrieved_at=datetime.now(tz=UTC),
        )
        self._cache[cache_key] = result
        logger.debug(
            "OECD series fetched: %s, %d observations",
            spec.label,
            len(data),
        )
        return result

    def _parse_csv(self, text: str) -> pd.Series:
        """Parse an OECD csvfile response into a pandas Series with DatetimeIndex.

        Expects columns TIME_PERIOD and OBS_VALUE. Handles monthly (YYYY-MM),
        annual (YYYY), and quarterly (YYYY-QN) time period formats.
        """
        try:
            df = pd.read_csv(StringIO(text))
        except Exception as exc:
            raise OECDClientError(
                f"Could not parse OECD response as CSV: {type(exc).__name__}: {exc}"
            ) from exc

        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            raise OECDClientError(
                f"Expected TIME_PERIOD and OBS_VALUE columns in OECD CSV response; "
                f"got: {list(df.columns)}"
            )

        pairs = df[["TIME_PERIOD", "OBS_VALUE"]].dropna(subset=["OBS_VALUE"])
        if pairs.empty:
            return pd.Series(dtype=float)

        normalized = pairs["TIME_PERIOD"].astype(str).map(_normalize_time_period)
        index = pd.to_datetime(normalized, format="mixed", yearfirst=True)
        series = pd.Series(pairs["OBS_VALUE"].astype(float).values, index=index)
        return series.sort_index()

    def close(self) -> None:
        """Close the underlying HTTP client. Call when the workflow run is complete."""
        self._http.close()
