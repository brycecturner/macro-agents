import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

OECD_BASE_URL = "https://stats.oecd.org/SDMX-JSON/data"


class OECDClientError(Exception):
    """Raised when the OECD API returns an error or is unreachable."""


@dataclass
class OECDSeriesResult:
    """Result of an OECD series fetch, with retrieval timestamp for citations.

    Citation format:
    ``OECD:{dataset}/{subject}/{country}, retrieved {retrieved_at.date()}``
    """

    dataset: str
    subject: str
    country: str
    data: pd.Series  # DatetimeIndex → float values
    retrieved_at: datetime


class OECDClient:
    """Wraps the OECD JSON RESTful API with caching, typed errors, and citations.

    Instantiate once per workflow run. All results are cached per instance so
    repeated requests for the same series within a single workflow execution do
    not make redundant API calls.

    The OECD API is free and requires no authentication.

    Supported series examples:
    - ECB policy rates:    dataset="MEI_FIN", subject="IRSTCB01", country="EA19"
    - Eurozone CPI:        dataset="PRICES_CPI", subject="CPALTT01", country="EA19"
    - G10 yield curves:   dataset="MEI_FIN", subject="IRLT", country="DEU" (etc.)
    - OECD composite PMI: dataset="MEI_CLI", subject="LOLITOAA", country="OECD"
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._http = httpx.Client(timeout=timeout)
        self._cache: dict[
            tuple[str, str, str, date | None, date | None], OECDSeriesResult
        ] = {}

    def get_series(
        self,
        dataset: str,
        subject: str,
        country: str,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> OECDSeriesResult:
        """Fetch an OECD time series.

        Args:
            dataset: OECD dataset code (e.g. ``"MEI_FIN"``, ``"PRICES_CPI"``).
            subject: Subject/measure code within the dataset (e.g. ``"IRLT"``).
            country: ISO country or region code (e.g. ``"DEU"``, ``"EA19"``).
            start_date: First observation date, inclusive. ``None`` means no lower
                bound. Formatted as ``YYYY-MM`` in the API request.
            end_date: Last observation date, inclusive. ``None`` means no upper bound.
                Formatted as ``YYYY-MM`` in the API request.

        Returns:
            :class:`OECDSeriesResult` containing the series data and the UTC
            timestamp at which the data was retrieved.

        Raises:
            OECDClientError: If the OECD API call fails or returns an
                unparseable response.
        """
        cache_key = (dataset, subject, country, start_date, end_date)
        if cache_key in self._cache:
            logger.debug("OECD cache hit: %s/%s/%s", dataset, subject, country)
            return self._cache[cache_key]

        url = f"{OECD_BASE_URL}/{dataset}/{subject}.{country}/all"
        params: dict[str, str] = {}
        if start_date is not None:
            params["startTime"] = start_date.strftime("%Y-%m")
        if end_date is not None:
            params["endTime"] = end_date.strftime("%Y-%m")

        try:
            response = self._http.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OECDClientError(
                f"OECD API returned HTTP {exc.response.status_code} for "
                f"dataset={dataset!r}, subject={subject!r}, country={country!r}: {exc}"
            ) from exc
        except httpx.RequestError as exc:
            raise OECDClientError(
                f"Failed to reach OECD API for dataset={dataset!r}, "
                f"subject={subject!r}, country={country!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            data = self._parse_sdmx_json(response.json())
        except OECDClientError:
            raise
        except Exception as exc:
            raise OECDClientError(
                f"Failed to parse OECD response for dataset={dataset!r}, "
                f"subject={subject!r}, country={country!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        result = OECDSeriesResult(
            dataset=dataset,
            subject=subject,
            country=country,
            data=data,
            retrieved_at=datetime.now(tz=UTC),
        )
        self._cache[cache_key] = result
        logger.debug(
            "OECD series fetched and cached: %s/%s/%s, %d observations",
            dataset,
            subject,
            country,
            len(data),
        )
        return result

    def _parse_sdmx_json(self, payload: dict) -> pd.Series:
        """Parse an OECD SDMX-JSON response into a pandas Series with DatetimeIndex.

        The OECD API returns data in SDMX-JSON format where observations are
        indexed by integer position and time period labels live in the structure
        dimensions block.
        """
        try:
            structure = payload["structure"]
            datasets = payload["dataSets"]
        except KeyError as exc:
            raise OECDClientError(
                f"Unexpected OECD response structure: missing key {exc}"
            ) from exc

        if not datasets:
            raise OECDClientError("OECD response contained no dataSets")

        obs_dimensions = structure["dimensions"]["observation"]
        time_dim = next((d for d in obs_dimensions if d["id"] == "TIME_PERIOD"), None)
        if time_dim is None:
            raise OECDClientError(
                "OECD response missing TIME_PERIOD dimension in observations"
            )

        time_periods: list[str] = [v["id"] for v in time_dim["values"]]

        # Collect all observations across all series in the first dataset.
        # Multiple series keys occur when the filter matches more than one
        # combination of dimensions (e.g. multiple subjects or countries).
        observations: dict[str, float] = {}
        for series in datasets[0].get("series", {}).values():
            for obs_idx_str, obs_values in series.get("observations", {}).items():
                value = obs_values[0]
                if value is not None:
                    time_label = time_periods[int(obs_idx_str)]
                    observations[time_label] = float(value)

        if not observations:
            return pd.Series(dtype=float)

        index = pd.to_datetime(list(observations.keys()))
        series = pd.Series(list(observations.values()), index=index)
        return series.sort_index()

    def close(self) -> None:
        """Close the underlying HTTP client. Call when the workflow run is complete."""
        self._http.close()
