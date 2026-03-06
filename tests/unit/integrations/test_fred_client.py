from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.integrations.fred_client import (
    FREDClient,
    FREDClientError,
    FREDReleaseDatesResult,
    FREDSeriesResult,
)


@pytest.fixture
def mock_fred():
    with patch("app.integrations.fred_client.Fred") as mock_cls:
        instance = MagicMock()
        mock_cls.return_value = instance
        yield instance


@pytest.fixture
def client(mock_fred) -> FREDClient:
    return FREDClient(api_key="test-key")


def _make_series(*values: float) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(list(values), index=index)


def _make_release_df(*date_strings: str) -> pd.DataFrame:
    return pd.DataFrame({"date": list(date_strings)})


class TestFREDClientGetSeries:
    def test_returns_series_result(self, client, mock_fred):
        mock_fred.get_series.return_value = _make_series(1.0, 2.0)

        result = client.get_series("T10Y2Y")

        assert isinstance(result, FREDSeriesResult)
        assert result.series_id == "T10Y2Y"
        assert len(result.data) == 2

    def test_retrieved_at_is_utc_datetime(self, client, mock_fred):
        mock_fred.get_series.return_value = _make_series(1.0)

        result = client.get_series("UNRATE")

        assert isinstance(result.retrieved_at, datetime)
        assert result.retrieved_at.tzinfo == UTC

    def test_passes_date_range_to_fredapi(self, client, mock_fred):
        mock_fred.get_series.return_value = _make_series()
        start = date(2023, 1, 1)
        end = date(2024, 1, 1)

        client.get_series("CPIAUCSL", start_date=start, end_date=end)

        mock_fred.get_series.assert_called_once_with(
            "CPIAUCSL", observation_start=start, observation_end=end
        )

    def test_passes_none_dates_when_not_provided(self, client, mock_fred):
        mock_fred.get_series.return_value = _make_series()

        client.get_series("FEDFUNDS")

        mock_fred.get_series.assert_called_once_with(
            "FEDFUNDS", observation_start=None, observation_end=None
        )

    def test_caches_result_on_repeat_call(self, client, mock_fred):
        mock_fred.get_series.return_value = _make_series(1.0)

        first = client.get_series("T10Y2Y")
        second = client.get_series("T10Y2Y")

        assert first is second
        mock_fred.get_series.assert_called_once()

    def test_cache_key_varies_by_series_id(self, client, mock_fred):
        mock_fred.get_series.return_value = _make_series(1.0)

        client.get_series("T10Y2Y")
        client.get_series("UNRATE")

        assert mock_fred.get_series.call_count == 2

    def test_cache_key_varies_by_start_date(self, client, mock_fred):
        mock_fred.get_series.return_value = _make_series(1.0)

        client.get_series("T10Y2Y", start_date=date(2023, 1, 1))
        client.get_series("T10Y2Y", start_date=date(2022, 1, 1))

        assert mock_fred.get_series.call_count == 2

    def test_cache_key_varies_by_end_date(self, client, mock_fred):
        mock_fred.get_series.return_value = _make_series(1.0)

        client.get_series("T10Y2Y", end_date=date(2024, 1, 1))
        client.get_series("T10Y2Y", end_date=date(2023, 1, 1))

        assert mock_fred.get_series.call_count == 2

    def test_raises_fred_client_error_on_api_failure(self, client, mock_fred):
        mock_fred.get_series.side_effect = ValueError("bad series id")

        with pytest.raises(
            FREDClientError, match="Failed to fetch FRED series 'T10Y2Y'"
        ):
            client.get_series("T10Y2Y")

    def test_error_message_includes_exception_type(self, client, mock_fred):
        mock_fred.get_series.side_effect = ValueError("bad series id")

        with pytest.raises(FREDClientError, match="ValueError"):
            client.get_series("T10Y2Y")

    def test_error_message_includes_original_message(self, client, mock_fred):
        mock_fred.get_series.side_effect = RuntimeError("connection timed out")

        with pytest.raises(FREDClientError, match="connection timed out"):
            client.get_series("T10Y2Y")

    def test_original_exception_is_chained(self, client, mock_fred):
        original = ValueError("bad series id")
        mock_fred.get_series.side_effect = original

        with pytest.raises(FREDClientError) as exc_info:
            client.get_series("T10Y2Y")

        assert exc_info.value.__cause__ is original

    def test_failed_call_is_not_cached(self, client, mock_fred):
        mock_fred.get_series.side_effect = [
            RuntimeError("timeout"),
            _make_series(1.0),
        ]

        with pytest.raises(FREDClientError):
            client.get_series("T10Y2Y")

        result = client.get_series("T10Y2Y")
        assert isinstance(result, FREDSeriesResult)


class TestFREDClientGetReleaseDates:
    def test_returns_release_dates_result(self, client, mock_fred):
        mock_fred.get_release_dates.return_value = _make_release_df(
            "2024-01-12", "2024-02-13"
        )

        result = client.get_release_dates(10)

        assert isinstance(result, FREDReleaseDatesResult)
        assert result.release_id == 10
        assert len(result.dates) == 2

    def test_parses_dates_as_date_objects(self, client, mock_fred):
        mock_fred.get_release_dates.return_value = _make_release_df("2024-03-12")

        result = client.get_release_dates(10)

        assert result.dates[0] == date(2024, 3, 12)
        assert all(isinstance(d, date) for d in result.dates)

    def test_retrieved_at_is_utc_datetime(self, client, mock_fred):
        mock_fred.get_release_dates.return_value = _make_release_df("2024-01-12")

        result = client.get_release_dates(10)

        assert isinstance(result.retrieved_at, datetime)
        assert result.retrieved_at.tzinfo == UTC

    def test_caches_result_on_repeat_call(self, client, mock_fred):
        mock_fred.get_release_dates.return_value = _make_release_df("2024-01-12")

        first = client.get_release_dates(10)
        second = client.get_release_dates(10)

        assert first is second
        mock_fred.get_release_dates.assert_called_once()

    def test_cache_key_varies_by_release_id(self, client, mock_fred):
        mock_fred.get_release_dates.return_value = _make_release_df("2024-01-12")

        client.get_release_dates(10)
        client.get_release_dates(20)

        assert mock_fred.get_release_dates.call_count == 2

    def test_raises_fred_client_error_on_api_failure(self, client, mock_fred):
        mock_fred.get_release_dates.side_effect = ConnectionError("network error")

        with pytest.raises(
            FREDClientError,
            match="Failed to fetch FRED release dates for release_id=10",
        ):
            client.get_release_dates(10)

    def test_error_message_includes_exception_type(self, client, mock_fred):
        mock_fred.get_release_dates.side_effect = ConnectionError("network error")

        with pytest.raises(FREDClientError, match="ConnectionError"):
            client.get_release_dates(10)

    def test_error_message_includes_original_message(self, client, mock_fred):
        mock_fred.get_release_dates.side_effect = ConnectionError("network error")

        with pytest.raises(FREDClientError, match="network error"):
            client.get_release_dates(10)

    def test_original_exception_is_chained(self, client, mock_fred):
        original = ConnectionError("network error")
        mock_fred.get_release_dates.side_effect = original

        with pytest.raises(FREDClientError) as exc_info:
            client.get_release_dates(10)

        assert exc_info.value.__cause__ is original

    def test_raises_on_malformed_response(self, client, mock_fred):
        # DataFrame missing the expected 'date' column
        mock_fred.get_release_dates.return_value = pd.DataFrame(
            {"wrong_column": ["2024-01-12"]}
        )

        with pytest.raises(FREDClientError, match="Unexpected response format"):
            client.get_release_dates(10)

    def test_failed_call_is_not_cached(self, client, mock_fred):
        mock_fred.get_release_dates.side_effect = [
            RuntimeError("timeout"),
            _make_release_df("2024-01-12"),
        ]

        with pytest.raises(FREDClientError):
            client.get_release_dates(10)

        result = client.get_release_dates(10)
        assert isinstance(result, FREDReleaseDatesResult)


class TestFREDClientCacheIsolation:
    def test_series_and_release_caches_are_independent(self, client, mock_fred):
        mock_fred.get_series.return_value = _make_series(1.0)
        mock_fred.get_release_dates.return_value = _make_release_df("2024-01-12")

        client.get_series("T10Y2Y")
        client.get_series("T10Y2Y")
        client.get_release_dates(10)
        client.get_release_dates(10)

        mock_fred.get_series.assert_called_once()
        mock_fred.get_release_dates.assert_called_once()

    def test_separate_client_instances_have_separate_caches(self, mock_fred):
        mock_fred.get_series.return_value = _make_series(1.0)

        client_a = FREDClient(api_key="key-a")
        client_b = FREDClient(api_key="key-b")

        client_a.get_series("T10Y2Y")
        client_b.get_series("T10Y2Y")

        assert mock_fred.get_series.call_count == 2
