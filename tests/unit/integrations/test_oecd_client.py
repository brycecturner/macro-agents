from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from app.integrations.oecd_client import OECDClient, OECDClientError, OECDSeriesResult

# ---------------------------------------------------------------------------
# Minimal valid SDMX-JSON payload for a three-observation monthly series
# ---------------------------------------------------------------------------

_MOCK_SDMX = {
    "header": {},
    "dataSets": [
        {
            "series": {
                "0:0": {
                    "observations": {
                        "0": [1.5, 0],
                        "1": [1.75, 0],
                        "2": [2.0, 0],
                    }
                }
            }
        }
    ],
    "structure": {
        "dimensions": {
            "series": [],
            "observation": [
                {
                    "id": "TIME_PERIOD",
                    "name": "Time period",
                    "values": [
                        {"id": "2023-01"},
                        {"id": "2023-02"},
                        {"id": "2023-03"},
                    ],
                }
            ],
        }
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_response(json_data: dict) -> MagicMock:
    """Build a mock httpx.Response for a successful 200 reply."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _error_response(status_code: int) -> MagicMock:
    """Build a mock httpx.Response that raises on raise_for_status."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}", request=MagicMock(), response=resp
    )
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> OECDClient:
    return OECDClient()


# ---------------------------------------------------------------------------
# TestOECDClientGetSeries — happy path
# ---------------------------------------------------------------------------


class TestOECDClientGetSeries:
    def test_returns_oecd_series_result(self, client):
        with patch.object(client._http, "get", return_value=_ok_response(_MOCK_SDMX)):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert isinstance(result, OECDSeriesResult)

    def test_result_fields_match_request(self, client):
        with patch.object(client._http, "get", return_value=_ok_response(_MOCK_SDMX)):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert result.dataset == "MEI_FIN"
        assert result.subject == "IRLT"
        assert result.country == "DEU"

    def test_data_is_pandas_series(self, client):
        with patch.object(client._http, "get", return_value=_ok_response(_MOCK_SDMX)):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert isinstance(result.data, pd.Series)

    def test_data_has_correct_length(self, client):
        with patch.object(client._http, "get", return_value=_ok_response(_MOCK_SDMX)):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert len(result.data) == 3

    def test_data_values_are_correct(self, client):
        with patch.object(client._http, "get", return_value=_ok_response(_MOCK_SDMX)):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert result.data.iloc[0] == pytest.approx(1.5)
        assert result.data.iloc[1] == pytest.approx(1.75)
        assert result.data.iloc[2] == pytest.approx(2.0)

    def test_data_index_is_datetime(self, client):
        with patch.object(client._http, "get", return_value=_ok_response(_MOCK_SDMX)):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert isinstance(result.data.index, pd.DatetimeIndex)

    def test_data_is_sorted_ascending(self, client):
        # Reverse the observation order to verify sorting
        reversed_sdmx = {
            **_MOCK_SDMX,
            "dataSets": [
                {
                    "series": {
                        "0:0": {
                            "observations": {
                                "0": [2.0, 0],
                                "1": [1.75, 0],
                                "2": [1.5, 0],
                            }
                        }
                    }
                }
            ],
            "structure": {
                "dimensions": {
                    "series": [],
                    "observation": [
                        {
                            "id": "TIME_PERIOD",
                            "values": [
                                {"id": "2023-03"},
                                {"id": "2023-02"},
                                {"id": "2023-01"},
                            ],
                        }
                    ],
                }
            },
        }
        with patch.object(
            client._http, "get", return_value=_ok_response(reversed_sdmx)
        ):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert result.data.index.is_monotonic_increasing

    def test_retrieved_at_is_utc_datetime(self, client):
        before = datetime.now(UTC)
        with patch.object(client._http, "get", return_value=_ok_response(_MOCK_SDMX)):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")
        after = datetime.now(UTC)

        assert isinstance(result.retrieved_at, datetime)
        assert result.retrieved_at.tzinfo == UTC
        assert before <= result.retrieved_at <= after

    def test_none_observations_are_skipped(self, client):
        sdmx_with_null = {
            **_MOCK_SDMX,
            "dataSets": [
                {
                    "series": {
                        "0:0": {
                            "observations": {
                                "0": [1.5, 0],
                                "1": [None, 0],  # null observation
                                "2": [2.0, 0],
                            }
                        }
                    }
                }
            ],
        }
        with patch.object(
            client._http, "get", return_value=_ok_response(sdmx_with_null)
        ):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert len(result.data) == 2


# ---------------------------------------------------------------------------
# TestOECDClientRequestParams — URL and query parameter construction
# ---------------------------------------------------------------------------


class TestOECDClientRequestParams:
    def test_url_contains_dataset_subject_and_country(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU")

        url = mock_get.call_args.args[0]
        assert "MEI_FIN" in url
        assert "IRLT" in url
        assert "DEU" in url

    def test_start_date_formatted_as_year_month(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU", start_date=date(2020, 3, 15))

        params = mock_get.call_args.kwargs["params"]
        assert params["startTime"] == "2020-03"

    def test_end_date_formatted_as_year_month(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU", end_date=date(2023, 12, 31))

        params = mock_get.call_args.kwargs["params"]
        assert params["endTime"] == "2023-12"

    def test_no_time_params_when_dates_are_none(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU")

        params = mock_get.call_args.kwargs["params"]
        assert "startTime" not in params
        assert "endTime" not in params

    def test_both_date_params_passed_together(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series(
                "MEI_FIN",
                "IRLT",
                "DEU",
                start_date=date(2018, 1, 1),
                end_date=date(2023, 6, 30),
            )

        params = mock_get.call_args.kwargs["params"]
        assert params["startTime"] == "2018-01"
        assert params["endTime"] == "2023-06"


# ---------------------------------------------------------------------------
# TestOECDClientCaching — instance-level cache behaviour
# ---------------------------------------------------------------------------


class TestOECDClientCaching:
    def test_second_call_with_same_params_returns_same_object(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            first = client.get_series("MEI_FIN", "IRLT", "DEU")
            second = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert first is second

    def test_cached_call_makes_only_one_http_request(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU")
            client.get_series("MEI_FIN", "IRLT", "DEU")

        mock_get.assert_called_once()

    def test_cache_key_varies_by_dataset(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU")
            client.get_series("PRICES_CPI", "IRLT", "DEU")

        assert mock_get.call_count == 2

    def test_cache_key_varies_by_subject(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU")
            client.get_series("MEI_FIN", "IRSTCB01", "DEU")

        assert mock_get.call_count == 2

    def test_cache_key_varies_by_country(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU")
            client.get_series("MEI_FIN", "IRLT", "GBR")

        assert mock_get.call_count == 2

    def test_cache_key_varies_by_start_date(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU", start_date=date(2020, 1, 1))
            client.get_series("MEI_FIN", "IRLT", "DEU", start_date=date(2021, 1, 1))

        assert mock_get.call_count == 2

    def test_cache_key_varies_by_end_date(self, client):
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client._http, "get", mock_get):
            client.get_series("MEI_FIN", "IRLT", "DEU", end_date=date(2022, 12, 31))
            client.get_series("MEI_FIN", "IRLT", "DEU", end_date=date(2023, 12, 31))

        assert mock_get.call_count == 2

    def test_failed_call_is_not_cached(self, client):
        responses = [_error_response(500), _ok_response(_MOCK_SDMX)]
        mock_get = MagicMock(side_effect=responses)
        with patch.object(client._http, "get", mock_get):
            with pytest.raises(OECDClientError):
                client.get_series("MEI_FIN", "IRLT", "DEU")

            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert isinstance(result, OECDSeriesResult)
        assert mock_get.call_count == 2

    def test_separate_instances_have_separate_caches(self):
        client_a = OECDClient()
        client_b = OECDClient()
        mock_get = MagicMock(return_value=_ok_response(_MOCK_SDMX))
        with patch.object(client_a._http, "get", mock_get):
            with patch.object(client_b._http, "get", mock_get):
                client_a.get_series("MEI_FIN", "IRLT", "DEU")
                client_b.get_series("MEI_FIN", "IRLT", "DEU")

        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# TestOECDClientErrors — error handling and typing
# ---------------------------------------------------------------------------


class TestOECDClientErrors:
    def test_http_4xx_raises_oecd_client_error(self, client):
        with patch.object(client._http, "get", return_value=_error_response(404)):
            with pytest.raises(OECDClientError, match="HTTP 404"):
                client.get_series("MEI_FIN", "IRLT", "DEU")

    def test_http_5xx_raises_oecd_client_error(self, client):
        with patch.object(client._http, "get", return_value=_error_response(503)):
            with pytest.raises(OECDClientError, match="HTTP 503"):
                client.get_series("MEI_FIN", "IRLT", "DEU")

    def test_network_error_raises_oecd_client_error(self, client):
        with patch.object(
            client._http, "get", side_effect=httpx.ConnectError("connection refused")
        ):
            with pytest.raises(OECDClientError, match="ConnectError"):
                client.get_series("MEI_FIN", "IRLT", "DEU")

    def test_timeout_raises_oecd_client_error(self, client):
        with patch.object(
            client._http, "get", side_effect=httpx.TimeoutException("timed out")
        ):
            with pytest.raises(OECDClientError):
                client.get_series("MEI_FIN", "IRLT", "DEU")

    def test_http_error_message_includes_dataset(self, client):
        with patch.object(client._http, "get", return_value=_error_response(400)):
            with pytest.raises(OECDClientError, match="MEI_FIN"):
                client.get_series("MEI_FIN", "IRLT", "DEU")

    def test_network_error_original_exception_is_chained(self, client):
        original = httpx.ConnectError("refused")
        with patch.object(client._http, "get", side_effect=original):
            with pytest.raises(OECDClientError) as exc_info:
                client.get_series("MEI_FIN", "IRLT", "DEU")

        assert exc_info.value.__cause__ is original

    def test_http_error_original_exception_is_chained(self, client):
        error_resp = _error_response(502)
        original = httpx.HTTPStatusError(
            "502", request=MagicMock(), response=error_resp
        )
        error_resp.raise_for_status.side_effect = original
        with patch.object(client._http, "get", return_value=error_resp):
            with pytest.raises(OECDClientError) as exc_info:
                client.get_series("MEI_FIN", "IRLT", "DEU")

        assert exc_info.value.__cause__ is original

    def test_missing_top_level_key_raises_oecd_client_error(self, client):
        with patch.object(
            client._http, "get", return_value=_ok_response({"unexpected": "payload"})
        ):
            with pytest.raises(OECDClientError):
                client.get_series("MEI_FIN", "IRLT", "DEU")

    def test_empty_datasets_raises_oecd_client_error(self, client):
        payload = {**_MOCK_SDMX, "dataSets": []}
        with patch.object(client._http, "get", return_value=_ok_response(payload)):
            with pytest.raises(OECDClientError, match="no dataSets"):
                client.get_series("MEI_FIN", "IRLT", "DEU")

    def test_missing_time_period_dimension_raises_oecd_client_error(self, client):
        payload = {
            "dataSets": [{"series": {"0:0": {"observations": {"0": [1.0, 0]}}}}],
            "structure": {
                "dimensions": {
                    "observation": [{"id": "OTHER_DIM", "values": [{"id": "2023-01"}]}]
                }
            },
        }
        with patch.object(client._http, "get", return_value=_ok_response(payload)):
            with pytest.raises(OECDClientError, match="TIME_PERIOD"):
                client.get_series("MEI_FIN", "IRLT", "DEU")


# ---------------------------------------------------------------------------
# TestOECDClientEmptyData — edge case: valid response with no observations
# ---------------------------------------------------------------------------


class TestOECDClientEmptyData:
    def test_no_observations_returns_empty_series(self, client):
        payload = {
            "dataSets": [{"series": {}}],
            "structure": {
                "dimensions": {"observation": [{"id": "TIME_PERIOD", "values": []}]}
            },
        }
        with patch.object(client._http, "get", return_value=_ok_response(payload)):
            result = client.get_series("MEI_FIN", "IRLT", "DEU")

        assert isinstance(result.data, pd.Series)
        assert len(result.data) == 0
