from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

from app.integrations.oecd_client import (
    OECDClient,
    OECDClientError,
    OECDSeriesResult,
    OECDSeriesSpec,
)

# ---------------------------------------------------------------------------
# Minimal valid CSV payloads
# ---------------------------------------------------------------------------

_MOCK_CSV = """\
STRUCTURE,STRUCTURE_ID,ACTION,TIME_PERIOD,OBS_VALUE,OBS_STATUS
dataflow,test,A,2023-01,1.5,A
dataflow,test,A,2023-02,1.75,A
dataflow,test,A,2023-03,2.0,A
"""

_MOCK_CSV_REVERSED = """\
STRUCTURE,STRUCTURE_ID,ACTION,TIME_PERIOD,OBS_VALUE,OBS_STATUS
dataflow,test,A,2023-03,2.0,A
dataflow,test,A,2023-02,1.75,A
dataflow,test,A,2023-01,1.5,A
"""

_MOCK_CSV_WITH_NULL = """\
STRUCTURE,STRUCTURE_ID,ACTION,TIME_PERIOD,OBS_VALUE,OBS_STATUS
dataflow,test,A,2023-01,1.5,A
dataflow,test,A,2023-02,,A
dataflow,test,A,2023-03,2.0,A
"""

_MOCK_CSV_EMPTY = """\
STRUCTURE,STRUCTURE_ID,ACTION,TIME_PERIOD,OBS_VALUE,OBS_STATUS
"""

_MOCK_CSV_NO_OBS_VALUE = """\
STRUCTURE,TIME_PERIOD,OTHER_COLUMN
dataflow,2023-01,1.5
"""

_MOCK_CSV_NO_TIME_PERIOD = """\
STRUCTURE,OBS_VALUE,OTHER_COLUMN
dataflow,1.5,A
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spec(
    label: str = "Test Series",
    agency: str = "OECD.SDD.STES",
    dataflow: str = "DSD_STES@DF_CLI",
    version: str = "4.1",
    dimension_key: str = "OECD.M.CCICP.IX._Z.AA.IX._Z.H",
) -> OECDSeriesSpec:
    return OECDSeriesSpec(
        agency=agency,
        dataflow=dataflow,
        version=version,
        dimension_key=dimension_key,
        label=label,
    )


def _ok_response(csv_text: str = _MOCK_CSV) -> MagicMock:
    """Build a mock httpx.Response for a successful 200 reply."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.text = csv_text
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
        with patch.object(client._http, "get", return_value=_ok_response()):
            result = client.get_series(_make_spec())

        assert isinstance(result, OECDSeriesResult)

    def test_result_spec_matches_request(self, client):
        spec = _make_spec()
        with patch.object(client._http, "get", return_value=_ok_response()):
            result = client.get_series(spec)

        assert result.spec is spec

    def test_data_is_pandas_series(self, client):
        with patch.object(client._http, "get", return_value=_ok_response()):
            result = client.get_series(_make_spec())

        assert isinstance(result.data, pd.Series)

    def test_data_has_correct_length(self, client):
        with patch.object(client._http, "get", return_value=_ok_response()):
            result = client.get_series(_make_spec())

        assert len(result.data) == 3

    def test_data_values_are_correct(self, client):
        with patch.object(client._http, "get", return_value=_ok_response()):
            result = client.get_series(_make_spec())

        assert result.data.iloc[0] == pytest.approx(1.5)
        assert result.data.iloc[1] == pytest.approx(1.75)
        assert result.data.iloc[2] == pytest.approx(2.0)

    def test_data_index_is_datetime(self, client):
        with patch.object(client._http, "get", return_value=_ok_response()):
            result = client.get_series(_make_spec())

        assert isinstance(result.data.index, pd.DatetimeIndex)

    def test_data_is_sorted_ascending(self, client):
        with patch.object(
            client._http, "get", return_value=_ok_response(_MOCK_CSV_REVERSED)
        ):
            result = client.get_series(_make_spec())

        assert result.data.index.is_monotonic_increasing

    def test_retrieved_at_is_utc_datetime(self, client):
        before = datetime.now(UTC)
        with patch.object(client._http, "get", return_value=_ok_response()):
            result = client.get_series(_make_spec())
        after = datetime.now(UTC)

        assert isinstance(result.retrieved_at, datetime)
        assert result.retrieved_at.tzinfo == UTC
        assert before <= result.retrieved_at <= after

    def test_null_obs_value_rows_are_skipped(self, client):
        with patch.object(
            client._http, "get", return_value=_ok_response(_MOCK_CSV_WITH_NULL)
        ):
            result = client.get_series(_make_spec())

        assert len(result.data) == 2


# ---------------------------------------------------------------------------
# TestOECDClientRequestParams — URL and query parameter construction
# ---------------------------------------------------------------------------


class TestOECDClientRequestParams:
    def test_url_contains_agency_dataflow_and_version(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        spec = _make_spec(
            agency="OECD.SDD.TPS",
            dataflow="DSD_PRICES@DF_PRICES_ALL",
            version="1.0",
            dimension_key="EA20.A.N.CPI.PA._T.N.GY",
        )
        with patch.object(client._http, "get", mock_get):
            client.get_series(spec)

        url = mock_get.call_args.args[0]
        assert "OECD.SDD.TPS" in url
        assert "DSD_PRICES@DF_PRICES_ALL" in url
        assert "1.0" in url

    def test_url_contains_dimension_key(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        spec = _make_spec(dimension_key="EA20.A.N.CPI.PA._T.N.GY")
        with patch.object(client._http, "get", mock_get):
            client.get_series(spec)

        url = mock_get.call_args.args[0]
        assert "EA20.A.N.CPI.PA._T.N.GY" in url

    def test_format_param_is_csvfile(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec())

        params = mock_get.call_args.kwargs["params"]
        assert params["format"] == "csvfile"

    def test_dimension_at_observation_param_present(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec())

        params = mock_get.call_args.kwargs["params"]
        assert params["dimensionAtObservation"] == "AllDimensions"

    def test_start_date_formatted_as_year_month(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec(), start_date=date(2020, 3, 15))

        params = mock_get.call_args.kwargs["params"]
        assert params["startPeriod"] == "2020-03"

    def test_end_date_formatted_as_year_month(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec(), end_date=date(2023, 12, 31))

        params = mock_get.call_args.kwargs["params"]
        assert params["endPeriod"] == "2023-12"

    def test_no_time_params_when_dates_are_none(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec())

        params = mock_get.call_args.kwargs["params"]
        assert "startPeriod" not in params
        assert "endPeriod" not in params

    def test_both_date_params_passed_together(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(
                _make_spec(),
                start_date=date(2018, 1, 1),
                end_date=date(2023, 6, 30),
            )

        params = mock_get.call_args.kwargs["params"]
        assert params["startPeriod"] == "2018-01"
        assert params["endPeriod"] == "2023-06"


# ---------------------------------------------------------------------------
# TestOECDClientCaching — instance-level cache behaviour
# ---------------------------------------------------------------------------


class TestOECDClientCaching:
    def test_second_call_with_same_params_returns_same_object(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        spec = _make_spec()
        with patch.object(client._http, "get", mock_get):
            first = client.get_series(spec)
            second = client.get_series(spec)

        assert first is second

    def test_cached_call_makes_only_one_http_request(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        spec = _make_spec()
        with patch.object(client._http, "get", mock_get):
            client.get_series(spec)
            client.get_series(spec)

        mock_get.assert_called_once()

    def test_cache_key_varies_by_agency(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec(agency="OECD.SDD.STES"))
            client.get_series(_make_spec(agency="OECD.SDD.TPS"))

        assert mock_get.call_count == 2

    def test_cache_key_varies_by_dataflow(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec(dataflow="DSD_STES@DF_CLI"))
            client.get_series(_make_spec(dataflow="DSD_STES@DF_FINMARK"))

        assert mock_get.call_count == 2

    def test_cache_key_varies_by_dimension_key(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec(dimension_key="OECD.M.CCICP.IX._Z.AA.IX._Z.H"))
            client.get_series(
                _make_spec(dimension_key="EA19.M.IRSTCI.PA._Z._Z._Z._Z.N")
            )

        assert mock_get.call_count == 2

    def test_cache_key_varies_by_start_date(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec(), start_date=date(2020, 1, 1))
            client.get_series(_make_spec(), start_date=date(2021, 1, 1))

        assert mock_get.call_count == 2

    def test_cache_key_varies_by_end_date(self, client):
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client._http, "get", mock_get):
            client.get_series(_make_spec(), end_date=date(2022, 12, 31))
            client.get_series(_make_spec(), end_date=date(2023, 12, 31))

        assert mock_get.call_count == 2

    def test_failed_call_is_not_cached(self, client):
        responses = [_error_response(500), _ok_response()]
        mock_get = MagicMock(side_effect=responses)
        spec = _make_spec()
        with patch.object(client._http, "get", mock_get):
            with pytest.raises(OECDClientError):
                client.get_series(spec)

            result = client.get_series(spec)

        assert isinstance(result, OECDSeriesResult)
        assert mock_get.call_count == 2

    def test_separate_instances_have_separate_caches(self):
        client_a = OECDClient()
        client_b = OECDClient()
        spec = _make_spec()
        mock_get = MagicMock(return_value=_ok_response())
        with patch.object(client_a._http, "get", mock_get):
            with patch.object(client_b._http, "get", mock_get):
                client_a.get_series(spec)
                client_b.get_series(spec)

        assert mock_get.call_count == 2


# ---------------------------------------------------------------------------
# TestOECDClientErrors — error handling and typing
# ---------------------------------------------------------------------------


class TestOECDClientErrors:
    def test_http_4xx_raises_oecd_client_error(self, client):
        with patch.object(client._http, "get", return_value=_error_response(404)):
            with pytest.raises(OECDClientError, match="HTTP 404"):
                client.get_series(_make_spec())

    def test_http_5xx_raises_oecd_client_error(self, client):
        with patch.object(client._http, "get", return_value=_error_response(503)):
            with pytest.raises(OECDClientError, match="HTTP 503"):
                client.get_series(_make_spec())

    def test_network_error_raises_oecd_client_error(self, client):
        with patch.object(
            client._http, "get", side_effect=httpx.ConnectError("connection refused")
        ):
            with pytest.raises(OECDClientError, match="ConnectError"):
                client.get_series(_make_spec())

    def test_timeout_raises_oecd_client_error(self, client):
        with patch.object(
            client._http, "get", side_effect=httpx.TimeoutException("timed out")
        ):
            with pytest.raises(OECDClientError):
                client.get_series(_make_spec())

    def test_http_error_message_includes_spec_label(self, client):
        with patch.object(client._http, "get", return_value=_error_response(400)):
            with pytest.raises(OECDClientError, match="Test Series"):
                client.get_series(_make_spec(label="Test Series"))

    def test_network_error_original_exception_is_chained(self, client):
        original = httpx.ConnectError("refused")
        with patch.object(client._http, "get", side_effect=original):
            with pytest.raises(OECDClientError) as exc_info:
                client.get_series(_make_spec())

        assert exc_info.value.__cause__ is original

    def test_http_error_original_exception_is_chained(self, client):
        error_resp = _error_response(502)
        original = httpx.HTTPStatusError(
            "502", request=MagicMock(), response=error_resp
        )
        error_resp.raise_for_status.side_effect = original
        with patch.object(client._http, "get", return_value=error_resp):
            with pytest.raises(OECDClientError) as exc_info:
                client.get_series(_make_spec())

        assert exc_info.value.__cause__ is original

    def test_csv_missing_obs_value_column_raises_oecd_client_error(self, client):
        with patch.object(
            client._http, "get", return_value=_ok_response(_MOCK_CSV_NO_OBS_VALUE)
        ):
            with pytest.raises(OECDClientError, match="OBS_VALUE"):
                client.get_series(_make_spec())

    def test_csv_missing_time_period_column_raises_oecd_client_error(self, client):
        with patch.object(
            client._http, "get", return_value=_ok_response(_MOCK_CSV_NO_TIME_PERIOD)
        ):
            with pytest.raises(OECDClientError, match="TIME_PERIOD"):
                client.get_series(_make_spec())


# ---------------------------------------------------------------------------
# TestOECDClientEmptyData — edge case: valid response with no observations
# ---------------------------------------------------------------------------


class TestOECDClientEmptyData:
    def test_no_observations_returns_empty_series(self, client):
        with patch.object(
            client._http, "get", return_value=_ok_response(_MOCK_CSV_EMPTY)
        ):
            result = client.get_series(_make_spec())

        assert isinstance(result.data, pd.Series)
        assert len(result.data) == 0


# ---------------------------------------------------------------------------
# TestOECDNormalizeTimePeriod — quarterly time period handling
# ---------------------------------------------------------------------------


class TestOECDNormalizeTimePeriod:
    def test_monthly_period_passes_through(self):
        from app.integrations.oecd_client import _normalize_time_period

        assert _normalize_time_period("2024-01") == "2024-01"

    def test_annual_period_passes_through(self):
        from app.integrations.oecd_client import _normalize_time_period

        assert _normalize_time_period("2024") == "2024"

    def test_q1_maps_to_january(self):
        from app.integrations.oecd_client import _normalize_time_period

        assert _normalize_time_period("2024-Q1") == "2024-01"

    def test_q2_maps_to_april(self):
        from app.integrations.oecd_client import _normalize_time_period

        assert _normalize_time_period("2024-Q2") == "2024-04"

    def test_q3_maps_to_july(self):
        from app.integrations.oecd_client import _normalize_time_period

        assert _normalize_time_period("2024-Q3") == "2024-07"

    def test_q4_maps_to_october(self):
        from app.integrations.oecd_client import _normalize_time_period

        assert _normalize_time_period("2024-Q4") == "2024-10"

    def test_quarterly_csv_parsed_to_datetime_index(self, client):
        quarterly_csv = """\
STRUCTURE,STRUCTURE_ID,ACTION,TIME_PERIOD,OBS_VALUE,OBS_STATUS
dataflow,test,A,2023-Q1,1.0,A
dataflow,test,A,2023-Q2,2.0,A
dataflow,test,A,2023-Q3,3.0,A
"""
        with patch.object(
            client._http, "get", return_value=_ok_response(quarterly_csv)
        ):
            result = client.get_series(_make_spec())

        assert len(result.data) == 3
        assert isinstance(result.data.index, pd.DatetimeIndex)
        assert result.data.index.is_monotonic_increasing
