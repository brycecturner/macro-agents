"""Unit tests for GET /debug/oecd-probe.

All OECD network calls are mocked — no real outbound requests are made.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.integrations.oecd_client import OECDSeriesResult, OECDSeriesSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC = OECDSeriesSpec(
    agency="OECD.SDD.STES",
    dataflow="DSD_STES@DF_FINMARK",
    version="4.0",
    dimension_key="EA19.M.IRSTCI.PA._Z._Z._Z._Z.N",
    label="ECB Policy Rate (Short-term Call Rate)",
)

_RETRIEVED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def _make_result(values: list[float], spec: OECDSeriesSpec = _SPEC) -> OECDSeriesResult:
    """Build an OECDSeriesResult with a simple DatetimeIndex series."""
    index = pd.date_range("2025-10-01", periods=len(values), freq="MS")
    return OECDSeriesResult(
        spec=spec,
        data=pd.Series(values, index=index),
        retrieved_at=_RETRIEVED_AT,
    )


def _make_empty_result(spec: OECDSeriesSpec = _SPEC) -> OECDSeriesResult:
    return OECDSeriesResult(
        spec=spec,
        data=pd.Series(dtype=float),
        retrieved_at=_RETRIEVED_AT,
    )


def _patch_oecd_client(side_effects: list) -> MagicMock:
    """Return a mock OECDClient whose get_series yields side_effects in order."""
    mock_client = MagicMock()
    mock_client.get_series.side_effect = side_effects
    return mock_client


# ---------------------------------------------------------------------------
# TestOECDProbeEndpoint
# ---------------------------------------------------------------------------


class TestOECDProbeEndpoint:
    def test_returns_200_when_all_series_ok(self, client: TestClient) -> None:
        mock = _patch_oecd_client([_make_result([1.5, 1.6, 1.7])] * 3)
        with patch("app.api.debug.OECDClient", return_value=mock):
            response = client.get("/debug/oecd-probe")
        assert response.status_code == 200

    def test_response_contains_probed_at_timestamp(self, client: TestClient) -> None:
        mock = _patch_oecd_client([_make_result([1.5])] * 3)
        with patch("app.api.debug.OECDClient", return_value=mock):
            body = client.get("/debug/oecd-probe").json()
        assert "probed_at" in body
        # Must be a parseable ISO timestamp
        datetime.fromisoformat(body["probed_at"])

    def test_response_contains_three_series(self, client: TestClient) -> None:
        mock = _patch_oecd_client([_make_result([1.5])] * 3)
        with patch("app.api.debug.OECDClient", return_value=mock):
            body = client.get("/debug/oecd-probe").json()
        assert len(body["series"]) == 3

    def test_all_ok_true_when_all_succeed(self, client: TestClient) -> None:
        mock = _patch_oecd_client([_make_result([1.5, 1.6])] * 3)
        with patch("app.api.debug.OECDClient", return_value=mock):
            body = client.get("/debug/oecd-probe").json()
        assert body["all_ok"] is True

    def test_all_ok_false_when_one_series_errors(self, client: TestClient) -> None:
        from app.integrations.oecd_client import OECDClientError

        mock = _patch_oecd_client(
            [
                _make_result([1.5]),
                OECDClientError("HTTP 422"),
                _make_result([98.5]),
            ]
        )
        with patch("app.api.debug.OECDClient", return_value=mock):
            body = client.get("/debug/oecd-probe").json()
        assert body["all_ok"] is False

    def test_ok_series_has_correct_status_and_value(self, client: TestClient) -> None:
        mock = _patch_oecd_client([_make_result([1.931671, 1.929287, 1.5])] * 3)
        with patch("app.api.debug.OECDClient", return_value=mock):
            body = client.get("/debug/oecd-probe").json()
        first = body["series"][0]
        assert first["status"] == "ok"
        assert first["latest_value"] == pytest.approx(1.5, rel=1e-4)
        assert first["observation_count"] == 3
        assert first["error"] is None

    def test_error_series_has_error_status_and_message(
        self, client: TestClient
    ) -> None:
        from app.integrations.oecd_client import OECDClientError

        mock = _patch_oecd_client(
            [
                OECDClientError("HTTP 422 Unprocessable Entity"),
                _make_result([2.1]),
                _make_result([98.5]),
            ]
        )
        with patch("app.api.debug.OECDClient", return_value=mock):
            body = client.get("/debug/oecd-probe").json()
        first = body["series"][0]
        assert first["status"] == "error"
        assert "422" in first["error"]
        assert first["latest_value"] is None
        assert first["observation_count"] is None

    def test_empty_series_has_empty_status(self, client: TestClient) -> None:
        mock = _patch_oecd_client(
            [
                _make_empty_result(),
                _make_result([2.1]),
                _make_result([98.5]),
            ]
        )
        with patch("app.api.debug.OECDClient", return_value=mock):
            body = client.get("/debug/oecd-probe").json()
        first = body["series"][0]
        assert first["status"] == "empty"
        assert first["observation_count"] == 0
        assert first["latest_value"] is None

    def test_series_entries_include_dimension_key(self, client: TestClient) -> None:
        mock = _patch_oecd_client([_make_result([1.5])] * 3)
        with patch("app.api.debug.OECDClient", return_value=mock):
            body = client.get("/debug/oecd-probe").json()
        for entry in body["series"]:
            assert "dimension_key" in entry
            assert "label" in entry
            assert "agency" in entry
            assert "dataflow" in entry
            assert "version" in entry

    def test_oecd_client_is_closed_after_probe(self, client: TestClient) -> None:
        mock = _patch_oecd_client([_make_result([1.5])] * 3)
        with patch("app.api.debug.OECDClient", return_value=mock):
            client.get("/debug/oecd-probe")
        mock.close.assert_called_once()

    def test_oecd_client_closed_even_when_all_series_error(
        self, client: TestClient
    ) -> None:
        from app.integrations.oecd_client import OECDClientError

        mock = _patch_oecd_client([OECDClientError("boom")] * 3)
        with patch("app.api.debug.OECDClient", return_value=mock):
            client.get("/debug/oecd-probe")
        mock.close.assert_called_once()

    def test_all_ok_false_when_series_is_empty(self, client: TestClient) -> None:
        mock = _patch_oecd_client(
            [
                _make_result([1.5]),
                _make_empty_result(),
                _make_result([98.5]),
            ]
        )
        with patch("app.api.debug.OECDClient", return_value=mock):
            body = client.get("/debug/oecd-probe").json()
        assert body["all_ok"] is False
