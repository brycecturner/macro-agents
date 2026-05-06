"""Unit tests for IBKRClient.

All tests mock the underlying httpx.Client so no real network calls are made.
The primary acceptance criterion for Ticket 006 is that routing between paper
and real accounts is governed exclusively by pod_settings.trading_mode.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.core.pod_settings import PodSettings
from app.integrations.ibkr_client import (
    IBKRAccountSummary,
    IBKRClient,
    IBKRClientError,
    IBKROrder,
    IBKROrderResult,
    IBKRPosition,
    IBKRPriceHistory,
)
from app.models.enums import KillAuthority, OrderType, TradingMode

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------

REAL_ACCOUNT = "U1234567"
PAPER_ACCOUNT = "DU9876543"
BASE_URL = "https://localhost:5000"


# ---------------------------------------------------------------------------
# Mock API payload factories
# ---------------------------------------------------------------------------

_POSITION_ITEM = {
    "ticker": "TLT",
    "conid": 12345,
    "position": 100.0,
    "mktPrice": 95.50,
    "mktValue": 9550.0,
    "avgCost": 90.0,
    "unrealizedPnl": 550.0,
}

_ACCOUNT_SUMMARY_PAYLOAD = {
    "netliquidation": {"amount": 500_000.0},
    "buyingpower": {"amount": 250_000.0},
    "totalcashvalue": {"amount": 150_000.0},
}

_PRICE_HISTORY_PAYLOAD = {
    "data": [
        {
            "t": 1_700_000_000_000,
            "o": 94.0,
            "h": 96.0,
            "l": 93.5,
            "c": 95.5,
            "v": 1_000_000,
        },
        {
            "t": 1_700_086_400_000,
            "o": 95.5,
            "h": 97.0,
            "l": 95.0,
            "c": 96.0,
            "v": 1_200_000,
        },
    ]
}

_CONID_SEARCH_PAYLOAD = [
    {"conid": "12345", "assetClass": "ETF", "symbol": "TLT"},
]

_ORDER_RESULT_PAYLOAD = [
    {"order_id": "999001", "order_status": "PreSubmitted"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ok_response(payload: dict | list) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def _error_response(status_code: int) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"HTTP {status_code}", request=MagicMock(), response=resp
    )
    return resp


def _pod_settings(trading_mode: TradingMode) -> PodSettings:
    return PodSettings(
        pod_id=uuid.uuid4(),
        trading_mode=trading_mode,
        target_vol_per_position=0.05,
        max_position_pct=0.25,
        rebalance_threshold_pct=0.01,
        rebalance_day=0,
        intake_timeout_hours=24,
        kill_authority_default=KillAuthority.alert_only,
        vol_lookback_days=60,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def paper_client() -> IBKRClient:
    return IBKRClient(
        base_url=BASE_URL,
        account_id=REAL_ACCOUNT,
        paper_account_id=PAPER_ACCOUNT,
        pod_settings=_pod_settings(TradingMode.paper),
    )


@pytest.fixture
def real_client() -> IBKRClient:
    return IBKRClient(
        base_url=BASE_URL,
        account_id=REAL_ACCOUNT,
        paper_account_id=PAPER_ACCOUNT,
        pod_settings=_pod_settings(TradingMode.real),
    )


# ---------------------------------------------------------------------------
# TestIBKRClientTradingModeRouting — primary acceptance criterion
# ---------------------------------------------------------------------------


class TestIBKRClientTradingModeRouting:
    """Confirm that every account-scoped request routes to the correct account
    based on pod_settings.trading_mode, read at call time."""

    def test_get_positions_paper_uses_paper_account(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response([_POSITION_ITEM]))
        with patch.object(paper_client._http, "request", mock_request):
            paper_client.get_positions()

        url = mock_request.call_args.args[1]
        assert PAPER_ACCOUNT in url
        assert REAL_ACCOUNT not in url

    def test_get_positions_real_uses_real_account(self, real_client):
        mock_request = MagicMock(return_value=_ok_response([_POSITION_ITEM]))
        with patch.object(real_client._http, "request", mock_request):
            real_client.get_positions()

        url = mock_request.call_args.args[1]
        assert REAL_ACCOUNT in url
        assert PAPER_ACCOUNT not in url

    def test_get_account_summary_paper_uses_paper_account(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response(_ACCOUNT_SUMMARY_PAYLOAD))
        with patch.object(paper_client._http, "request", mock_request):
            paper_client.get_account_summary()

        url = mock_request.call_args.args[1]
        assert PAPER_ACCOUNT in url

    def test_get_account_summary_real_uses_real_account(self, real_client):
        mock_request = MagicMock(return_value=_ok_response(_ACCOUNT_SUMMARY_PAYLOAD))
        with patch.object(real_client._http, "request", mock_request):
            real_client.get_account_summary()

        url = mock_request.call_args.args[1]
        assert REAL_ACCOUNT in url
        assert PAPER_ACCOUNT not in url

    def test_submit_order_paper_uses_paper_account(self, paper_client):
        order = IBKROrder(
            symbol="TLT",
            order_type=OrderType.market,
            side="BUY",
            quantity=100,
        )
        # Three requests: iserver session init + conid lookup + order submission
        mock_request = MagicMock(
            side_effect=[
                _ok_response({"accounts": [PAPER_ACCOUNT]}),
                _ok_response(_CONID_SEARCH_PAYLOAD),
                _ok_response(_ORDER_RESULT_PAYLOAD),
            ]
        )
        with patch.object(paper_client._http, "request", mock_request):
            paper_client.submit_order(order)

        order_url = mock_request.call_args_list[2].args[1]
        assert PAPER_ACCOUNT in order_url
        assert REAL_ACCOUNT not in order_url

    def test_submit_order_real_uses_real_account(self, real_client):
        order = IBKROrder(
            symbol="TLT",
            order_type=OrderType.market,
            side="BUY",
            quantity=100,
        )
        mock_request = MagicMock(
            side_effect=[
                _ok_response({"accounts": [REAL_ACCOUNT]}),
                _ok_response(_CONID_SEARCH_PAYLOAD),
                _ok_response(_ORDER_RESULT_PAYLOAD),
            ]
        )
        with patch.object(real_client._http, "request", mock_request):
            real_client.submit_order(order)

        order_url = mock_request.call_args_list[2].args[1]
        assert REAL_ACCOUNT in order_url
        assert PAPER_ACCOUNT not in order_url

    def test_cancel_order_paper_uses_paper_account(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response({"msg": "cancelled"}))
        with patch.object(paper_client._http, "request", mock_request):
            paper_client.cancel_order("999001")

        url = mock_request.call_args.args[1]
        assert PAPER_ACCOUNT in url

    def test_cancel_order_real_uses_real_account(self, real_client):
        mock_request = MagicMock(return_value=_ok_response({"msg": "cancelled"}))
        with patch.object(real_client._http, "request", mock_request):
            real_client.cancel_order("999001")

        url = mock_request.call_args.args[1]
        assert REAL_ACCOUNT in url
        assert PAPER_ACCOUNT not in url

    def test_position_result_carries_correct_account_id(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response([_POSITION_ITEM]))
        with patch.object(paper_client._http, "request", mock_request):
            positions = paper_client.get_positions()

        assert positions[0].account_id == PAPER_ACCOUNT

    def test_account_summary_result_carries_correct_account_id(self, real_client):
        mock_request = MagicMock(return_value=_ok_response(_ACCOUNT_SUMMARY_PAYLOAD))
        with patch.object(real_client._http, "request", mock_request):
            summary = real_client.get_account_summary()

        assert summary.account_id == REAL_ACCOUNT


# ---------------------------------------------------------------------------
# TestIBKRClientGetPositions
# ---------------------------------------------------------------------------


class TestIBKRClientGetPositions:
    def test_returns_list_of_ibkr_positions(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response([_POSITION_ITEM]))
        with patch.object(paper_client._http, "request", mock_request):
            result = paper_client.get_positions()

        assert isinstance(result, list)
        assert all(isinstance(p, IBKRPosition) for p in result)

    def test_returns_correct_position_fields(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response([_POSITION_ITEM]))
        with patch.object(paper_client._http, "request", mock_request):
            positions = paper_client.get_positions()

        p = positions[0]
        assert p.symbol == "TLT"
        assert p.conid == 12345
        assert p.position == pytest.approx(100.0)
        assert p.market_price == pytest.approx(95.50)
        assert p.market_value == pytest.approx(9550.0)
        assert p.avg_cost == pytest.approx(90.0)
        assert p.unrealized_pnl == pytest.approx(550.0)

    def test_retrieved_at_is_utc_datetime(self, paper_client):
        before = datetime.now(UTC)
        mock_request = MagicMock(return_value=_ok_response([_POSITION_ITEM]))
        with patch.object(paper_client._http, "request", mock_request):
            positions = paper_client.get_positions()
        after = datetime.now(UTC)

        assert before <= positions[0].retrieved_at <= after
        assert positions[0].retrieved_at.tzinfo == UTC

    def test_empty_positions_returns_empty_list(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response([]))
        with patch.object(paper_client._http, "request", mock_request):
            result = paper_client.get_positions()

        assert result == []

    def test_multiple_positions_all_returned(self, paper_client):
        items = [_POSITION_ITEM, {**_POSITION_ITEM, "ticker": "GLD", "conid": 99999}]
        mock_request = MagicMock(return_value=_ok_response(items))
        with patch.object(paper_client._http, "request", mock_request):
            result = paper_client.get_positions()

        assert len(result) == 2

    def test_non_list_response_raises_error(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response({"unexpected": "dict"}))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError):
                paper_client.get_positions()

    def test_missing_field_in_position_raises_error(self, paper_client):
        bad_item = {"ticker": "TLT", "conid": 12345}  # missing position, prices
        mock_request = MagicMock(return_value=_ok_response([bad_item]))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError):
                paper_client.get_positions()


# ---------------------------------------------------------------------------
# TestIBKRClientGetAccountSummary
# ---------------------------------------------------------------------------


class TestIBKRClientGetAccountSummary:
    def test_returns_ibkr_account_summary(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response(_ACCOUNT_SUMMARY_PAYLOAD))
        with patch.object(paper_client._http, "request", mock_request):
            result = paper_client.get_account_summary()

        assert isinstance(result, IBKRAccountSummary)

    def test_net_liquidation_parsed_correctly(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response(_ACCOUNT_SUMMARY_PAYLOAD))
        with patch.object(paper_client._http, "request", mock_request):
            result = paper_client.get_account_summary()

        assert result.net_liquidation == pytest.approx(500_000.0)

    def test_buying_power_parsed_correctly(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response(_ACCOUNT_SUMMARY_PAYLOAD))
        with patch.object(paper_client._http, "request", mock_request):
            result = paper_client.get_account_summary()

        assert result.buying_power == pytest.approx(250_000.0)

    def test_cash_balance_parsed_correctly(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response(_ACCOUNT_SUMMARY_PAYLOAD))
        with patch.object(paper_client._http, "request", mock_request):
            result = paper_client.get_account_summary()

        assert result.cash_balance == pytest.approx(150_000.0)

    def test_retrieved_at_is_utc(self, paper_client):
        before = datetime.now(UTC)
        mock_request = MagicMock(return_value=_ok_response(_ACCOUNT_SUMMARY_PAYLOAD))
        with patch.object(paper_client._http, "request", mock_request):
            result = paper_client.get_account_summary()
        after = datetime.now(UTC)

        assert before <= result.retrieved_at <= after

    def test_non_dict_response_raises_error(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response([]))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError):
                paper_client.get_account_summary()

    def test_missing_key_in_summary_raises_error(self, paper_client):
        bad_payload = {"netliquidation": {"amount": 100.0}}  # missing other keys
        mock_request = MagicMock(return_value=_ok_response(bad_payload))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError):
                paper_client.get_account_summary()


# ---------------------------------------------------------------------------
# TestIBKRClientGetPriceHistory
# ---------------------------------------------------------------------------


class TestIBKRClientGetPriceHistory:
    def _make_client_with_cached_conid(self) -> IBKRClient:
        """Return a paper client with TLT conid already cached to skip the lookup."""
        client = IBKRClient(
            base_url=BASE_URL,
            account_id=REAL_ACCOUNT,
            paper_account_id=PAPER_ACCOUNT,
            pod_settings=_pod_settings(TradingMode.paper),
        )
        client._conid_cache["TLT"] = 12345
        return client

    def test_returns_ibkr_price_history(self):
        client = self._make_client_with_cached_conid()
        mock_request = MagicMock(return_value=_ok_response(_PRICE_HISTORY_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.get_price_history("TLT", "1y", "1d")

        assert isinstance(result, IBKRPriceHistory)

    def test_symbol_period_barsize_stored_on_result(self):
        client = self._make_client_with_cached_conid()
        mock_request = MagicMock(return_value=_ok_response(_PRICE_HISTORY_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.get_price_history("TLT", "1y", "1d")

        assert result.symbol == "TLT"
        assert result.period == "1y"
        assert result.bar_size == "1d"

    def test_bars_count_matches_payload(self):
        client = self._make_client_with_cached_conid()
        mock_request = MagicMock(return_value=_ok_response(_PRICE_HISTORY_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.get_price_history("TLT", "1y", "1d")

        assert len(result.bars) == 2

    def test_bar_ohlcv_values_parsed_correctly(self):
        client = self._make_client_with_cached_conid()
        mock_request = MagicMock(return_value=_ok_response(_PRICE_HISTORY_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.get_price_history("TLT", "1y", "1d")

        bar = result.bars[0]
        assert bar.open == pytest.approx(94.0)
        assert bar.high == pytest.approx(96.0)
        assert bar.low == pytest.approx(93.5)
        assert bar.close == pytest.approx(95.5)
        assert bar.volume == pytest.approx(1_000_000)

    def test_bar_timestamps_are_utc(self):
        client = self._make_client_with_cached_conid()
        mock_request = MagicMock(return_value=_ok_response(_PRICE_HISTORY_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.get_price_history("TLT", "1y", "1d")

        for bar in result.bars:
            assert bar.timestamp.tzinfo == UTC

    def test_retrieved_at_is_utc(self):
        client = self._make_client_with_cached_conid()
        before = datetime.now(UTC)
        mock_request = MagicMock(return_value=_ok_response(_PRICE_HISTORY_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.get_price_history("TLT", "1y", "1d")
        after = datetime.now(UTC)

        assert before <= result.retrieved_at <= after

    def test_conid_lookup_called_for_unknown_symbol(self):
        client = IBKRClient(
            base_url=BASE_URL,
            account_id=REAL_ACCOUNT,
            paper_account_id=PAPER_ACCOUNT,
            pod_settings=_pod_settings(TradingMode.paper),
        )
        mock_request = MagicMock(
            side_effect=[
                _ok_response({"accounts": [PAPER_ACCOUNT]}),
                _ok_response(_CONID_SEARCH_PAYLOAD),
                _ok_response(_PRICE_HISTORY_PAYLOAD),
            ]
        )
        with patch.object(client._http, "request", mock_request):
            client.get_price_history("TLT", "1y", "1d")

        # iserver session init + conid lookup + history = 3 requests
        assert mock_request.call_count == 3

    def test_conid_cached_after_first_lookup(self):
        client = IBKRClient(
            base_url=BASE_URL,
            account_id=REAL_ACCOUNT,
            paper_account_id=PAPER_ACCOUNT,
            pod_settings=_pod_settings(TradingMode.paper),
        )
        mock_request = MagicMock(
            side_effect=[
                _ok_response({"accounts": [PAPER_ACCOUNT]}),
                _ok_response(_CONID_SEARCH_PAYLOAD),
                _ok_response(_PRICE_HISTORY_PAYLOAD),
                _ok_response(_PRICE_HISTORY_PAYLOAD),
            ]
        )
        with patch.object(client._http, "request", mock_request):
            client.get_price_history("TLT", "1y", "1d")
            client.get_price_history("TLT", "5y", "1w")

        # First call: session init + conid lookup + history = 3 requests
        # Second call: session and conid both cached, only history = 1 request
        assert mock_request.call_count == 4

    def test_missing_data_key_raises_error(self):
        client = self._make_client_with_cached_conid()
        mock_request = MagicMock(return_value=_ok_response({"other": "stuff"}))
        with patch.object(client._http, "request", mock_request):
            with pytest.raises(IBKRClientError):
                client.get_price_history("TLT", "1y", "1d")

    def test_symbol_not_found_raises_error(self):
        client = IBKRClient(
            base_url=BASE_URL,
            account_id=REAL_ACCOUNT,
            paper_account_id=PAPER_ACCOUNT,
            pod_settings=_pod_settings(TradingMode.paper),
        )
        mock_request = MagicMock(return_value=_ok_response([]))
        with patch.object(client._http, "request", mock_request):
            with pytest.raises(IBKRClientError, match="No contracts found"):
                client.get_price_history("INVALID", "1y", "1d")


# ---------------------------------------------------------------------------
# TestIBKRClientSubmitOrder
# ---------------------------------------------------------------------------


class TestIBKRClientSubmitOrder:
    def _client_with_conid(self, trading_mode: TradingMode) -> IBKRClient:
        client = IBKRClient(
            base_url=BASE_URL,
            account_id=REAL_ACCOUNT,
            paper_account_id=PAPER_ACCOUNT,
            pod_settings=_pod_settings(trading_mode),
        )
        client._conid_cache["TLT"] = 12345
        return client

    def test_market_order_returns_order_result(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT", order_type=OrderType.market, side="BUY", quantity=100
        )
        mock_request = MagicMock(return_value=_ok_response(_ORDER_RESULT_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.submit_order(order)

        assert isinstance(result, IBKROrderResult)

    def test_order_result_has_order_id(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT", order_type=OrderType.market, side="BUY", quantity=100
        )
        mock_request = MagicMock(return_value=_ok_response(_ORDER_RESULT_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.submit_order(order)

        assert result.order_id == "999001"

    def test_order_result_has_status(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT", order_type=OrderType.market, side="BUY", quantity=100
        )
        mock_request = MagicMock(return_value=_ok_response(_ORDER_RESULT_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.submit_order(order)

        assert result.status == "PreSubmitted"

    def test_order_result_symbol_matches(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT", order_type=OrderType.market, side="BUY", quantity=100
        )
        mock_request = MagicMock(return_value=_ok_response(_ORDER_RESULT_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.submit_order(order)

        assert result.symbol == "TLT"

    def test_submitted_at_is_utc(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT", order_type=OrderType.market, side="BUY", quantity=100
        )
        before = datetime.now(UTC)
        mock_request = MagicMock(return_value=_ok_response(_ORDER_RESULT_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            result = client.submit_order(order)
        after = datetime.now(UTC)

        assert before <= result.submitted_at <= after

    def test_limit_order_includes_price_in_body(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT",
            order_type=OrderType.limit,
            side="BUY",
            quantity=100,
            price=95.0,
        )
        mock_request = MagicMock(return_value=_ok_response(_ORDER_RESULT_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            client.submit_order(order)

        body = mock_request.call_args.kwargs["json"]
        assert body["orders"][0]["price"] == pytest.approx(95.0)

    def test_limit_order_without_price_raises_error(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT",
            order_type=OrderType.limit,
            side="BUY",
            quantity=100,
            price=None,
        )
        with pytest.raises(IBKRClientError, match="Price is required"):
            client.submit_order(order)

    def test_order_body_contains_correct_side(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT", order_type=OrderType.market, side="SELL", quantity=50
        )
        mock_request = MagicMock(return_value=_ok_response(_ORDER_RESULT_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            client.submit_order(order)

        body = mock_request.call_args.kwargs["json"]
        assert body["orders"][0]["side"] == "SELL"

    def test_order_body_contains_correct_quantity(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT", order_type=OrderType.market, side="BUY", quantity=250
        )
        mock_request = MagicMock(return_value=_ok_response(_ORDER_RESULT_PAYLOAD))
        with patch.object(client._http, "request", mock_request):
            client.submit_order(order)

        body = mock_request.call_args.kwargs["json"]
        assert body["orders"][0]["quantity"] == 250

    def test_empty_order_response_raises_error(self):
        client = self._client_with_conid(TradingMode.paper)
        order = IBKROrder(
            symbol="TLT", order_type=OrderType.market, side="BUY", quantity=100
        )
        mock_request = MagicMock(return_value=_ok_response([]))
        with patch.object(client._http, "request", mock_request):
            with pytest.raises(IBKRClientError):
                client.submit_order(order)


# ---------------------------------------------------------------------------
# TestIBKRClientCancelOrder
# ---------------------------------------------------------------------------


class TestIBKRClientCancelOrder:
    def test_cancel_succeeds_without_error(self, paper_client):
        mock_request = MagicMock(
            return_value=_ok_response({"msg": "Request was submitted"})
        )
        with patch.object(paper_client._http, "request", mock_request):
            paper_client.cancel_order("999001")  # should not raise

    def test_cancel_sends_delete_method(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response({"msg": "ok"}))
        with patch.object(paper_client._http, "request", mock_request):
            paper_client.cancel_order("999001")

        assert mock_request.call_args.args[0] == "DELETE"

    def test_cancel_url_contains_order_id(self, paper_client):
        mock_request = MagicMock(return_value=_ok_response({"msg": "ok"}))
        with patch.object(paper_client._http, "request", mock_request):
            paper_client.cancel_order("999001")

        url = mock_request.call_args.args[1]
        assert "999001" in url

    def test_cancel_http_error_raises_ibkr_error(self, paper_client):
        mock_request = MagicMock(return_value=_error_response(404))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError, match="HTTP 404"):
                paper_client.cancel_order("999001")


# ---------------------------------------------------------------------------
# TestIBKRClientErrors — typed exceptions, status codes, exception chaining
# ---------------------------------------------------------------------------


class TestIBKRClientErrors:
    def test_http_4xx_raises_ibkr_client_error(self, paper_client):
        mock_request = MagicMock(return_value=_error_response(403))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError, match="HTTP 403"):
                paper_client.get_positions()

    def test_http_5xx_raises_ibkr_client_error(self, paper_client):
        mock_request = MagicMock(return_value=_error_response(503))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError, match="HTTP 503"):
                paper_client.get_positions()

    def test_http_error_status_code_attribute_set(self, paper_client):
        mock_request = MagicMock(return_value=_error_response(404))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError) as exc_info:
                paper_client.get_positions()

        assert exc_info.value.status_code == 404

    def test_network_error_raises_ibkr_client_error(self, paper_client):
        mock_request = MagicMock(side_effect=httpx.ConnectError("refused"))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError, match="ConnectError"):
                paper_client.get_positions()

    def test_network_error_status_code_is_none(self, paper_client):
        mock_request = MagicMock(side_effect=httpx.ConnectError("refused"))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError) as exc_info:
                paper_client.get_positions()

        assert exc_info.value.status_code is None

    def test_timeout_raises_ibkr_client_error(self, paper_client):
        mock_request = MagicMock(side_effect=httpx.TimeoutException("timeout"))
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError):
                paper_client.get_positions()

    def test_http_error_original_exception_chained(self, paper_client):
        error_resp = _error_response(500)
        original = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=error_resp
        )
        error_resp.raise_for_status.side_effect = original
        with patch.object(paper_client._http, "request", return_value=error_resp):
            with pytest.raises(IBKRClientError) as exc_info:
                paper_client.get_positions()

        assert exc_info.value.__cause__ is original

    def test_network_error_original_exception_chained(self, paper_client):
        original = httpx.ConnectError("refused")
        mock_request = MagicMock(side_effect=original)
        with patch.object(paper_client._http, "request", mock_request):
            with pytest.raises(IBKRClientError) as exc_info:
                paper_client.get_positions()

        assert exc_info.value.__cause__ is original

    def test_ibkr_client_error_is_exception_subclass(self):
        err = IBKRClientError("test error", status_code=500)
        assert isinstance(err, Exception)
        assert err.status_code == 500
        assert str(err) == "test error"
