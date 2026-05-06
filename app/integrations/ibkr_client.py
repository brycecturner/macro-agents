import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from app.core.pod_settings import PodSettings
from app.models.enums import OrderType, TradingMode

logger = logging.getLogger(__name__)


class IBKRClientError(Exception):
    """Raised when the IBKR Client Portal API returns an error or is unreachable."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class IBKRPosition:
    """A single position returned by IBKR.

    Citation format: ``IBKR:{symbol} position, {retrieved_at}``
    """

    account_id: str
    symbol: str
    conid: int
    position: float  # positive = long, negative = short
    market_price: float
    market_value: float
    avg_cost: float
    unrealized_pnl: float
    retrieved_at: datetime


@dataclass
class IBKRAccountSummary:
    """Account-level summary from IBKR — NAV, buying power, and cash balance.

    Citation format: ``IBKR:{account_id} account_summary, {retrieved_at}``
    """

    account_id: str
    net_liquidation: float
    buying_power: float
    cash_balance: float
    retrieved_at: datetime


@dataclass
class IBKRBar:
    """A single OHLCV price bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class IBKRPriceHistory:
    """OHLCV price history for an instrument.

    Citation format: ``IBKR:{symbol} price_history, {retrieved_at}``
    """

    symbol: str
    period: str
    bar_size: str
    bars: list[IBKRBar]
    retrieved_at: datetime


@dataclass
class IBKROrder:
    """An order to submit to IBKR.

    ``side`` must be ``"BUY"`` or ``"SELL"`` — the service layer is responsible
    for translating thesis direction (long/short) to the correct side.
    ``price`` is required when ``order_type`` is ``OrderType.limit``.
    """

    symbol: str
    order_type: OrderType
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float | None = None


@dataclass
class IBKROrderResult:
    """Result of a submitted order."""

    order_id: str
    symbol: str
    status: str
    submitted_at: datetime


class IBKRClient:
    """Wraps the IBKR Client Portal REST API.

    Routes all requests to the paper or real IBKR account based on
    ``pod_settings.trading_mode``, which is read at call time — never cached
    at construction. This ensures trading_mode changes in pod_configs are
    reflected immediately without restarting the application.

    The IBKR Client Portal gateway uses a self-signed TLS certificate, so SSL
    verification is disabled. The gateway manages session authentication; this
    client does not send explicit auth headers.

    Instantiate once per request. ``_conid_cache`` avoids redundant symbol
    lookups within a single request lifecycle.

    Args:
        base_url: IBKR Client Portal gateway base URL
            (e.g. ``"https://localhost:5000"``).
        account_id: Real (live) IBKR account ID.
        paper_account_id: Paper trading IBKR account ID.
        pod_settings: Operational config loaded fresh at request time.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        account_id: str,
        paper_account_id: str,
        pod_settings: PodSettings,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._account_id = account_id
        self._paper_account_id = paper_account_id
        self._pod_settings = pod_settings
        # verify=False: IBKR Client Portal gateway uses a self-signed certificate
        self._http = httpx.Client(timeout=timeout, verify=False)
        self._conid_cache: dict[str, int] = {}
        self._iserver_initialised = False

    @property
    def _active_account_id(self) -> str:
        """Account ID for the current trading_mode, evaluated at call time."""
        if self._pod_settings.trading_mode == TradingMode.paper:
            return self._paper_account_id
        return self._account_id

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Make a request to the IBKR Client Portal API.

        Raises:
            IBKRClientError: On HTTP errors or network failures, with
                ``status_code`` populated for HTTP errors.
        """
        url = f"{self._base_url}{path}"
        try:
            response = self._http.request(method, url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise IBKRClientError(
                f"IBKR API returned HTTP {exc.response.status_code} "
                f"for {method} {path}: {exc}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise IBKRClientError(
                f"Failed to reach IBKR API for {method} {path}: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        return response.json()

    def _ensure_iserver_session(self) -> None:
        """Initialise the iserver session if not already done.

        The Client Portal gateway requires a call to /iserver/accounts before
        any other /iserver/ endpoint will respond. Without it every iserver
        request returns 404. Called lazily on the first iserver operation.
        """
        if self._iserver_initialised:
            return
        self._request("GET", "/v1/api/iserver/accounts")
        self._iserver_initialised = True
        logger.debug("iserver session initialised")

    def _get_conid(self, symbol: str) -> int:
        """Resolve the IBKR contract ID (conid) for a ticker symbol.

        Results are cached per client instance to avoid redundant lookups
        within a single request lifecycle.

        Raises:
            IBKRClientError: If no contract is found for the symbol.
        """
        self._ensure_iserver_session()
        if symbol in self._conid_cache:
            return self._conid_cache[symbol]

        payload = self._request(
            "GET",
            "/v1/api/iserver/secdef/search",
            params={"symbol": symbol},
        )

        if not isinstance(payload, list) or not payload:
            raise IBKRClientError(f"No contracts found for symbol {symbol!r}")

        # Prefer ETF/STK asset class; fall back to first result
        conid: int | None = None
        for contract in payload:
            if contract.get("assetClass") in ("STK", "ETF", None):
                conid = int(contract["conid"])
                break
        if conid is None:
            conid = int(payload[0]["conid"])

        self._conid_cache[symbol] = conid
        logger.debug("Resolved conid for %s: %d", symbol, conid)
        return conid

    def get_positions(self) -> list[IBKRPosition]:
        """Fetch current positions from IBKR for the active account.

        Routes to paper or real account based on current trading_mode.

        Returns:
            List of :class:`IBKRPosition` objects. Empty list if no positions
            are open.

        Raises:
            IBKRClientError: If the API call fails or the response is malformed.
        """
        account_id = self._active_account_id
        retrieved_at = datetime.now(tz=UTC)

        payload = self._request("GET", f"/v1/api/portfolio/{account_id}/positions/0")

        if not isinstance(payload, list):
            raise IBKRClientError(
                f"Unexpected positions response format for account {account_id!r}"
            )

        positions: list[IBKRPosition] = []
        for item in payload:
            try:
                positions.append(
                    IBKRPosition(
                        account_id=account_id,
                        symbol=str(item["ticker"]),
                        conid=int(item["conid"]),
                        position=float(item["position"]),
                        market_price=float(item["mktPrice"]),
                        market_value=float(item["mktValue"]),
                        avg_cost=float(item["avgCost"]),
                        unrealized_pnl=float(item["unrealizedPnl"]),
                        retrieved_at=retrieved_at,
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise IBKRClientError(
                    f"Failed to parse position item for account {account_id!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        logger.debug(
            "Fetched %d positions for account %s (trading_mode=%s)",
            len(positions),
            account_id,
            self._pod_settings.trading_mode.value,
        )
        return positions

    def get_account_summary(self) -> IBKRAccountSummary:
        """Fetch account summary (NAV, buying power, cash) from IBKR.

        Routes to paper or real account based on current trading_mode.

        Returns:
            :class:`IBKRAccountSummary` with net liquidation value, buying
            power, and total cash balance.

        Raises:
            IBKRClientError: If the API call fails or the response is malformed.
        """
        account_id = self._active_account_id
        retrieved_at = datetime.now(tz=UTC)

        payload = self._request("GET", f"/v1/api/portfolio/{account_id}/summary")

        if not isinstance(payload, dict):
            raise IBKRClientError(
                f"Unexpected account summary response format for account {account_id!r}"
            )

        try:
            return IBKRAccountSummary(
                account_id=account_id,
                net_liquidation=float(payload["netliquidation"]["amount"]),
                buying_power=float(payload["buyingpower"]["amount"]),
                cash_balance=float(payload["totalcashvalue"]["amount"]),
                retrieved_at=retrieved_at,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise IBKRClientError(
                f"Failed to parse account summary for {account_id!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def get_price_history(
        self,
        symbol: str,
        period: str,
        bar_size: str,
    ) -> IBKRPriceHistory:
        """Fetch OHLCV price history for a symbol from IBKR.

        Args:
            symbol: ETF ticker (e.g. ``"TLT"``, ``"SPY"``).
            period: History window (e.g. ``"1y"``, ``"5y"``, ``"6m"``).
            bar_size: Bar granularity (e.g. ``"1d"``, ``"1w"``).

        Returns:
            :class:`IBKRPriceHistory` containing the bars and a retrieval
            timestamp suitable for citation.

        Raises:
            IBKRClientError: If the symbol is not found or the API call fails.
        """
        conid = self._get_conid(symbol)
        retrieved_at = datetime.now(tz=UTC)

        payload = self._request(
            "GET",
            "/v1/api/iserver/marketdata/history",
            params={"conid": conid, "period": period, "bar": bar_size},
        )

        if not isinstance(payload, dict) or "data" not in payload:
            raise IBKRClientError(
                f"Unexpected price history response format for symbol {symbol!r}"
            )

        bars: list[IBKRBar] = []
        for item in payload["data"]:
            try:
                # IBKR timestamps are milliseconds since epoch
                bars.append(
                    IBKRBar(
                        timestamp=datetime.fromtimestamp(int(item["t"]) / 1000, tz=UTC),
                        open=float(item["o"]),
                        high=float(item["h"]),
                        low=float(item["l"]),
                        close=float(item["c"]),
                        volume=float(item.get("v", 0)),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise IBKRClientError(
                    f"Failed to parse price bar for {symbol!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

        logger.debug(
            "Fetched %d bars for %s (period=%s, bar_size=%s)",
            len(bars),
            symbol,
            period,
            bar_size,
        )
        return IBKRPriceHistory(
            symbol=symbol,
            period=period,
            bar_size=bar_size,
            bars=bars,
            retrieved_at=retrieved_at,
        )

    def submit_order(self, order: IBKROrder) -> IBKROrderResult:
        """Submit an order to IBKR.

        Routes to paper or real account based on current trading_mode.
        Limit orders require ``order.price`` to be set.

        Args:
            order: :class:`IBKROrder` describing the order to submit.

        Returns:
            :class:`IBKROrderResult` with the assigned order ID and status.

        Raises:
            IBKRClientError: If ``price`` is missing on a limit order, if the
                order is rejected, or if the API call fails.
        """
        if order.order_type == OrderType.limit and order.price is None:
            raise IBKRClientError(
                f"Price is required for limit orders (symbol={order.symbol!r})"
            )

        account_id = self._active_account_id
        conid = self._get_conid(order.symbol)

        order_body: dict[str, Any] = {
            "orders": [
                {
                    "acctId": account_id,
                    "conid": conid,
                    "orderType": order.order_type.value.upper(),
                    "side": order.side.upper(),
                    "quantity": order.quantity,
                    "tif": "DAY",
                }
            ]
        }
        if order.order_type == OrderType.limit:
            order_body["orders"][0]["price"] = order.price

        payload = self._request(
            "POST",
            f"/v1/api/iserver/account/{account_id}/orders",
            json=order_body,
        )

        if not isinstance(payload, list) or not payload:
            raise IBKRClientError(
                f"Unexpected order submission response for symbol {order.symbol!r}"
            )

        result = payload[0]
        try:
            order_id = str(
                result.get("order_id") or result.get("orderId") or result["id"]
            )
            status = str(
                result.get("order_status") or result.get("status", "submitted")
            )
        except (KeyError, TypeError) as exc:
            raise IBKRClientError(
                f"Failed to parse order result for {order.symbol!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        logger.debug(
            "Order submitted: symbol=%s, side=%s, qty=%s, order_id=%s (account=%s)",
            order.symbol,
            order.side,
            order.quantity,
            order_id,
            account_id,
        )
        return IBKROrderResult(
            order_id=order_id,
            symbol=order.symbol,
            status=status,
            submitted_at=datetime.now(tz=UTC),
        )

    def cancel_order(self, order_id: str) -> None:
        """Cancel an outstanding order on IBKR.

        Routes to paper or real account based on current trading_mode.

        Args:
            order_id: The order ID returned by :meth:`submit_order`.

        Raises:
            IBKRClientError: If the cancellation fails or the order is not found.
        """
        account_id = self._active_account_id
        self._request(
            "DELETE",
            f"/v1/api/iserver/account/{account_id}/order/{order_id}",
        )
        logger.debug("Order cancelled: order_id=%s (account=%s)", order_id, account_id)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()
