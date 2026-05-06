"""IBKR Client Portal connectivity smoke test.

Tests all core IBKRClient features against a running Client Portal gateway:
  1. conid resolution (symbol lookup)
  2. price history — SPY, 1 month of daily bars
  3. account summary — NAV, buying power, cash
  4. positions — current open positions

Note: IBKR news is not tested here. The CP API news endpoint requires a
conid and is instrument-specific — it cannot serve as a general macro
headline feed. Unscheduled event detection uses WebSearchClient instead.
See TICKET-028 for details.

No database required. PodSettings is constructed directly with paper-mode
defaults so this script runs before any seed or migration.

Prerequisites:
  - IBKR Client Portal gateway is running locally (default: https://localhost:5000)
  - You are authenticated in the gateway UI
  - .env is populated with IBKR_BASE_URL, IBKR_ACCOUNT_ID, IBKR_PAPER_ACCOUNT_ID

Usage:
    uv run python scripts/test_ibkr_connectivity.py
"""

import sys
import uuid
from datetime import UTC, datetime

from app.core.settings import get_settings
from app.integrations.ibkr_client import IBKRClient, IBKRClientError
from app.models.enums import KillAuthority, TradingMode
from app.core.pod_settings import PodSettings


def _paper_settings() -> PodSettings:
    return PodSettings(
        pod_id=uuid.uuid4(),
        trading_mode=TradingMode.paper,
        target_vol_per_position=0.05,
        max_position_pct=0.25,
        rebalance_threshold_pct=0.01,
        rebalance_day=0,
        intake_timeout_hours=24,
        kill_authority_default=KillAuthority.alert_only,
        vol_lookback_days=60,
    )


def _check(label: str, fn) -> bool:
    """Run fn(), print result, return True on success."""
    print(f"\n{'─' * 60}")
    print(f"CHECK: {label}")
    try:
        result = fn()
        print(f"  PASS")
        print(f"  {result}")
        return True
    except IBKRClientError as exc:
        print(f"  FAIL — IBKRClientError: {exc}")
        return False
    except Exception as exc:
        print(f"  FAIL — {type(exc).__name__}: {exc}")
        return False


def main() -> None:
    settings = get_settings()

    missing = [
        name
        for name, val in [
            ("IBKR_BASE_URL", settings.ibkr_base_url),
            ("IBKR_ACCOUNT_ID", settings.ibkr_account_id),
            ("IBKR_PAPER_ACCOUNT_ID", settings.ibkr_paper_account_id),
        ]
        if not val
    ]
    if missing:
        print(f"ERROR: missing required env vars: {', '.join(missing)}")
        print("Set them in .env and re-run.")
        sys.exit(1)

    client = IBKRClient(
        base_url=settings.ibkr_base_url,
        account_id=settings.ibkr_account_id,
        paper_account_id=settings.ibkr_paper_account_id,
        pod_settings=_paper_settings(),
    )

    print(f"IBKR Connectivity Smoke Test")
    print(f"Gateway : {settings.ibkr_base_url}")
    print(f"Account : {settings.ibkr_paper_account_id} (paper)")
    print(f"Started : {datetime.now(tz=UTC).isoformat()}")

    results: list[bool] = []

    # 1. conid resolution is exercised implicitly by get_price_history, but
    #    call _get_conid directly first so a lookup failure is clearly labelled.
    results.append(
        _check(
            "conid lookup — SPY",
            lambda: f"conid={client._get_conid('SPY')}",
        )
    )

    # 2. Price history — 1 month of daily bars (~30 trading days)
    def _price_history():
        history = client.get_price_history("SPY", period="1m", bar_size="1d")
        first = history.bars[0] if history.bars else None
        last = history.bars[-1] if history.bars else None
        return (
            f"{len(history.bars)} bars | "
            f"first={first.timestamp.date() if first else 'n/a'} "
            f"close={first.close if first else 'n/a'} | "
            f"last={last.timestamp.date() if last else 'n/a'} "
            f"close={last.close if last else 'n/a'}"
        )

    results.append(_check("price history — SPY 1m daily", _price_history))

    # 3. Account summary
    def _account_summary():
        s = client.get_account_summary()
        return (
            f"NAV=${s.net_liquidation:,.2f} | "
            f"buying_power=${s.buying_power:,.2f} | "
            f"cash=${s.cash_balance:,.2f}"
        )

    results.append(_check("account summary (paper)", _account_summary))

    # 4. Positions
    def _positions():
        positions = client.get_positions()
        if not positions:
            return "0 open positions"
        lines = [
            f"{p.symbol} qty={p.position} mktVal=${p.market_value:,.2f}"
            for p in positions
        ]
        return f"{len(positions)} position(s): " + " | ".join(lines)

    results.append(_check("positions (paper)", _positions))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'═' * 60}")
    print(f"RESULT: {passed}/{total} checks passed")
    if passed < total:
        print("One or more checks failed — see output above.")
        sys.exit(1)
    else:
        print("All checks passed. IBKR Client Portal connectivity confirmed.")

    client.close()


if __name__ == "__main__":
    main()
