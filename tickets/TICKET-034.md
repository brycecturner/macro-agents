# TICKET 034 — Order Executor

**Section:** 8 — Execution Engine

## Acceptance Criteria

- OrderExecutor submits orders to IBKR Client Portal API
- Order direction (buy/sell) is always derived from thesis_instruments.direction — never hardcoded
- Submits limit order at mid-price; waits up to 5 minutes for fill
- On fill timeout: cancels limit order and resubmits as market order
- Logs all execution details to trades table: submitted_price, fill_price,
  fill_time, slippage, order_type
- Raises typed OrderExecutionError on failure; does not silently swallow
  errors
- Tests mock IBKR API and confirm limit-then-market fallback and logging
