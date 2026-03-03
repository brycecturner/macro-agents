# TICKET 006 — IBKR Client Portal API Client

**Section:** 2 — Data Integrations

## Acceptance Criteria

- IBKRClient class wraps the IBKR Client Portal REST API
- Methods: get_positions(), get_account_summary(), get_price_history(symbol,
  period, bar_size), submit_order(order), cancel_order(order_id),
  get_news(since_timestamp)
- Client reads trading_mode from pod settings and routes to paper or real
  account accordingly
- All requests include authentication headers; session management handled
  automatically
- Client raises a typed IBKRClientError with status code on failure
- Tests mock the API and confirm correct routing between paper and real modes
