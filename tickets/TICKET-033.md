# TICKET 033 — Portfolio Manager

**Section:** 8 — Execution Engine

## Acceptance Criteria

- PortfolioManager class maintains current target portfolio state
- compute_target_weights() returns target dollar allocation for all active
  theses using PositionSizer
- compute_rebalance_orders() diffs target weights against current IBKR
  positions and returns a list of required orders
- Orders below the rebalance threshold (1% NAV, from pod settings) are
  filtered out and logged as skipped
- Tests confirm correct diff computation, threshold filtering, and order
  list output
