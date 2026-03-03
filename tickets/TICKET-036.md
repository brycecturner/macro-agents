# TICKET 036 — Weekly Rebalance Job

**Section:** 8 — Execution Engine

## Acceptance Criteria

- APScheduler job runs at Monday market open (9:35am ET)
- Calls PortfolioManager.compute_rebalance_orders()
- Submits all required orders via OrderExecutor
- Logs rebalance run: orders submitted, orders skipped (below threshold),
  estimated trading costs
- Manual rebalance trigger available via POST /portfolio/rebalance (requires
  confirmation)
- Tests confirm job execution, order submission, skip logging, and manual
  trigger
