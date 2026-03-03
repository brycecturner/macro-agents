# TICKET 040 — Dashboard Home Page

**Section:** 10 — Dashboard & Navigation

## Acceptance Criteria

- Server-rendered home page shows:
  - Pod-level performance summary: NAV, daily change, rolling Sharpe (30d),
    max drawdown since inception, benchmark comparison
  - Table of active theses with columns: title, instrument, direction, P&L,
    days in trade, condition status (all passing / N at risk / falsified)
  - trading_mode badge displayed prominently
- Closed theses are hidden from the dashboard 24 hours after all associated positions are confirmed flat by IBKR
- Clicking a thesis row navigates to the brief page
- Performance data loaded from PerformanceCalculator
- Tests confirm all metrics render and thesis table is correct
