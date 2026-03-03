# TICKET 039 — Performance Metrics Calculator

**Section:** 9 — Performance Measurement

## Acceptance Criteria

- PerformanceCalculator class computes rolling metrics from portfolio_snapshots
- Pod-level: rolling Sharpe (30d, 90d, 1Y), max drawdown (rolling and
  since inception), benchmark comparison vs SPY and 60/40
- Thesis-level: P&L attribution, return vs backtest expectation, days in
  trade, condition status summary
- Tests confirm correct Sharpe, drawdown, and attribution calculations
  against known inputs
