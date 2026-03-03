# TICKET 014 — BacktestWorkflow

**Section:** 3 — Workflow Engine

## Acceptance Criteria

- Implements BaseWorkflow
- Proxy-based historical analog analysis — NOT a rules-based backtest
- Consumes analog periods from HistoricalAnalogWorkflow and instrument
  price history from InstrumentAnalysisWorkflow via WorkflowContext
- For each analog period computes: total return, annualized return, max
  drawdown, volatility, and directional correctness vs thesis direction
- Aggregate output: average return, worst case, best case, win rate,
  average max drawdown, benchmark comparison (SPY and 60/40) over same periods
- Output explicitly labeled "Historical Analog Analysis" — never "backtest"
- Output includes explicit note on statistical limitations when fewer than
  5 analog periods are available
- Results stored as JSON in object storage; referenced by ID in workflow_runs
- All data cited with FRED, OECD, and IBKR source labels
- Tests confirm correct return/drawdown calculations and benchmark comparison
