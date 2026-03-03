# TICKET 012 — InstrumentAnalysisWorkflow

**Section:** 3 — Workflow Engine

## Acceptance Criteria

- Implements BaseWorkflow
- Pulls 5-year price history for the thesis instrument(s) from IBKR
- Computes: annualized return, annualized volatility, max drawdown,
  60-day realized volatility (used later for position sizing)
- Assesses correlation between instrument price history and the relevant
  FRED macro series
- All data cited with IBKR source label and timestamp
- Tests confirm correct computation of all statistics
