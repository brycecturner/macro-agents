# TICKET 010 — MacroContextWorkflow

**Section:** 3 — Workflow Engine

## Acceptance Criteria

- Implements BaseWorkflow
- Pulls FRED series relevant to the thesis based on thesis content:
  at minimum yield curve (T10Y2Y), CPI (CPIAUCSL), Fed Funds Rate (FEDFUNDS),
  and unemployment (UNRATE)
- Produces a structured summary of current macro backdrop
- All data points cited with FRED series ID and retrieval date
- Agent inferences explicitly flagged in agent_inferences field
- Tests confirm correct FRED series selection and citation format
