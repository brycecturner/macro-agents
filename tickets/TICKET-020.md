# TICKET 020 — Trade Brief Storage & Retrieval

**Section:** 5 — Trade Brief

## Acceptance Criteria

- Brief assembled from workflow_runs outputs and stored as JSON in object
  storage; referenced by thesis_id
- FastAPI route GET /theses/{thesis_id}/brief returns structured brief data
- Brief structure matches PRD Section 4.4 Tier 1 exactly: summary, instrument,
  direction, time horizon, backtest stats, assumptions, falsification
  conditions, recommendation, source index
- Source index lists all citations from all workflow runs for this thesis
- Tests confirm brief assembly and all required fields are present
