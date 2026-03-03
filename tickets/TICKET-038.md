# TICKET 038 — Daily Performance Snapshot Job

**Section:** 9 — Performance Measurement

## Acceptance Criteria

- APScheduler job runs at market close after the monitoring job
- Fetches current positions and NAV from IBKR
- Computes pod-level metrics: NAV, daily NAV change, gross exposure,
  net exposure
- Stores snapshot to portfolio_snapshots table
- Tests confirm snapshot computation and storage
