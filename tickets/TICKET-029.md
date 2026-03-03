# TICKET 029 — Daily Monitoring Job

**Section:** 6 — Falsification Monitoring

## Acceptance Criteria

- APScheduler job runs at market close (4:30pm ET) on trading days
- For each active thesis: runs all falsification conditions through the
  appropriate evaluator (state or event)
- Results logged to condition_evaluations
- If any condition is falsified: triggers kill authority workflow for
  that thesis
- Job execution logged with start time, end time, theses processed,
  conditions evaluated, and any falsifications triggered
- Tests confirm job runs all conditions, handles failures gracefully, and
  triggers kill workflow on falsification
