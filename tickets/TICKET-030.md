# TICKET 030 — Kill Authority Workflow

**Section:** 7 — Kill Authority & Alerts

## Acceptance Criteria

- KillAuthorityWorkflow checks kill_authority setting on thesis
- For alert_only: creates alert record, sends email, takes no trading action
- For auto_close: submits closing order to IBKR, creates alert record,
  sends email confirming action
- kill_authority change via UI logged to audit_log with changed_by and
  timestamp
- Tests confirm correct behavior for each kill_authority mode
