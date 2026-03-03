# TICKET 021 — Trade Brief UI (Tier 1)

**Section:** 5 — Trade Brief

## Acceptance Criteria

- Server-rendered brief page displays all Tier 1 fields
- Falsification conditions displayed with condition_type label
- Source index rendered at bottom with properly formatted citations
- Human decision field (Go / No-Go / Hold for review) rendered as buttons
- On decision: thesis status updated, decision logged to audit_log with
  timestamp
- intake_unconfirmed flag displayed prominently if set; dismissal button
  logs acknowledgment to audit_log
- Tests confirm all fields render and decision buttons update status correctly
