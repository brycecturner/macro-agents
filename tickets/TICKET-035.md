# TICKET 035 — Close Trade Service

**Section:** 8 — Execution Engine

## Acceptance Criteria

- CloseTrade service method exists in app/services/
- Accepts thesis_id and close_reason (human_manual, kill_condition, auto_close)
- Reads direction and role from thesis_instruments for each instrument in the thesis
- Submits a market sell order for long positions, buy-to-cover for short positions
- In v1 all positions are long — but close logic must read from thesis_instruments.direction, never assume long
- Updates thesis status to closed
- Writes to audit_log with changed_by, timestamp, and close_reason
- Sends confirmation email via AlertEmailService
- "Close Trade" button visible on brief page for active theses
- Button triggers a confirmation modal with Cancel and Confirm
- No action taken until Confirm clicked; modal dismisses on Cancel
- Kill authority workflow calls the same CloseTrade method — no separate implementation
- Tests confirm: full position close order submitted, status updated, audit entry written, email sent, and that human and agent paths call identical service method
