# TICKET 018 — Intake Agent

**Section:** 4 — Idea Pipeline — Input & Intake

## Acceptance Criteria

- On thesis submission, an agent generates the intake message containing:
  instrument mapping, one-paragraph thesis restatement, and 2-3 proposed
  falsification conditions
- Intake message displayed on thesis detail page in the web UI
- No email sent at intake stage — the entire intake conversation happens
  in the UI
- Thesis status updated to 'intake_sent'
- User responds via a form on the thesis detail page in the UI only
- On response: corrections stored, thesis status updated, research workflow
  queued
- On timeout (configurable, default 24 hours): research workflow proceeds
  with original interpretation; thesis flagged with 'intake_unconfirmed'
- intake_unconfirmed flag is non-dismissible until user explicitly
  acknowledges it via the UI
- Tests confirm intake generation, UI response handling, and timeout behavior
