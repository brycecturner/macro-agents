# TICKET 018 — Intake Agent

**Section:** 4 — Idea Pipeline — Input & Intake

## Acceptance Criteria

- On thesis submission, an agent generates an intake message containing:
  (1) a concise restatement of the thesis as the agent understood it
  (instrument, direction, time horizon, and macro mechanism), and
  (2) any clarifying questions the agent needs answered before the full
  research pipeline can run effectively (e.g. ambiguous instrument,
  unclear time horizon, conflicting signals in the notes)
- The agent does NOT generate falsification conditions at intake — those
  are produced by FalsificationGenerationWorkflow in the research pipeline
- Intake message displayed on thesis detail page in the web UI
- No email sent at intake stage — the entire intake conversation happens
  in the UI
- Thesis status updated to 'intake_sent'
- User responds via a form on the thesis detail page; response may include
  corrections to the restatement or answers to the agent's questions
- On response: user response stored, research workflow queued
- On timeout (configurable, default 24 hours): research workflow proceeds
  with original interpretation; thesis_confirmed set to False
- thesis_confirmed=False banner is non-dismissible until user explicitly
  acknowledges it via the UI
- Tests confirm intake generation, UI response handling, and timeout behavior
