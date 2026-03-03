# TICKET 023 — Falsification Condition Editing & Lock Enforcement

**Section:** 5 — Trade Brief

## Acceptance Criteria

- Falsification conditions are editable (add, remove, modify) only when
  thesis status is 'approved' — never when 'active' or any other status
- Edit controls (add, edit, delete buttons) rendered on brief page only
  when thesis status is 'approved'
- API endpoints for creating, updating, and deleting conditions return
  a typed ConditionLockedError if thesis status is 'active'
- "Test Now" button available on every condition regardless of thesis status;
  runs the condition evaluator on demand and displays result inline via HTMX:
  passing or failing, current data value, threshold, and citation
- Test Now is read-only — it never modifies condition or thesis state
- Close-and-reopen is the only path to editing a condition on an active thesis; no exceptions
- Tests confirm: edit endpoints reject active thesis conditions, Test Now works
  on both approved and active theses, edit controls absent from active thesis brief UI
