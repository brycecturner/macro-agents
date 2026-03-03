# TICKET 017 — Idea Input UI & API

**Section:** 4 — Idea Pipeline — Input & Intake

## Acceptance Criteria

- FastAPI route POST /theses accepts thesis_title, time_horizon, direction,
  and freeform notes
- Input validated: all three required fields must be non-empty
- Thesis saved to database with status 'draft' and pod_id from default pod
- Server-rendered form page (Jinja2) with three required fields and a large
  text area for freeform notes
- Submission redirects to a thesis detail page showing status
- Tests confirm validation, storage, and status assignment
