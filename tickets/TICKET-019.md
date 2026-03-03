# TICKET 019 — Research Workflow Orchestration

**Section:** 4 — Idea Pipeline — Input & Intake

## Acceptance Criteria

- WorkflowRunner triggered automatically after intake confirmation or timeout
- Runs all 7 core workflows sequentially in order defined in PRD Section 4.3
- Brief assembled from all WorkflowResult objects and stored
- Further reading entries saved to further_reading table from
  WebResearchWorkflow output
- Thesis status updated to 'researched'
- Email notification sent to user on brief completion — this is the first
  and only email in the idea pipeline flow; contains a direct link to the
  brief; no reply handling
- Tests confirm end-to-end orchestration from intake to brief completion
  and email notification sent on completion
