# TICKET 009 — Workflow Registry

**Section:** 3 — Workflow Engine

## Acceptance Criteria

- workflow_registry table is populated at app startup by scanning the
  /workflows directory and registering all BaseWorkflow subclasses
- WorkflowRunner class accepts a thesis and executes a named list of
  workflows sequentially, passing accumulated WorkflowResult objects as
  context to each step
- Each workflow execution is logged to workflow_runs with: thesis_id,
  workflow_name, status, structured_output (JSON), citations (JSON),
  started_at, completed_at
- On workflow failure, status is set to 'failed', error is logged, and the
  runner continues to subsequent steps with a partial context flag
- Tests confirm sequential execution, context accumulation, and failure
  handling
