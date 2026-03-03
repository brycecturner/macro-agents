# TICKET 008 — BaseWorkflow Class & WorkflowResult

**Section:** 3 — Workflow Engine

## Acceptance Criteria

- BaseWorkflow abstract class exists with interface matching PRD Section 4.5:
  name, description, required_inputs, execute(thesis, context)
- WorkflowResult dataclass exists with fields: workflow_name, status,
  structured_output, citations, agent_inferences, raw_output
- Citation dataclass exists with fields: source_type, label, url,
  retrieval_date (format matches PRD citation table)
- WorkflowContext dataclass exists and holds: thesis object, prior
  WorkflowResult list, pod settings
- Attempting to instantiate BaseWorkflow directly raises NotImplementedError
- Tests confirm interface contract is enforced
