# TICKET 024 — Deep Dive Workflows & UI (Tier 2)

**Section:** 5 — Trade Brief

## Acceptance Criteria

- SensitivityAnalysisWorkflow, RegimeStressTestWorkflow,
  PortfolioCorrelationWorkflow, HistoricalAnalogDetailWorkflow implemented
  as BaseWorkflow subclasses
- Each workflow registered in workflow_registry
- Brief page displays a trigger button for each available deep dive
- On trigger: workflow runs against current thesis; results appended to brief
  page via HTMX without full page reload
- Deep dive results include their own citation section
- Tests confirm each workflow executes and results append correctly
