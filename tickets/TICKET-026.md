# TICKET 026 — State Condition Evaluator

**Section:** 6 — Falsification Monitoring

## Acceptance Criteria

- ConditionEvaluator class evaluates a single state-type falsification
  condition against current data
- Fetches required data from FRED or IBKR based on condition definition
- Returns: result (passing / falsified), data_value, threshold, citation
- Result logged to condition_evaluations with full citation
- Tests confirm correct evaluation logic and citation format for state
  conditions
