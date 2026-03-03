# TICKET 027 — Event Condition Evaluator

**Section:** 6 — Falsification Monitoring

## Acceptance Criteria

- EventConditionEvaluator checks whether the condition's trigger_type has
  fired since last_evaluated_at
- For scheduled types: queries economic_calendar for releases since
  last_evaluated_at matching trigger_type
- For unscheduled types: queries news_events for detected events since
  last_evaluated_at matching trigger_type
- If trigger found: evaluates condition and logs result
- If no trigger: logs result as 'no_trigger' with timestamp
- Tests confirm both scheduled and unscheduled trigger detection, and
  correct no_trigger logging
