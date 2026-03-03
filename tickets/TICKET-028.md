# TICKET 028 — IBKR News Poller & LLM Event Classifier

**Section:** 6 — Falsification Monitoring

## Acceptance Criteria

- Scheduled job polls IBKR news API for headlines since last run
- LLM classifier reads headlines and assigns zero or one unscheduled
  trigger_type per headline: TARIFF_ANNOUNCEMENT, FED_SPEECH,
  GEOPOLITICAL_EVENT, SURPRISE_RATE_MOVE
- Headlines that match no trigger type are discarded
- Matched events stored in news_events with: headline, url, detected_at,
  trigger_type, classifier_confidence
- User can view and override misclassifications from the web UI
- Tests confirm classifier output structure and storage; mock LLM for unit
  tests
