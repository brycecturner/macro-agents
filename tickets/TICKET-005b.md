# TICKET 005b — OECD API Client

**Section:** 2 — Data Integrations

## Acceptance Criteria

- OECDClient class wraps the OECD JSON RESTful API (no library required,
  direct HTTP calls via httpx)
- Methods: get_series(dataset, subject, country, start_date, end_date)
- Supports key developed market series: ECB policy rates, Eurozone CPI,
  non-US G10 yield curves, OECD composite PMI
- All responses include retrieval timestamp for citation purposes
- Client raises a typed OECDClientError on API failure
- Results cached locally within a single workflow run
- Tests mock the API and confirm correct parsing and error handling
