# TICKET 005 — FRED Economic Calendar

**Section:** 2 — Data Integrations

## Acceptance Criteria

- Script fetches upcoming and recent release dates from FRED for all
  supported scheduled trigger types: CPI_RELEASE, FOMC_DECISION,
  NFP_RELEASE, PMI_RELEASE, GDP_RELEASE, PCE_RELEASE
- Results stored in the economic_calendar table with release_type,
  scheduled_date, and actual_date (populated after release)
- Script is idempotent — safe to run repeatedly without duplicating rows
- Tests confirm correct parsing and storage of release dates
