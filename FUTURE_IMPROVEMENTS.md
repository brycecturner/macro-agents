# Future Improvements

Decisions that were consciously deferred during implementation. Captured here so they don't get lost.

---

## FRED Release ID Mapping (deferred during Ticket 005)

Currently hardcoded in `app/core/constants.py` as a static dict mapping trigger type names (e.g. `CPI_RELEASE`) to FRED numeric release IDs.

**Proposed improvement:** Store the mapping in a `fred_release_configs` table in PostgreSQL. A monthly job calls `fred.search_releases()` to verify and update IDs. The economic calendar sync reads from the table instead of constants.

**Why deferred:** FRED release IDs essentially never change, so a monthly sync provides little practical value relative to the added complexity (new migration, new table, bootstrap/seed logic, new job).

**Revisit if:** FRED restructures its release catalog or we add many new trigger types that are hard to look up manually.

---

## Workflow Operational Constants (flagged during Ticket 013)

**Pull per-workflow operational constants into a PG config table.**

Currently, values like `_N_QUERIES_MAX`, `_MAX_CANDIDATES`, and `_N_FURTHER_READING_MAX`
in `WebResearchWorkflow` (and similar constants in other workflows) are module-level
Python constants. They should be moved into a dedicated `workflow_configs` table in
PostgreSQL, loaded at runtime via a `WorkflowSettings` Pydantic model analogous to
`PodSettings`. This would allow tuning workflow behavior (e.g. number of searches,
candidate pool size) without code deploys, and would follow the same pattern as
`pod_configs` for operational parameters.

Affected files: `app/workflows/web_research.py` (and likely other workflow files
as they accumulate similar constants).
