# Future Improvements

Decisions that were consciously deferred during implementation. Captured here so they don't get lost.

---

## FRED Release ID Mapping (deferred during Ticket 005)

Currently hardcoded in `app/core/constants.py` as a static dict mapping trigger type names (e.g. `CPI_RELEASE`) to FRED numeric release IDs.

**Proposed improvement:** Store the mapping in a `fred_release_configs` table in PostgreSQL. A monthly job calls `fred.search_releases()` to verify and update IDs. The economic calendar sync reads from the table instead of constants.

**Why deferred:** FRED release IDs essentially never change, so a monthly sync provides little practical value relative to the added complexity (new migration, new table, bootstrap/seed logic, new job).

**Revisit if:** FRED restructures its release catalog or we add many new trigger types that are hard to look up manually.
