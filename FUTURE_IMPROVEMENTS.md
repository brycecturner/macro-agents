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

---

## Third-Party News API for Unscheduled Event Detection (deferred during Ticket 028)

The daily monitoring job detects unscheduled macro events (tariff announcements, surprise Fed statements, geopolitical developments) via targeted web searches through `WebSearchClient`. This is pull-based and keyword-driven — it will miss events that don't match query terms and runs on a schedule rather than in real time.

The IBKR Client Portal news endpoint was evaluated and ruled out: it requires a `conid` and is instrument-specific; it cannot serve as a general macro headline feed.

**Alternatives to evaluate when revisiting:**

| Provider | Free tier | Notes |
|---|---|---|
| Alpaca News API | Yes (limited) | Financial headlines with ticker tags; REST API; finance-focused |
| Polygon.io | Yes (limited) | Market news feed; good ETF and macro coverage |
| NewsAPI.org | Yes (100 req/day) | General news aggregator; broad event coverage |

**Recommendation:** Alpaca or Polygon are the natural first choices — finance-focused and free tiers are adequate for v1 volume. A push-based webhook or WebSocket feed would be preferable to polling if event detection latency becomes a concern.

**Revisit if:** The web search approach misses events that move positions, or if near-real-time event detection becomes a requirement.

---

## Workflow Chain Registry (flagged during Ticket 013 / run_workflow.py work)

**Move the pipeline execution order from a hardcoded list in `scripts/run_workflow.py` into a database-backed registry.**

Currently, `PIPELINE` in `scripts/run_workflow.py` is a static list of `(class_name, module_path)` tuples that defines both workflow discovery and execution order. The PRD mentions a `workflow_registry` table (Section 4.6) but it is not yet implemented.

**Proposed improvement:** Implement `workflow_registry` as a proper database table with:
- `id` — UUID v4 PK, generated in the application layer (consistent with all other tables)
- `name` — workflow class name (unique, used as identifier)
- `module_path` — importable Python path for dynamic loading
- `pipeline_position` — integer defining execution order within the sequential chain
- `description` — human-readable description surfaced in the UI
- `is_deep_dive` — boolean distinguishing core research workflows from user-initiated deep dives
- `is_enabled` — allows disabling a workflow without a code deploy

The `WorkflowRunner` and `run_workflow.py` both read from this table at startup instead of hardcoding the list. Adding a new workflow requires a migration and a seed row — not a code change in the runner.

**Why deferred:** The static list is sufficient while the pipeline is being built out. The registry becomes meaningful once the full 7-workflow chain is implemented and the UI needs to enumerate available workflows.

**Revisit when:** All core research workflows are implemented (after `RecommendationWorkflow`), or when the UI needs to display or trigger individual workflows dynamically.
