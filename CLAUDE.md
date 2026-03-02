# CLAUDE.md
## Behavioral Configuration for Claude Code

This file defines how Claude Code should behave when working on this project.
It is the second source of truth after the PRD. When this file and the PRD
conflict, flag the conflict and ask before proceeding.

---

## Source of Truth Hierarchy

1. `PRD.md` — product requirements, architecture decisions, and scope boundaries
2. `CLAUDE.md` (this file) — engineering standards and behavioral rules
3. The specific ticket being worked on — implementation detail

When making any decision not clearly covered by one of these three sources,
**stop and ask**. Do not silently resolve ambiguity by making assumptions.
This applies especially to:

- Architectural decisions (new abstractions, changed data models, new dependencies)
- Anything that contradicts or extends the PRD
- Any decision that would affect more than the current ticket
- Cases where two reasonable approaches exist and the right one isn't obvious

A short "I'm about to do X because Y — does that work?" is always the right
move when in doubt. It is never annoying. Silent decisions that turn out to
be wrong are.

---

## Key Project Files

These files are always in the repo root. Read them before starting any ticket.

| File | Purpose |
|------|---------|
| `PRD.md` | Full product requirements, architecture decisions, and scope boundaries |
| `CLAUDE.md` | This file — engineering standards and behavioral rules |
| `ERD.md` | Entity relationship diagram for all core database tables with field definitions |
| `implementation_tickets.txt` | Full ordered list of implementation tickets with acceptance criteria |

When starting a new ticket: read the relevant PRD sections first, check the ERD for any tables the ticket touches, then implement.

---

## Project Structure

```
/
├── app/
│   ├── api/              # FastAPI route handlers
│   ├── models/           # SQLAlchemy ORM models
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/         # Business logic layer
│   ├── workflows/        # BaseWorkflow subclasses (one file per workflow)
│   ├── integrations/     # External API clients (IBKR, FRED, web search, Anthropic)
│   ├── jobs/             # APScheduler job definitions
│   ├── templates/        # Jinja2 HTML templates
│   └── core/             # Settings, database session, shared utilities
├── migrations/           # Alembic migration files
├── tests/
│   ├── unit/             # Unit tests mirroring app/ structure
│   └── integration/      # End-to-end integration tests
├── .env.example          # All required env vars documented, no real values
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── CLAUDE.md
└── PRD.md
```

New files go in the most specific appropriate directory. If a new directory
is needed that doesn't exist above, ask before creating it.

---

## Code Style

### Formatter & Linter
- **Black** for formatting. Line length: 88 (Black default). No exceptions.
- **Ruff** for linting. Configuration lives in `pyproject.toml`.
- Both must pass with zero errors before any commit.
- Run before committing: `uv run black . && uv run ruff check .`

### Python Standards
- Python 3.11+. Use modern syntax: match statements, `X | Y` union types,
  `tomllib`, etc. where appropriate.
- Type hints on all function signatures and class attributes. No `Any` unless
  genuinely unavoidable — if you use it, leave a comment explaining why.
- Pydantic for all data validation (request schemas, settings, API responses).
- SQLAlchemy ORM for all database access. No raw SQL except in migrations.
- f-strings for string formatting. No `.format()` or `%` formatting.
- Prefer explicit over implicit. Readable over clever.

### Naming Conventions
- Classes: `PascalCase`
- Functions and variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods and attributes: `_leading_underscore`
- Database table names: `snake_case` (plural, e.g., `theses`, `workflow_runs`)
- Files: `snake_case.py`

### Imports
- Standard library first, then third-party, then internal — separated by
  blank lines (ruff enforces this automatically).
- Absolute imports only. No relative imports.

---

## Model Selection

Different workflows require different levels of reasoning. Model selection
is a per-workflow configuration — not a global setting. The model for each
workflow is set in the workflow class definition and can be overridden via
pod settings.

### Model assignments

| Workflow / Task | Model | Rationale |
|----------------|-------|-----------|
| `MacroContextWorkflow` | claude-sonnet-4-6 | Structured data retrieval and summarization |
| `HistoricalAnalogWorkflow` | claude-sonnet-4-6 | Pattern matching against well-defined criteria |
| `InstrumentAnalysisWorkflow` | claude-sonnet-4-6 | Quantitative analysis with structured output |
| `WebResearchWorkflow` | claude-sonnet-4-6 | Search and synthesis of well-scoped sources |
| `BacktestWorkflow` | claude-sonnet-4-6 | Rules-based computation with defined output schema |
| `FalsificationGenerationWorkflow` | claude-opus-4-6 | Requires judgment: translating qualitative assumptions into rigorous falsifiable conditions |
| `RecommendationWorkflow` | claude-opus-4-6 | Requires judgment: synthesizing ambiguous research into a Go/No-Go call |
| Intake conversation | claude-opus-4-6 | Getting thesis interpretation right at intake prevents compounding errors downstream |
| LLM event classifier (news) | claude-sonnet-4-6 | Structured classification with defined label set |
| Deep dive workflows | claude-sonnet-4-6 | Structured analysis tasks with defined outputs |

### Implementation requirement
The model name must be a configurable attribute on each workflow class, not
hardcoded in the API call. This allows model assignments to be tuned without
code changes.

```python
class RecommendationWorkflow(BaseWorkflow):
    model: str = "claude-opus-4-6"  # Configurable per workflow
    ...
```

---

## Model Cost Tracking

Every call to the Anthropic API must log token usage to the `llm_usage_log`
table. This is non-negotiable — missing a call is a bug.

### What to log
For every API call:
- `workflow_run_id` — foreign key to workflow_runs (nullable for non-workflow
  calls such as intake and event classifier)
- `thesis_id` — foreign key to theses (nullable where not applicable)
- `model` — exact model string used (e.g., `claude-opus-4-6`)
- `task_type` — human-readable label for the call (e.g., `macro_context`,
  `intake`, `event_classifier`, `recommendation`)
- `input_tokens` — from API response usage object
- `output_tokens` — from API response usage object
- `estimated_cost_usd` — computed at log time from token counts and known
  model pricing (see pricing constants below)
- `called_at` — timestamp

### Pricing constants
Maintain a `MODEL_PRICING` dict in `app/core/constants.py`. Update it when
Anthropic changes pricing. Cost is computed at log time so historical records
reflect the price paid, not current pricing.

```python
MODEL_PRICING = {
    "claude-opus-4-6": {"input_per_1k": 0.000015, "output_per_1k": 0.000075},
    "claude-sonnet-4-6": {"input_per_1k": 0.000003, "output_per_1k": 0.000015},
}
```

### Implementation requirement
Wrap all Anthropic API calls in a single `AnthropicClient` class in
`app/integrations/anthropic_client.py`. This class handles the API call,
extracts token usage from the response, computes cost, and writes to
`llm_usage_log` in one operation. No workflow or service should call the
Anthropic SDK directly — all calls go through `AnthropicClient`.

```python
class AnthropicClient:
    def complete(
        self,
        messages: list[dict],
        model: str,
        task_type: str,
        workflow_run_id: int | None = None,
        thesis_id: int | None = None,
        **kwargs,
    ) -> AnthropicResponse:
        # Makes API call, logs usage, returns response
        ...
```

### Schema note
The `llm_usage_log` table is designed to support a cost dashboard in v2.
It must include all fields needed to aggregate by: model, task_type,
thesis_id, date, and pod_id (via thesis). Do not add a dashboard or UI for
this in v1 — logging only.

---

## Testing

### Philosophy
- Tests are written **alongside** implementation, not after. Each ticket
  should be delivered with its tests passing.
- A final end-to-end integration test suite is written when the full system
  is complete (Ticket 042).
- Tests live in `tests/unit/` mirroring the `app/` structure.
  For example: `app/workflows/macro_context.py` →
  `tests/unit/workflows/test_macro_context.py`

### Framework
- **pytest** only. No unittest.
- All tests must be written as **test classes**, not standalone functions.
  This is non-negotiable — it keeps related tests organized and makes
  shared fixtures explicit.

```python
# CORRECT
class TestMacroContextWorkflow:
    def test_returns_workflow_result(self, mock_fred_client):
        ...

    def test_citations_include_series_id(self, mock_fred_client):
        ...

# WRONG — do not do this
def test_returns_workflow_result():
    ...
```

### Coverage Requirements
- All public methods on service classes and workflow classes must have tests.
- All FastAPI routes must have tests covering: happy path, validation errors,
  and not-found cases.
- All external API calls (IBKR, FRED, SMTP, Anthropic) must be mocked in
  unit tests. No unit test should make a real network call.
- `AnthropicClient` must have tests confirming that token usage is logged
  for every call, including failed calls.
- Integration tests may use real network calls where explicitly noted.

### Test Naming
- Test classes: `Test{ClassName}` or `Test{ConceptBeingTested}`
- Test methods: `test_{what_it_does}` or `test_{scenario}_{expected_outcome}`
  Examples: `test_returns_empty_list_when_no_theses`,
  `test_raises_error_when_ibkr_unavailable`

### Fixtures
- Shared fixtures live in `tests/conftest.py`
- Workflow-specific fixtures live in the test file that uses them
- Mock external clients at the fixture level, not inside individual tests

---

## Git Conventions

### Commit Message Format
All commit messages must follow this format exactly:

```
<type>(TICKET-###): <ticket title> — <short description of change>
```

Types: `feat`, `fix`, `chore`, `test`, `refactor`, `docs`

Examples:
```
feat(TICKET-008): BaseWorkflow Class & WorkflowResult — add Citation dataclass
fix(TICKET-014): BacktestWorkflow — correct Sharpe ratio annualization
test(TICKET-025): State Condition Evaluator — add edge case for missing FRED data
chore(TICKET-001): Project Scaffold — add ruff config to pyproject.toml
```

- One logical change per commit. Don't bundle unrelated changes.
- If a change fixes something discovered while working on a different ticket,
  use the ticket number where the fix belongs, not the one being worked on.

### Branching
- Feature branches per ticket: `ticket-###-short-description`
  Example: `ticket-008-base-workflow-class`
- Merge to `main` when the ticket is complete and all tests pass.
- `main` must always be in a runnable state.

---

## Dependency Management

- **Package manager: `uv`**. No pip, no poetry, no conda.
- All dependencies declared in `pyproject.toml`. No `requirements.txt` — that file should not exist in this project.
- `uv.lock` is committed to version control. This is the source of pinned versions.
- To add a dependency: `uv add <package>`. To remove: `uv remove <package>`. Never edit `pyproject.toml` dependency lists by hand.
- Add a comment in `pyproject.toml` above any non-obvious dependency explaining why it's needed.
- Do not add a new dependency to solve a problem that the standard library or an already-included dependency can solve.
- Before adding any new third-party library, flag it and confirm. New dependencies are an architectural decision.

---

## Environment & Secrets

- All secrets and infrastructure config in `.env` (never committed).
- `.env.example` always kept up to date with every required variable
  documented but no real values.
- No hardcoded values for: API keys, database URLs, email credentials,
  or port numbers. These are loaded from `.env` via the core settings
  module at startup — never via `os.environ.get()` scattered through
  the codebase.

## Operational Configuration

- All operational parameters (`trading_mode`, `target_vol_per_position`,
  `rebalance_threshold_pct`, etc.) live in the `pod_configs` table.
  See PRD Section 9 for the full parameter table.
- Services and workflows receive a `PodSettings` Pydantic model — they
  never read from `pod_configs` directly.
- Default values are set by the seed script only. They do not exist
  anywhere else in the codebase.
- Every change to a `pod_configs` value must write to `audit_log` in
  the same transaction with: parameter name, previous value, new value,
  changed_by, and timestamp.
- Never hardcode a value that is listed in the `pod_configs` parameter
  table. If you find yourself writing a magic number like `0.05` or `24`
  in application logic, stop and load it from `PodSettings` instead.

---

## External API Behavior

- All external API calls (IBKR, FRED, SMTP, Anthropic) go through their
  dedicated client class in `app/integrations/`. No direct API calls from
  routes, services, or workflows.
- Every client raises a typed exception class on failure. No silent failures.
  No returning `None` to indicate an error.
- All responses from external APIs include a retrieval timestamp. This is
  required for citation formatting.
- IBKR routing (paper vs. live) is enforced inside `IBKRClient` by reading
  `trading_mode` from pod settings at request time. It is never controlled
  by environment variables or code flags.
- All Anthropic API calls go through `AnthropicClient`. Never call the
  Anthropic SDK directly. See Model Cost Tracking section.

---

## Database

- All schema changes via Alembic migrations. Never modify the database
  directly or use `create_all()` outside of tests.
- Migrations must be reversible (implement `downgrade()`).
- **Primary keys:** Every table uses UUID v4. UUIDs are generated in the
  application layer (use `uuid.uuid4()`) before the database write — not
  by the database itself. `pods` additionally has a human-readable `name`
  field for display; all foreign key references still use the UUID.
  Never use a workflow name or any string as an identifier or foreign key.
- The `audit_log` table is append-only. No update or delete operations,
  ever. If something needs to be corrected, a new row is added.
- The `llm_usage_log` table is also append-only. Records are never updated
  or deleted.
- Every write to a tracked entity (thesis status, kill_authority, trading_mode,
  go/no-go decisions) must also write to `audit_log` in the same transaction.

---

## What "Done" Means for a Ticket

A ticket is not done until:

1. Implementation is complete and matches the acceptance criteria
2. All tests pass (`pytest`)
3. Black and Ruff pass with zero errors (`black . && ruff check .`)
4. The `.env.example` is updated if new environment variables were added
5. The commit message follows the format above
6. No TODOs or placeholder code left in the diff (leave a comment and raise
   it as a question if something needs a future decision)
