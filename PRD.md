# Product Requirements Document
## Agent-Powered Macro Hedge Fund System

**Version:** 2.1  
**Status:** Draft — Ready for Implementation  
**Author:** Bryce  
**Target Builder:** Claude Code  

---

## 1. Vision & Overview

Build a systematic, agent-powered macro investment platform that allows a human portfolio manager to rapidly generate, research, validate, and monitor macro trade ideas — and execute them through a real brokerage account. The system is designed around a core belief: the human is best at forming hypotheses; agents are best at researching, validating, and monitoring them.

The platform has two distinct subsystems:

1. **The Idea Pipeline** — everything from raw thought to a validated, actionable trade brief: idea capture, agent research, backtesting, structured reporting, and ongoing hypothesis monitoring.
2. **The Execution Engine** — portfolio management, trade execution via Interactive Brokers, position sizing, rebalancing, and performance measurement.

The system is being built for a single user (pod) in v1, but the data model must support multiple pods from day one. See Section 9 for multi-pod architecture requirements.

---

## 2. Core Design Principles

- **Falsifiability over intuition.** Every trade idea that enters the system must have explicit, programmatically testable kill conditions. Vague theses don't get implemented.
- **Human judgment on entry, agent rigor on validation.** The human generates the idea; agents do the work of stress-testing it.
- **Rule-based over discretionary.** Where possible, decisions should follow explicit, auditable rules rather than ad hoc judgment.
- **Design for scale, build for one.** Multi-pod, multi-user architecture in the schema; single-pod simplicity in v1 features.
- **Paper before real.** The execution engine runs in paper trading mode first. Real trading is enabled explicitly per-pod.
- **Every claim must be traceable.** All data in research briefs carries a source. Agent reasoning is explicitly labeled as such and never presented as data.

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    IDEA PIPELINE                        │
│                                                         │
│  Idea Input → Intake Volley → Agent Research → Brief    │
│       ↓                                                 │
│  Falsification Conditions Generated                     │
│       ↓                                                 │
│  Human Review → Go / No-Go Decision                     │
└─────────────────────────┬───────────────────────────────┘
                          │ Go
┌─────────────────────────▼───────────────────────────────┐
│                  EXECUTION ENGINE                       │
│                                                         │
│  Portfolio Manager → Position Sizer → IBKR Executor     │
│       ↓                                                 │
│  Daily Falsification Monitor → Email Alert / Auto-Close │
│       ↓                                                 │
│  Performance Measurement & Attribution                  │
└─────────────────────────────────────────────────────────┘
```

**Primary instruments:** ETFs (e.g., TLT for duration, GLD for commodities, SPY, EEM). No single stocks, futures, or options in v1.

**Data sources:**
- Interactive Brokers Client Portal API — market data, price history, execution, news
- FRED API — US macro time series (yield curves, CPI, PMI, unemployment, Fed data)
- OECD API — developed market macro series (ECB rates, Eurozone CPI, non-US yield curves, OECD PMI)
- FRED release calendar — scheduled economic event triggers
- Web search (Anthropic tool) — news, Fed speeches, analyst commentary, macro research
- Research articles & academic papers — accessed via web during agent research workflows

---

## 4. The Idea Pipeline

### 4.1 Idea Input

The user submits a trade hypothesis via a lightweight input interface. The interface must be as frictionless as possible — the user thinks in prose, not forms.

**Input structure (hybrid):**

Required structured fields — kept to the absolute minimum:
- `thesis_title` — short label (e.g., "Yield Curve Steepener")
- `time_horizon` — rough duration (e.g., "6 months", "1 year")
- `direction` — long or short the primary instrument

Everything else goes in a **freeform notes field.** The agent is responsible for extracting structured information from the notes (instrument, macro thesis, key assumptions, suggested kill conditions). Do not make the user fill out what the agent can infer.

**Interface:** Server-rendered web UI (FastAPI + Jinja2). A simple form with the three required fields and a large text area. Submission triggers the intake workflow.

### 4.2 Intake Conversation

Before the full research workflow fires, the agent sends a single intake message to the user. This is **one volley — not an ongoing conversation.** The purpose is to confirm alignment before expensive research work begins, analogous to briefing a research analyst before sending them off.

**The intake message contains three things:**

1. **Instrument mapping** — "I interpreted this as a long TLT position. Is that correct?"
2. **Thesis mechanism** — a one-paragraph restatement of the macro thesis as the agent understood it, so the user can catch misinterpretations early.
3. **Proposed falsification conditions** — 2-3 candidate kill conditions the agent plans to formalize, presented for user confirmation or correction before they're locked in.

The intake message is displayed in the web UI. The entire intake conversation — submission, one-volley response, and user corrections — happens in the UI. No email is sent at this stage.

**Non-response handling:** If the user does not respond within a configurable timeout (default: 24 hours), the research workflow proceeds using the agent's best interpretation. The final brief must display a prominent, non-dismissible flag: *"Intake confirmation not received — agent proceeded with assumed interpretation. Review instrument mapping and falsification conditions carefully."* This flag persists until the user explicitly acknowledges it via the UI.

**Intake corrections:** If the user responds with corrections, the agent incorporates them and proceeds immediately. No second intake round — one volley is the rule. If the correction is itself ambiguous, the agent proceeds with its best interpretation and flags the remaining uncertainty in the brief.

### 4.3 Agent Research Workflow

**Architecture: single agent, sequential execution.**

A single research agent runs all seven core workflows sequentially, accumulating context across steps. Each workflow produces a structured `WorkflowResult` object — typed output, citations, and explicitly flagged inferences. Subsequent workflows consume prior `WorkflowResult` objects directly, not freeform text. This makes handoffs between steps reliable and debuggable.

This architecture is intentional for v1. A single agent with a linear execution chain is easier to debug and easier to iterate on research quality than a multi-agent system. The `WorkflowResult` interface is designed from day one to support parallel multi-agent execution in v2 — switching will be an orchestration layer addition, not a rewrite.

**WorkflowResult interface:**
```python
class WorkflowResult:
    workflow_name: str
    status: str                  # completed / failed / partial
    structured_output: dict      # typed per workflow, defined in each workflow class
    citations: list[Citation]    # all sources used
    agent_inferences: list[str]  # explicitly flagged inferences, not data-backed
    raw_output: str              # full agent output, stored but not passed forward
```

**Execution sequence:**

1. `MacroContextWorkflow` — pull relevant FRED series; summarize current macro backdrop
2. `HistoricalAnalogWorkflow` — identify historical periods with similar macro configurations
3. `InstrumentAnalysisWorkflow` — pull ETF price history from IBKR; assess correlation to thesis
4. `WebResearchWorkflow` — search for relevant news, Fed communications, analyst commentary; collect Further Reading candidates
5. `BacktestWorkflow` — proxy-based historical analog backtest (see Section 4.5 for full specification)
6. `FalsificationGenerationWorkflow` — formalize 3-5 discrete, programmatically testable kill conditions
7. `RecommendationWorkflow` — synthesize all prior WorkflowResult objects into a Go/No-Go recommendation with rationale

**Citation requirement:** Every factual claim, data point, and chart in the research output must carry a traceable source. No exceptions.

| Source type | Required label format |
|-------------|----------------------|
| FRED data series | `FRED:{series_id}, retrieved {date}` |
| IBKR market data | `IBKR:{instrument} {data_type}, {timestamp}` |
| Web source | Full URL + retrieval date |
| Agent inference / reasoning | `[Agent inference]` — explicitly labeled, never attributed to a data source |

`[Agent inference]` signals the agent is reasoning from pattern or analogy, not citing a source. Users should treat these with appropriate skepticism. Deep dive workflows exist partly to validate agent inferences with real data.

**Web source quality hierarchy:** Primary sources (Fed.gov, BLS, IMF, BIS, academic journals, central bank publications) are preferred over aggregators, news blogs, or opinion pieces. When a web source conflicts with a FRED or IBKR data point, the structured data source takes precedence and the conflict must be noted explicitly in the brief.

### 4.4 Output: The Trade Brief

The output is a structured trade brief with three tiers.

**Tier 1 — Executive Summary (always shown):**
- Thesis title and one-paragraph summary
- Primary instrument(s) and direction
- Time horizon
- Historical analog analysis summary (avg return, worst/best case, win rate, max drawdown, benchmark comparison, number of analog periods with explicit note on statistical limitations)
- Key assumptions
- Falsification conditions (listed, with evaluation type: state / event-triggered)
- Agent Go/No-Go recommendation with one-paragraph rationale
- Human decision field: Go / No-Go / Hold for review
- Source index — all citations used in the brief body, listed at the bottom

**Tier 2 — Deep Dives (optional, on-demand):**
The brief displays a set of deep dive workflow triggers the user can launch individually. These fire a single named workflow against the current thesis and append results to the brief. Deep dives are never generated automatically — always user-initiated.

Initial deep dive library:
- Sensitivity Analysis — rerun backtest varying entry timing
- Regime Stress Test — filter history to specific macro regimes
- Portfolio Correlation Check — analyze correlation to existing positions
- Historical Analog Detail — full breakdown of the analog periods identified

**Tier 3 — Further Reading (always shown, bottom of brief):**
A curated list of 3-5 sources surfaced during `WebResearchWorkflow`, ranked and annotated by the agent. Generated automatically — no user action required.

Ranking order: cited sources first (directly referenced in the brief body), then additional relevant sources the agent found and absorbed but did not directly cite. Ties broken by source quality (primary sources ranked above aggregators).

Each entry includes:
- Title and hyperlink
- Source type: `web` / `FRED series` / `academic paper`
- One sentence written by the agent explaining why this source is relevant to the thesis

Scope includes all source types. A link to the FRED series underpinning a falsification condition is as useful as a news article.

### 4.5 Research Workflow Definitions

Research workflows are Python classes with a defined interface. They live in a `/workflows` directory in the codebase. Each workflow is registered in the database with a name and description. Workflows are not stored as prompts — prompts are generated dynamically at runtime from the workflow definition and current thesis context. This makes the system robust to model changes.

**Workflow interface (all workflows implement this):**
```python
class BaseWorkflow:
    name: str
    description: str
    required_inputs: list[str]  # fields from the thesis object

    def execute(self, thesis: Thesis, context: WorkflowContext) -> WorkflowResult:
        # Returns structured output with citations
        ...
```

**Core research workflows (run automatically on every thesis):**

| Workflow | Description | Primary data source |
|----------|-------------|-------------------|
| `MacroContextWorkflow` | Pull FRED series relevant to thesis; summarize current macro backdrop | FRED API |
| `HistoricalAnalogWorkflow` | Find historical periods with similar macro config; summarize outcomes | FRED API + IBKR |
| `InstrumentAnalysisWorkflow` | Pull ETF price history; assess correlation to macro thesis | IBKR API |
| `WebResearchWorkflow` | Search for relevant news, Fed communications, analyst commentary; collect Further Reading candidates | Web search |
| `BacktestWorkflow` | Proxy-based historical analog backtest — measures instrument performance during macro-analog periods | FRED API + OECD API + IBKR API |

**BacktestWorkflow specification:**

This workflow does not attempt to define or simulate trading rules. Instead it measures how the thesis instrument has historically performed during periods where the macro backdrop was similar to the current configuration — the same periods identified by `HistoricalAnalogWorkflow`. This is the correct approach for macro theses, which are fundamentally claims about world-state rather than mechanical signals.

**Inputs (from prior WorkflowResult objects in context):**
- Analog periods from `HistoricalAnalogWorkflow` (date ranges + macro conditions)
- Instrument price history from `InstrumentAnalysisWorkflow`
- Thesis time horizon and direction

**What it computes for each analog period:**
- Instrument return over the period (total and annualized)
- Maximum drawdown during the period
- Volatility during the period
- Whether the period was directionally correct (matched thesis direction)

**Aggregate output across all analog periods:**
- Average return, worst case return, best case return
- Win rate (% of periods directionally correct)
- Average max drawdown
- Benchmark comparison: same metrics for SPY and 60/40 over the same periods

**What the output is honest about:**
The brief must clearly label this as a "historical analog analysis" not a "backtest" — the distinction matters. It is not simulating a trading strategy; it is measuring instrument behavior during analogous macro regimes. The number of analog periods is typically small (2-5), which limits statistical significance and must be noted explicitly in the output.
| `FalsificationGenerationWorkflow` | Formalize 3-5 kill conditions with measurable proxies | FRED + IBKR |
| `RecommendationWorkflow` | Synthesize all prior WorkflowResult objects into Go/No-Go with rationale | Prior workflow outputs |

**Deep dive workflows (user-initiated only):**

| Workflow | Description |
|----------|-------------|
| `SensitivityAnalysisWorkflow` | Rerun backtest varying entry timing ±1, 2, 3 months |
| `RegimeStressTestWorkflow` | Filter history to specific macro regimes (hiking cycle, recession, etc.) |
| `PortfolioCorrelationWorkflow` | Compute correlation of proposed position to all current active positions |
| `HistoricalAnalogDetailWorkflow` | Full breakdown of each analog period: duration, drawdown, catalysts |

Additional workflows can be added by creating a new Python class in `/workflows` and registering it. This is the primary extension point for new research capabilities.

### 4.6 Idea Storage

**Database:** PostgreSQL with pgvector extension for semantic search.

**Primary key policy:**
- All tables use UUID v4 primary keys (`id: uuid`, generated by the application layer before the database write).
- `pods` additionally has a human-readable `name` field (e.g., "Inflation Trades", "Rates Book") used for display. The UUID is always the identity used in foreign keys and references — never the name.

Rationale: UUIDs decouple identity from insertion order, are safe to generate before a database write, and avoid leaking row counts to any future external-facing surface. Workflow identity in particular must never be tied to a name string — names can change, UUIDs cannot.

Core tables:

```
pods                     — UUID PK + human-readable name field; multi-pod entity (v1 has one row)
pod_configs              — UUID PK; all operational parameters for a pod (one row per pod)
users                    — UUID PK
pod_memberships          — UUID PK; join table for users ↔ pods
theses                   — UUID PK; belongs to a pod
thesis_instruments       — UUID PK; ETFs associated with a thesis, with direction (long/short) and role (primary/hedge/secondary)
falsification_conditions — UUID PK; kill conditions for each thesis
condition_evaluations    — UUID PK; daily log of condition check results
workflow_registry        — UUID PK; registered workflows with name and description
workflow_runs            — UUID PK; log of all workflow executions with outputs and citations
further_reading          — UUID PK; curated sources per thesis, ranked and annotated
positions                — UUID PK; real/paper positions, belongs to a pod
trades                   — UUID PK; execution log
portfolio_snapshots      — UUID PK; daily portfolio state
economic_calendar        — UUID PK; scheduled macro release dates from FRED
news_events              — UUID PK; detected unscheduled events from IBKR news feed
alerts                   — UUID PK; alert log with delivery status
llm_usage_log            — UUID PK; token usage and cost per Anthropic API call (append-only)
audit_log                — UUID PK; append-only state change log
```

The `theses` table tracks status: `draft → intake_sent → researched → approved → active → closed → rejected`.

**Thesis closure and dashboard removal:**
When a thesis is closed, its positions are submitted for closure via the `CloseTrade` service. The 24-hour removal clock starts on **confirmed fill from IBKR** — not on order submission. Once all positions associated with the thesis are confirmed flat and 24 hours have elapsed, the thesis is removed from the dashboard. It remains permanently in the `theses` table with `status: closed` and a `closed_at` timestamp. It is always accessible via the thesis search and list view.

Users can search past theses by keyword (pgvector semantic search) or filter by status, instrument, date, or outcome.

---

## 5. Falsification Conditions & Hypothesis Monitoring

This is a core differentiating feature of the system.

### 5.1 What a Falsification Condition Is

Every approved thesis must have at least one falsification condition — a discrete, programmatically testable statement about the world that, if violated, signals the thesis is no longer valid. Conditions must have a measurable proxy. Qualitative beliefs must be expressed as quantitative triggers.

**Example translations:**
- "The Fed is sounding dovish" → "Fed Funds futures pricing >2 cuts over next 3 meetings"
- "Treasuries rally after tariff announcements" → "TLT 5-day return after tariff event > 0.5%"
- "Yield curve is steepening" → "10Y-2Y spread > -0.1% (no longer inverted)"

### 5.2 Condition Types

Each condition has a `condition_type` that determines when it evaluates:

- **`state`** — a continuous world condition evaluated on every daily sweep (e.g., "10Y yield above 4.5%").
- **`event`** — only evaluates when a specific trigger event has occurred since the last evaluation. The daily sweep touches the condition but skips evaluation logic unless the event has fired.

Each `event` condition carries a `trigger_type` field that maps to an event detection mechanism (see Section 5.3).

### 5.3 Event Detection

Two mechanisms run in parallel — one for scheduled events, one for unscheduled:

**Scheduled macro events** (CPI, FOMC, NFP, PMI, etc.):
The system maintains an `economic_calendar` table populated from the FRED release calendar. The daily monitoring job checks whether a relevant scheduled release has occurred since each condition was last evaluated. If today is post-release and the condition's `trigger_type` matches the release type, the condition evaluates. This is deterministic and requires no inference.

Supported scheduled trigger types: `CPI_RELEASE`, `FOMC_DECISION`, `NFP_RELEASE`, `PMI_RELEASE`, `GDP_RELEASE`, `PCE_RELEASE`.

**Unscheduled events** (tariff announcements, geopolitical developments, surprise Fed statements):
The daily job polls the IBKR news API for headlines since the last run. A lightweight LLM classifier reads the headlines and determines whether any match the unscheduled trigger types on active conditions. If the classifier fires, the event is logged to `news_events` and the relevant conditions evaluate.

The classification result (headline, classifier confidence, matched trigger type) is stored in `news_events` and cited in the `condition_evaluations` log. Users can review and override misclassifications.

Supported unscheduled trigger types: `TARIFF_ANNOUNCEMENT`, `FED_SPEECH`, `GEOPOLITICAL_EVENT`, `SURPRISE_RATE_MOVE`.

### 5.4 Daily Monitoring Job

A daily scheduled job (runs at market close) processes all conditions for all active theses:

1. For `state` conditions: evaluate against current data and log result.
2. For `event` conditions:
   - Check `economic_calendar` for relevant scheduled releases since last evaluation.
   - Check `news_events` for relevant unscheduled events since last evaluation.
   - If trigger found: evaluate and log result.
   - If no trigger: log as `no_trigger` and skip.
3. If any condition evaluates as **falsified**: trigger kill authority workflow.
4. Log all results to `condition_evaluations` with full citation of data used.

### 5.5 Kill Authority

Kill authority is configurable per-thesis:

- **`alert_only`** (default for all theses): Send alert email to user. No automatic action. User decides.
- **`auto_close`**: Automatically submit closing order to IBKR when condition is falsified. Send alert email confirming action taken.

The `kill_authority` field lives on the thesis and can be changed by the user at any time via the web UI.

*(v2)* **`auto_reduce`**: Reduce position size by 50% on falsification. Do not fully close.

### 5.6 Alert Delivery

**Primary alert channel: email.** Alerts are sent regardless of whether the user is logged into the web UI. Email is outbound-only in this system — users never reply to emails to take action. All actions (acknowledging alerts, changing kill authority, closing trades) go through the UI.

Alert email contains:
- Thesis title and brief description
- Which falsification condition was violated
- The specific data point that triggered it (with citation)
- Current kill authority setting and action taken (or not taken)
- Direct link to the thesis brief in the web UI

All alerts are logged to the `alerts` table with delivery status and timestamp.

Future channels (v2): Slack webhook, SMS via Twilio. The alert system must be built with a pluggable delivery interface from day one so adding channels does not require changes to the core monitoring logic.

### 5.7 Falsification Condition Lifecycle

**Before a thesis becomes active** (`status: approved`): kill conditions are fully editable. The user can add, remove, and modify conditions via the brief UI. Each condition can be evaluated on demand — a "Test Now" button runs the condition evaluator against current data and shows the result inline (passing or failing, with the current data point and citation). This is the correct time to validate conditions are well-formed and sensible.

**Once a thesis becomes active** (`status: active`): kill conditions are permanently locked. No edits of any kind are permitted.
- The UI does not render edit controls for conditions on active theses
- The API rejects any attempt to modify a condition on an active thesis with a typed error
- The "Test Now" button remains available — evaluating a condition is read-only and permitted at any time

**The only path to changing a condition is close and reopen.** If a condition needs to be corrected after a thesis becomes active, the thesis must be closed first. Closing unlocks the conditions for editing. The user can then modify conditions, re-approve the thesis, and make it active again. This flow is intentional — it forces a conscious decision and keeps the audit trail clean.

Rationale: the value of a rules-based system depends entirely on not changing the rules while the trade is on. Editing conditions on active theses is how discretionary judgment sneaks back in. The close-and-reopen friction is a feature, not a limitation.

### 5.8 Condition Chains (v2)

In v1, all conditions are discrete — a single falsification triggers the kill workflow. The data model must support condition chains from day one: each condition has an optional `chain_operator` field (AND / OR) and `chain_group` to support multi-condition logic in v2 without a schema migration.

### 5.8 Latent Thesis Tracker (v2)

A separate entity (not a thesis) for tracking beliefs about the world that aren't yet actionable: "I believe inflation will re-accelerate, but it hasn't yet. Check back in 12 months." Stored as `watch_conditions` with their own monitoring logic. Does not share the thesis schema. Flagged for v2.

---

## 6. Execution Engine

### 6.1 Broker Integration

**Broker:** Interactive Brokers (IBKR)
**API:** IBKR Client Portal API. **Not TWS. This is a hard requirement.**

Rationale: Client Portal API is the modern, REST-based interface. TWS requires a running desktop application and is not appropriate for a server-based system.

The execution engine must support:
- Submitting limit and market orders
- Retrieving real-time and historical positions
- Fetching account balance and buying power
- Paper trading mode (IBKR paper account) and real trading mode (IBKR real account), switchable per-pod
- Polling IBKR news API for unscheduled event detection (see Section 5.3)

### 6.2 Portfolio & Position Management

The portfolio manager maintains the current target state of the portfolio and computes the delta between target and actual positions.

**Position sizing: risk-based volatility targeting.**

Each approved thesis is sized to contribute a target percentage of portfolio volatility. Dollar allocation is back-calculated from the instrument's realized volatility over a 60-day rolling lookback window.

- **Target volatility contribution per position: 5% of portfolio volatility.** (Configurable in pod settings.)
- **Maximum position size: 25% of NAV.** Hard cap applied after volatility-based sizing.
- If volatility-based sizing would exceed the 25% cap, the position is capped and the excess risk budget is not redistributed in v1.

**Formula:**
```
position_size = (target_vol_contribution * portfolio_nav) / instrument_realized_vol_60d
position_size = min(position_size, 0.25 * portfolio_nav)
```

### 6.3 Rebalancing

**Trigger:** Weekly scheduled rebalance on Monday open (or first trading day of the week).

**Logic:**
1. Compute target weights for all active theses using position sizing formula.
2. Diff target weights against current IBKR positions.
3. **Minimum rebalancing threshold: 1% of NAV per position.** Diffs below this threshold are skipped to avoid unnecessary small trades.
4. Submit rebalancing orders for positions exceeding the threshold.
5. Log all decisions (including skipped diffs) and estimated trading costs.

**Manual override:** User can trigger an ad hoc rebalance at any time from the web UI.

### 6.4 Execution

Orders are submitted to IBKR Client Portal API. For v1:
- Use limit orders at mid-price for all ETF trades
- Fill timeout: 5 minutes. If unfilled, cancel and resubmit as market order.
- Log all execution details: submitted price, fill price, fill time, slippage.

### 6.5 Paper vs. Real Mode

Each pod has a `trading_mode` field: `paper` or `real`. This is an explicit, auditable database field — not an environment variable or code flag. `trading_mode` defaults to `paper` on pod creation.

**Switching paper → real:**

1. User initiates switch via UI. A confirmation modal requires the user to type "CONFIRM REAL TRADING" before proceeding.
2. **Audit entry written immediately:** `mode_switch_attempted` — timestamp, user, current positions to be transitioned.
3. Cash check: query real IBKR account for available buying power. Compute total capital required to open all active thesis positions at current volatility-based sizing.
4. **Audit entry written immediately:** `cash_check_passed` or `cash_check_failed` — buying power available, capital required, and result.
5. If cash check fails: **block the switch entirely.** Alert the user with the specific shortfall. No positions are opened or closed. Process ends here.
6. If cash check passes: open positions in real account for all active theses using current volatility-based sizing. Log each fill as it confirms from IBKR.
7. **Audit entry written immediately on confirmed fill:** `real_positions_opened` — instrument, quantity, fill price, thesis_id.
8. Close all paper positions.
9. **Audit entry written immediately per close:** `paper_positions_closed` — instrument, quantity, thesis_id.
10. Update `trading_mode` to `real` in pod_configs.

If any step from 6 onwards fails, the failure is logged and alerted immediately. Earlier steps remain in the audit log. Manual resolution is required — the system does not attempt to auto-rollback.

**Switching real → paper:**

1. User initiates switch via UI. Confirmation modal required.
2. **Audit entry written immediately:** `mode_switch_attempted` — timestamp, user, current real positions to be closed.
3. Close all real positions. Log each fill as it confirms from IBKR.
4. **Audit entry written immediately per close:** `real_positions_closed` — instrument, quantity, fill price, thesis_id.
5. Update `trading_mode` to `paper` in pod_configs.
6. Paper account starts fresh — no positions mirrored. The next scheduled rebalance populates paper positions from current thesis sizing.
7. **Audit entry written immediately:** `mode_switch_completed` — timestamp, new trading_mode.

**General rules:**
- `IBKRClient` reads `trading_mode` from `pod_configs` at request time — never cached at startup
- No position-affecting operation reads `trading_mode` from anywhere other than `pod_configs`
- The UI displays current `trading_mode` prominently on every page

### 6.6 Instrument Roles & Multi-Leg Trades

The `thesis_instruments` table captures the role and direction of each instrument in a thesis. This supports multi-leg trades (spreads, pairs, hedges) in future versions without a schema change.

**Fields:**
- `instrument` — ETF ticker (e.g., "TLT")
- `direction` — enum: `long` / `short`
- `role` — enum: `primary` / `hedge` / `secondary`

In v1 with long-only ETFs, every thesis has one instrument with `direction: long` and `role: primary`. All services that interact with instruments — position sizer, order executor, close trade — must read `direction` and `role` from this table rather than assuming long-only. This ensures short and multi-leg support in v2 is a logic addition, not a structural change.

**Close sequencing for multi-leg trades:** Out of scope for v1. When implemented in v2, all legs are closed simultaneously. If one leg fails, an alert is raised and manual resolution is required. No sophisticated leg sequencing is attempted.

### 6.7 Close Trade

Close trade is a **single, first-class operation** in the system. There is one implementation — a `CloseTrade` service method — that is called by both the human-facing UI and the agent kill authority workflow. There are no separate code paths for human vs. agent closes.

**What it does:**
1. Submits a market order to IBKR to close the full position
2. Updates the thesis status to `closed`
3. Records the close reason: `human_manual`, `kill_condition`, or `auto_close`
4. Writes to `audit_log` with `changed_by` (user ID or agent identifier), timestamp, and close reason
5. Sends a confirmation email to the user

**Human-initiated close (UI):**
- A "Close Trade" button is visible on the trade brief page for any thesis with status `active`
- Clicking triggers a confirmation modal: "Are you sure you want to close this trade? This will submit a market order to close the full position."
- The modal has a Cancel and a Confirm button. No action is taken until Confirm is clicked.
- On confirmation, `CloseTrade` is called with `close_reason: human_manual`

**Agent-initiated close (kill authority):**
- When a falsification condition is triggered and `kill_authority` is set to `auto_close`, the kill authority workflow calls the same `CloseTrade` service method with `close_reason: kill_condition`
- No confirmation step for agent-initiated closes — the kill authority setting is the pre-authorization

**Partial closes:** Out of scope for v1. All closes are full position closes. Partial close support is flagged for v2.

---

## 7. Error Handling Principles

The system does not have fully specified error states for every UI surface — these will be defined iteratively. The following principles govern all error handling and must be applied consistently by Claude Code wherever error states are implemented.

- **No raw errors exposed.** Every error surfaces a human-readable message. No stack traces, no HTTP status codes, no technical details shown to the user.
- **Persistent errors require acknowledgment.** Errors that require user action (a workflow failed, an order was rejected, intake timed out) remain visible until the user explicitly acknowledges them. They are not auto-dismissed.
- **Transient errors offer retry.** Errors caused by external API failures (IBKR unreachable, FRED timeout, OECD unavailable) show a retry option rather than a dead end. The user should never face a permanently broken page due to a recoverable external failure.
- **Workflow failures are partial, not total.** If one workflow step fails during research, the brief is still generated with all available results. The failed step is clearly marked as incomplete. The overall research run does not crash.
- **Consequential actions never fail silently.** Any action with real financial consequences — submitting an order, closing a trade, switching to real mode — must either succeed visibly or fail visibly with an alert. Silent failures on these actions are bugs.
- **External API failures are named.** Error messages tell the user which external system failed (e.g., "Unable to reach IBKR — position data may be stale") rather than generic "something went wrong" messages.
- **Audit entries are written immediately and individually.** Every meaningful step in a multi-step operation gets its own audit_log entry, written the moment that step completes — not batched at the end. A failure mid-process must never prevent earlier steps from being logged. This applies to any consequential multi-step flow: mode switches, close trade, rebalancing.

---

## 8. Performance Measurement

The system tracks performance at both the pod level and the individual thesis level.

**Pod-level metrics (computed daily, stored in `portfolio_snapshots`):**
- NAV and daily NAV change
- Gross and net exposure
- Rolling Sharpe ratio (30d, 90d, 1Y)
- Max drawdown (rolling and since inception)
- Benchmark comparison: SPY and a 60/40 benchmark (SPY 60% + AGG 40%)

**Thesis-level metrics:**
- P&L attributed to each position
- Return vs. backtest expectation
- Days in trade
- Falsification condition status: all passing / N conditions at risk / falsified

**Dashboard:** The web UI home page shows pod-level performance summary and a table of active theses with condition status indicators. Clicking a thesis opens the full trade brief.

---

## 9. Data Architecture

### 8.1 Data Sources

| Source | Purpose | Cost |
|--------|---------|------|
| IBKR Client Portal API | Market data, price history, order execution, news feed | Included with account |
| FRED API | US macro time series: yield curves, CPI, PMI, unemployment, Fed funds futures | Free |
| OECD API | Developed market macro series: ECB rates, Eurozone CPI, non-US yield curves, OECD PMI | Free |
| FRED release calendar | Scheduled economic event dates for event-type condition triggers | Free |
| Web search (Anthropic tool) | News, Fed speeches, analyst commentary during research workflows | Included |
| Research articles & academic papers | Agent-accessed during research workflows via web search | Free |

Bloomberg is out of scope for v1. Revisit if consensus estimate data (survey vs. actual surprise) is needed — FRED and OECD cover non-US developed market macro series adequately for v1.

EM macro data (individual country inflation, rates, current account balances) is out of scope for v1. World Bank API and IMF data API are the natural additions when EM theses become relevant — both are free.

Alternative data (sentiment, positioning, transcripts) is out of scope for v1.

**Web source quality hierarchy:** Primary sources (Fed.gov, BLS, IMF, BIS, academic journals, central bank publications) are preferred over aggregators, news blogs, or opinion pieces. When a web source conflicts with a FRED or IBKR data point, the structured data source takes precedence and the conflict must be noted in the brief.

### 8.2 Data Storage

- **PostgreSQL with pgvector:** All structured data and semantic search. Single database for v1. pgvector extension handles embedding storage for thesis search.
- **Object storage (S3 or local filesystem for early dev):** Full research brief documents and backtest output stored as JSON. Referenced by ID from the `workflow_runs` table.

---

## 10. Configuration Management

### 9.1 Configuration Architecture

There are two distinct categories of configuration in the system:

**Infrastructure config** — database URL, API keys, SMTP credentials, IBKR account IDs. These live in `.env`, are loaded at application startup, and never change at runtime. They are never stored in the database.

**Operational config** — runtime parameters that control system behavior. These live in the `pod_configs` table, are loaded at request time via a `PodSettings` Pydantic model, and can be changed without a code deploy. All changes are written through the audit log.

### 9.2 The `pod_configs` Table

One row per pod. Every operational parameter for the system lives here — nothing is hardcoded in application logic.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trading_mode` | enum | `paper` | Routes IBKR calls to paper or real account |
| `target_vol_per_position` | float | `0.05` | Target volatility contribution per position (5%) |
| `max_position_pct` | float | `0.25` | Maximum position size as % of NAV (25%) |
| `rebalance_threshold_pct` | float | `0.01` | Minimum drift before rebalancing order submitted (1% NAV) |
| `rebalance_day` | int | `0` | Day of week for scheduled rebalance (0 = Monday) |
| `intake_timeout_hours` | int | `24` | Hours before intake proceeds without user confirmation |
| `kill_authority_default` | enum | `alert_only` | Default kill authority for new theses |
| `vol_lookback_days` | int | `60` | Rolling window for realized volatility calculation |

All changes to `pod_configs` values must be written through the audit log in the same transaction, recording the previous value, new value, changed_by, and timestamp.

### 9.3 Data Freshness Policy

**Market and price data (IBKR):** Always use the most recent US trading day's closing data. A `TradingCalendar` utility resolves "most recent trading day" accounting for US market holidays and weekends. This utility is used consistently across all workflows, the daily monitoring job, and the position sizer.

**US macro series (FRED):** Use the most recently available data point for each series. FRED series update on different schedules — some daily, some monthly. The system uses whatever the API returns as the latest observation; it does not attempt to infer or interpolate missing data points.

**Developed market macro series (OECD):** Use the most recently available data point returned by the OECD API. OECD release schedules differ from US trading days; the system does not apply the US trading calendar to OECD data.

**Daily monitoring job on weekends and holidays:** The job still runs but resolves data using the most recent trading day via `TradingCalendar`. State conditions are evaluated against the last available data — they do not skip evaluation on non-trading days.

### 9.4 Loading Config at Runtime

A `PodSettings` Pydantic model is loaded from `pod_configs` at request time and injected into any service or workflow that needs it. No service or workflow reads config values directly from the database — they receive a `PodSettings` object.

The seed script hardcodes the default values in the table above when creating the first pod. These defaults are the only place default values exist — not in application code, not in a config file.

---

## 11. Multi-Pod Architecture (v1 Schema Requirement)

**All entities in the system must belong to a pod from day one.** In v1 there is one pod and one user. The schema must not assume this.

Required:

- A `pods` table is the top-level entity. All theses, positions, trades, and performance data have a `pod_id` foreign key.
- A `users` table with a `pod_memberships` join table (a user can belong to multiple pods; a pod can have multiple users — both human and agent).
- Pod-level configuration stored in `pod_configs` (one row per pod): `trading_mode`, `rebalance_day`, `max_position_pct`, `target_vol_per_position`, `kill_authority_default`, `intake_timeout_hours`, `rebalance_threshold_pct`. All changes to these values are written through the audit log.

**Future state (inform schema, not v1 features):**
- A capital allocation layer above pods that distributes NAV across pods.
- Cross-pod exposure monitoring to prevent inadvertent factor doubling.

---

## 12. Team & Access Model

**v1:** Single human user (the portfolio manager).

**Future team model:** Small team of human analysts + agent workers.
- Human users have role-based access: PM (full), analyst (read + submit ideas), read-only.
- Agent "users" can submit research runs, post workflow outputs to briefs, and trigger workflow steps — but **cannot approve trades, change kill authority settings, or switch trading mode.** Those actions require a human.

**Audit trail:** Every state change on a thesis (status change, kill authority change, go/no-go decision, trading mode change) is logged with `changed_by` (user ID or agent ID) and timestamp. This table is append-only.

---

## 13. Model Selection

Model selection is per-workflow, not global. Each workflow class defines its own model as a configurable attribute. This allows high-judgment tasks to use the most capable model without paying that cost for structured, mechanical tasks.

| Workflow / Task | Model | Rationale |
|----------------|-------|-----------|
| `MacroContextWorkflow` | claude-sonnet-4-6 | Structured data retrieval and summarization |
| `HistoricalAnalogWorkflow` | claude-sonnet-4-6 | Pattern matching against well-defined criteria |
| `InstrumentAnalysisWorkflow` | claude-sonnet-4-6 | Quantitative analysis with structured output |
| `WebResearchWorkflow` | claude-sonnet-4-6 | Search and synthesis of well-scoped sources |
| `BacktestWorkflow` | claude-sonnet-4-6 | Rules-based computation with defined output schema |
| `FalsificationGenerationWorkflow` | claude-opus-4-6 | Requires judgment: translating qualitative assumptions into rigorous falsifiable conditions |
| `RecommendationWorkflow` | claude-opus-4-6 | Requires judgment: synthesizing ambiguous research into a Go/No-Go call |
| Intake conversation | claude-opus-4-6 | Correct thesis interpretation at intake prevents compounding errors downstream |
| LLM event classifier | claude-sonnet-4-6 | Structured classification with a defined label set |
| Deep dive workflows | claude-sonnet-4-6 | Structured analysis with defined outputs |

All Anthropic API calls go through a single `AnthropicClient` wrapper class. No workflow or service calls the Anthropic SDK directly. See CLAUDE.md for implementation details.

**Cost tracking:** Every API call logs token usage and estimated cost to `llm_usage_log`. This table is append-only and designed to support a cost dashboard in v2. No cost UI is built in v1.

---

## 14. Tech Stack

These are **hard requirements**, not suggestions.

| Layer | Requirement | Notes |
|-------|-------------|-------|
| Language | Python 3.11+ | All backend code |
| Web framework | FastAPI | API layer and server-rendered UI |
| Templating | Jinja2 + HTMX | Server-rendered HTML; no separate frontend build |
| Database | PostgreSQL 15+ with pgvector | Single DB for structured data and vector search |
| ORM / migrations | SQLAlchemy + Alembic | Schema migrations must be versioned from day one |
| Agent framework | Anthropic Python SDK | Model is per-workflow — see CLAUDE.md Model Selection section |
| IBKR integration | IBKR Client Portal REST API | Not TWS. Not ib_insync. |
| FRED integration | `fredapi` Python library | |
| Scheduler | APScheduler | Daily monitoring job, weekly rebalance |
| Email | SMTP (via SendGrid or AWS SES) | Alert delivery |
| Object storage | S3 (local filesystem for early dev) | Brief documents and backtest outputs |
| Infrastructure | Docker Compose for local dev | Single container deployment for cloud |
| Dependency management | `uv` with `pyproject.toml` + `uv.lock` | No requirements.txt |

**On the frontend choice:** React is explicitly not used. The rationale: this is a single-user internal tool where the UI is primarily document viewing and occasional form submission. React adds build tooling, state management complexity, and a separate deployment artifact — none of which are justified here. FastAPI + Jinja2 + HTMX produces a fully functional, maintainable UI in a fraction of the time. If the system ever becomes a multi-user product, this decision should be revisited.

---

## 15. v1 Scope Boundaries

**In scope:**
- Hybrid idea input (3 required fields + freeform)
- Single-volley intake conversation via web UI and email
- Single-agent sequential research workflow (7 core workflows)
- Structured WorkflowResult interface enabling future multi-agent migration
- Structured trade brief with Tier 1 summary, Tier 2 deep dives, and Tier 3 Further Reading
- Further Reading: 3-5 sources, cited sources first, one-sentence annotation, all source types
- Discrete falsification conditions with daily monitoring
- State and event-triggered condition types
- Scheduled event detection via FRED calendar
- Unscheduled event detection via IBKR news + LLM classifier
- Alert-only and auto-close kill authority modes
- Email alert delivery with pluggable interface for future channels
- Weekly rebalancing with 1% NAV threshold and manual override
- Risk-based position sizing (5% target vol, 25% max position)
- IBKR Client Portal API integration in paper mode
- Close trade operation (single implementation, callable by human and agent)
- Pod-level and thesis-level performance metrics
- Server-rendered web dashboard (FastAPI + Jinja2 + HTMX)
- Pod-aware schema throughout

**Explicitly out of scope for v1:**
- Multi-agent parallel research execution
- Condition chains (AND/OR multi-condition logic)
- Auto-reduce kill authority mode
- Partial trade closes
- Latent thesis tracker
- Real (live) trading
- Multi-pod UI and capital allocation layer
- Bloomberg or alternative data
- Options, futures, FX — ETFs only
- SMS or Slack alerts
- Multi-user access and roles
- External investor reporting

---

## 16. Resolved Implementation Decisions

| Decision | Resolution |
|----------|-----------|
| IBKR API | Client Portal REST API |
| Target volatility per position | 5% of portfolio volatility |
| Rebalancing threshold | 1% of NAV per position |
| Deep dive workflow library | Sensitivity analysis, regime stress test, portfolio correlation, historical analog detail. Claude Code to scaffold; PM to expand over time. |
| Frontend framework | FastAPI + Jinja2 + HTMX (server-rendered, no React) |
| Unscheduled event data source | IBKR news API + LLM classifier |
| Agent architecture | Single agent, sequential workflow execution. WorkflowResult interface designed for multi-agent migration in v2. |
| Further Reading | 3-5 sources per brief; cited sources ranked first, then additional relevant sources; one-sentence annotation per entry; all source types included. |
| Model selection | Per-workflow. Opus for FalsificationGeneration, Recommendation, and Intake. Sonnet for all other workflows. |
| Cost tracking | Token usage and estimated cost logged to llm_usage_log on every Anthropic API call. Append-only. No UI in v1. |

---

*End of PRD v2.1*
