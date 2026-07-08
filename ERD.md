# Entity Relationship Diagram
## Core Tables

```mermaid
erDiagram
    pods {
        uuid id PK
        string name
        timestamp created_at
    }

    pod_configs {
        uuid id PK
        uuid pod_id FK
        enum trading_mode "paper | real"
        float target_vol_per_position
        float max_position_pct
        float rebalance_threshold_pct
        int rebalance_day
        int intake_timeout_hours
        enum kill_authority_default "alert_only | auto_close"
        int vol_lookback_days
        timestamp updated_at
    }

    users {
        uuid id PK
        string name
        string email
        timestamp created_at
    }

    pod_memberships {
        uuid id PK
        uuid pod_id FK
        uuid user_id FK
        enum role "pm | analyst | readonly"
        timestamp created_at
    }

    theses {
        uuid id PK
        uuid pod_id FK
        string title
        string time_horizon
        enum direction "long | short"
        text notes
        enum status "draft | intake_sent | researched | approved | active | closed | rejected"
        enum kill_authority "alert_only | auto_close"
        boolean thesis_confirmed
        vector embedding
        jsonb brief "nullable — assembled Tier 1 trade brief"
        timestamp brief_generated_at "nullable"
        timestamp closed_at
        timestamp created_at
    }

    thesis_instruments {
        uuid id PK
        uuid thesis_id FK
        string instrument
        enum direction "long | short"
        enum role "primary | hedge | secondary"
    }

    falsification_conditions {
        uuid id PK
        uuid thesis_id FK
        text description
        enum condition_type "state | event"
        string trigger_type "nullable"
        text measurable_proxy
        text evaluation_logic
        enum chain_operator "AND | OR | null"
        string chain_group "nullable"
        timestamp last_evaluated_at
        timestamp created_at
    }

    workflow_runs {
        uuid id PK
        uuid thesis_id FK
        string workflow_name
        enum status "completed | failed | partial"
        jsonb structured_output
        jsonb citations
        jsonb agent_inferences
        text raw_output
        timestamp started_at
        timestamp completed_at
    }

    positions {
        uuid id PK
        uuid pod_id FK
        uuid thesis_id FK
        string instrument
        enum direction "long | short"
        float quantity
        float entry_price
        float current_price
        enum trading_mode "paper | real"
        timestamp opened_at
    }

    trades {
        uuid id PK
        uuid pod_id FK
        uuid thesis_id FK
        uuid position_id FK
        string instrument
        enum direction "long | short"
        enum order_type "limit | market"
        float submitted_price
        float fill_price
        float quantity
        float slippage
        enum close_reason "rebalance | kill_condition | auto_close | human_manual | null"
        timestamp fill_time
    }

    pods ||--|| pod_configs : "has one"
    pods ||--o{ pod_memberships : "has many"
    pods ||--o{ theses : "has many"
    pods ||--o{ positions : "has many"
    pods ||--o{ trades : "has many"

    users ||--o{ pod_memberships : "belongs to many"

    theses ||--o{ thesis_instruments : "has many"
    theses ||--o{ falsification_conditions : "has many"
    theses ||--o{ workflow_runs : "has many"
    theses ||--o{ positions : "has many"
    theses ||--o{ trades : "has many"

    audit_log {
        uuid id PK
        uuid pod_id FK
        uuid entity_id "UUID of the changed entity"
        string entity_type "e.g. thesis, pod_configs, position"
        string action "e.g. mode_switch_attempted, cash_check_passed, status_changed"
        jsonb previous_value "nullable"
        jsonb new_value "nullable"
        string changed_by "user UUID or agent identifier"
        timestamp created_at
    }

    llm_usage_log {
        uuid id PK
        uuid pod_id FK "nullable — via thesis"
        uuid thesis_id FK "nullable"
        uuid workflow_run_id FK "nullable"
        string model "e.g. claude-opus-4-6"
        string task_type "e.g. intake, macro_context, recommendation"
        int input_tokens
        int output_tokens
        float estimated_cost_usd
        timestamp called_at
    }

    positions ||--o{ trades : "has many"

    pods ||--o{ audit_log : "has many"
    pods ||--o{ llm_usage_log : "has many"
    theses ||--o{ llm_usage_log : "has many"
    workflow_runs ||--o{ llm_usage_log : "has many"
```
```

---

## Notes

- All PKs are UUID v4, generated in the application layer before database write
- `pods` is the top-level entity — all other core tables have a `pod_id` FK either directly or via `theses`
- `pod_configs` is one-to-one with `pods` — created by the seed script on pod creation with default values
- `falsification_conditions.chain_operator` and `chain_group` are nullable — reserved for v2 condition chain logic
- `trades.close_reason` is nullable — only populated on closing trades, not opening trades
- `theses.embedding` is a pgvector column for semantic thesis search
- `theses.brief` is a JSONB snapshot of the assembled Tier 1 trade brief, regenerated whenever the research pipeline completes; referenced directly by `thesis_id` rather than a separate object store (see PRD Section 8.2)
- `workflow_runs` stores full agent output as JSONB — raw_output is kept for debugging but not passed between workflow steps; only structured_output is consumed downstream
- `positions.trading_mode` records which account the position lives in at time of opening — this is a snapshot, not a live reference to pod_configs
- `audit_log` is append-only — no UPDATE or DELETE permitted at the database level; every step in a multi-step operation gets its own row written immediately
- `audit_log.entity_id` + `entity_type` together identify what changed; `action` names the specific step (e.g. `mode_switch_attempted`, `cash_check_failed`, `thesis_status_changed`)
- `llm_usage_log` is append-only — records every Anthropic API call with token counts and estimated cost at time of call; all FKs are nullable since not all calls originate from a workflow run
