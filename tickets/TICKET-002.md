# TICKET 002 — Database Schema & Migrations

**Section:** 1 — Project Foundation

## Acceptance Criteria

- Alembic initialized and connected to the PostgreSQL instance
- All tables from PRD Section 4.6 created via versioned migration files:
  pods, users, pod_memberships, theses, thesis_instruments,
  falsification_conditions, condition_evaluations, workflow_registry,
  workflow_runs, further_reading, positions, trades, portfolio_snapshots,
  economic_calendar, news_events, alerts
- All tables have a pod_id foreign key where specified in the PRD
- audit_log table exists and is append-only (no update/delete permissions)
- falsification_conditions table includes chain_operator and chain_group
  fields (nullable, for v2 use)
- pgvector extension enabled; theses table has an embedding column
- Migration runs cleanly from zero on a fresh database
- Seed script creates one default pod, one default user, and one pod_configs row with all default values from PRD Section 9.2
- pod_configs table has one row per pod with all parameters from PRD Section 9.2
