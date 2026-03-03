# TICKET 003 — Configuration, Settings & Trading Calendar Module

**Section:** 1 — Project Foundation

## Acceptance Criteria

- Infrastructure config (API keys, DB URL, etc.) loaded from .env at startup via a Settings Pydantic model
- PodSettings Pydantic model loads all operational parameters from pod_configs at request time
- No service or workflow reads pod_configs directly — all receive a PodSettings object
- No magic numbers in application logic; every operational parameter comes from PodSettings
- Changes to pod_configs values write to audit_log in the same transaction
- TradingCalendar utility class exists in app/core/ with method: most_recent_trading_day(as_of: date) -> date
- TradingCalendar accounts for US market holidays and weekends
- TradingCalendar is used consistently by all workflows, the monitoring job, and the position sizer — never inline date logic
- Tests confirm: Settings raises clear errors for missing env vars, PodSettings loads correctly from pod_configs, audit entry written on every pod_configs update, TradingCalendar returns correct dates around holidays and weekends
