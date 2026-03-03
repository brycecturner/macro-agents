# TICKET 037 — Paper vs. Real Mode Enforcement & Position Transition

**Section:** 8 — Execution Engine

## Acceptance Criteria

- trading_mode field on pod_configs controls all IBKR routing
- IBKRClient reads trading_mode from pod_configs at request time — never cached
- UI displays current trading_mode prominently on every page
- Switching trading_mode requires explicit user action via UI confirmation
  modal; user must type "CONFIRM REAL TRADING" to proceed
- Paper → real transition executes in this exact order, each step with its
  own immediate audit_log entry:
    1. mode_switch_attempted logged
    2. Cash check against real IBKR account buying power
    3. cash_check_passed or cash_check_failed logged with amounts
    4. If failed: block switch entirely, alert user with shortfall, stop
    5. If passed: open real positions for all active theses at current sizing
    6. real_positions_opened logged per confirmed IBKR fill
    7. Close all paper positions
    8. paper_positions_closed logged per confirmed close
    9. trading_mode updated to real in pod_configs
- Real → paper transition executes in this exact order:
    1. mode_switch_attempted logged
    2. Close all real positions
    3. real_positions_closed logged per confirmed IBKR fill
    4. trading_mode updated to paper in pod_configs
    5. mode_switch_completed logged
    6. Paper account starts fresh — next rebalance populates paper positions
- Mid-process failures are logged and alerted immediately; no auto-rollback
- All audit entries written immediately when the step completes — never batched
- Tests confirm: cash check blocks switch on insufficient funds, each audit
  entry written at correct step, paper account fresh after real→paper switch,
  IBKRClient routing correct for each mode
