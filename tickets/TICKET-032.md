# TICKET 032 — Position Sizer

**Section:** 8 — Execution Engine

## Acceptance Criteria

- PositionSizer class computes target dollar allocation for a thesis
- Uses formula from PRD Section 6.2:
  position_size = (target_vol * nav) / realized_vol_60d
  position_size = min(position_size, 0.25 * nav)
- target_vol loaded from pod settings (default 5%)
- realized_vol_60d fetched from IBKR price history
- Returns target position size in dollars and number of shares
- Tests confirm correct formula application, cap enforcement, and correct
  behavior when volatility data is unavailable
