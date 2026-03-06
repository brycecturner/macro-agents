"""Project-wide constants.

Operational parameters (target_vol, rebalance thresholds, etc.) live in
pod_configs and are loaded at runtime via PodSettings — not here.

This file is for values that are genuinely constant: external API mappings,
pricing tables that change rarely, and fixed window sizes used by jobs.
"""

# ---------------------------------------------------------------------------
# Anthropic model pricing (USD per token)
# Update when Anthropic changes pricing. Cost is computed at call time so
# historical llm_usage_log records reflect what was actually paid.
# ---------------------------------------------------------------------------

MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input_per_1k": 0.000015, "output_per_1k": 0.000075},
    "claude-sonnet-4-6": {"input_per_1k": 0.000003, "output_per_1k": 0.000015},
}

# ---------------------------------------------------------------------------
# FRED release ID mapping
# Maps our internal trigger type names to FRED's numeric release IDs.
# Used by sync_economic_calendar to fetch scheduled release dates.
#
# VERIFY these IDs against the FRED release catalog before going to production:
#   https://fred.stlouisfed.org/releases
# or via: fred.search_releases("<name>")
#
# IDs marked UNCERTAIN should be confirmed — FRED's release catalog may not
# cleanly represent these event types (e.g. FOMC decisions are Fed policy
# actions, not FRED economic data releases in the traditional sense).
# ---------------------------------------------------------------------------

FRED_RELEASE_IDS: dict[str, int] = {
    # Consumer Price Index — release 10 (BLS)
    "CPI_RELEASE": 10,
    # Employment Situation (Non-Farm Payrolls) — release 50 (BLS)
    "NFP_RELEASE": 50,
    # Gross Domestic Product — release 53 (BEA)
    "GDP_RELEASE": 53,
    # Personal Income and Outlays (PCE) — release 54 (BEA)
    "PCE_RELEASE": 54,
    # FOMC rate decisions — UNCERTAIN: FRED tracks Fed data series but FOMC
    # decision dates may not map cleanly to a single release ID.
    # Candidate: release 392. Verify before production use.
    "FOMC_DECISION": 392,
    # ISM Manufacturing PMI — UNCERTAIN: ISM data is a private release;
    # FRED may carry it under a release ID but coverage is not guaranteed.
    # Candidate: release 279. Verify before production use.
    "PMI_RELEASE": 279,
}

# ---------------------------------------------------------------------------
# Economic calendar sync window
# How far back and forward to pull release dates when syncing.
# ---------------------------------------------------------------------------

SYNC_LOOKBACK_DAYS: int = 365
SYNC_LOOKAHEAD_DAYS: int = 365
