# MacroContextWorkflow — Prompt Tuning Notes

All tuning levers live in `app/workflows/macro_context.py`.

---

## Tuning Levers

### 1. `_SYSTEM_PROMPT`
Controls the LLM's persona, output format, and tone.

Key things to iterate on:
- **Specificity of the JSON schema** — if the model drifts from the two-key format
  (`summary`, `agent_inferences`), tighten the instruction ("Respond with ONLY a
  JSON object...") or add an explicit example.
- **Summary length** — currently "3-5 sentences". Adjust if the output is too
  dense or too thin for downstream use.
- **Agent inference prefix** — currently requires `"[Agent inference]"` as a
  literal prefix. Keep this; it makes inferences easy to filter programmatically.
- **Factual anchoring** — the prompt says "Write factually — reference the data
  values provided." Strengthen this if the model hallucinates numbers.

### 2. `_build_user_message()`
Constructs the user turn from thesis fields + data lines.

Key things to iterate on:
- **Data line format** (`_prompt_line()`) — currently:
  `- T10Y2Y (T10Y2Y): -0.1234 (2024-05-31), YoY change: +0.5000`
  Consider adding human-readable labels (e.g. "10Y-2Y Spread") instead of
  raw FRED series IDs.
- **Number of recent data points in the prompt** — controlled by
  `_PROMPT_RECENT_MONTHS` (currently 6). Increase to show more trend context;
  decrease to reduce token cost. 6 months is a good baseline for trend signals
  without overwhelming the context.
- **Thesis notes** — notes are passed verbatim. If notes are long, consider
  truncating or summarizing before sending to the LLM.

### 3. `_FRED_CORE_SERIES` and `_OECD_CORE_SERIES`
Which series get fetched and included.

Current FRED series:
- `T10Y2Y` — 10Y-2Y yield spread (the yield curve)
- `CPIAUCSL` — US CPI (inflation)
- `FEDFUNDS` — Fed Funds effective rate
- `UNRATE` — US unemployment rate

Potential additions to consider:
- `DGS10` — 10-year Treasury yield (absolute level, not just spread)
- `T10YIE` — 10-year breakeven inflation rate
- `ICSA` — weekly initial jobless claims (higher frequency signal)
- `PPIACO` — PPI (leading inflation indicator)
- `DTWEXBGS` — USD trade-weighted index

Adding more series increases cost (more tokens) and may dilute the LLM's
focus. Add series only when the thesis type clearly benefits from them.

### 4. `_PROMPT_RECENT_MONTHS`
Number of monthly observations sent to the LLM (currently `6`).

- Increase to 12 for trend analysis over a full year.
- Decrease to 3 for cost-sensitive runs where only the latest snapshot matters.
- The full 30-year history is always written to `structured_output.series[id].historical_data`
  regardless of this setting — this only controls what goes into the LLM prompt.

### 5. `max_tokens=1024` in `complete()`
Cap on LLM output tokens.

- 1024 is generous for a JSON object with one paragraph + a few inferences.
- Can safely reduce to 512 for cost savings; increase to 2048 if the model
  is cutting off mid-sentence on complex theses.

---

## Running the Debug CLI

```bash
# Basic run with TOML thesis file
uv run python scripts/run_workflow.py --thesis examples/thesis_yield_curve.toml

# Include OECD data
uv run python scripts/run_workflow.py --thesis examples/thesis_yield_curve.toml --with-oecd

# Inline args (no file needed)
uv run python scripts/run_workflow.py \
    --title "Yield Curve Steepener" \
    --direction long \
    --horizon "6 months" \
    --notes "Long TLT as yield curve steepens."

# Reduce log noise
uv run python scripts/run_workflow.py \
    --thesis examples/thesis_yield_curve.toml --log-level INFO
```

The output sections to focus on during tuning:
- **STRUCTURED OUTPUT** — check `summary` quality and `series` data completeness
- **AGENT INFERENCES** — check that inferences are non-trivial and start with `[Agent inference]`
- **RAW LLM OUTPUT** — verify the model is returning valid JSON consistently
- **llm_usage_log lines** — watch token counts to gauge cost per run

---

## Evaluation Criteria for the Summary

A good MacroContextWorkflow summary should:
1. Reference at least 2-3 specific data values from the prompt (not just adjectives)
2. Connect the macro data to the specific thesis direction (long/short)
3. Be readable as a standalone paragraph — no "as mentioned above" references
4. Not contradict any of the raw data provided
5. Not extrapolate beyond a 6-9 month horizon (match thesis time_horizon)

Agent inferences should:
1. Each start with `[Agent inference]`
2. Add something beyond what's directly observable in the data
   (regime assessment, historical parallel, conditional outlook)
3. Be falsifiable — a good inference can be proven wrong by future data
