---
description: Run the full workflow pipeline (MacroContext through Recommendation) over a thesis TOML file. Use when asked to run the pipeline, test a thesis, or execute workflows.
allowed-tools: Bash
---

Run the full 7-step workflow pipeline over the specified thesis file.

Usage: /run-pipeline <path-to-thesis-toml>

Example: /run-pipeline examples/short_war_thesis.toml

Command to execute:
```
uv run python scripts/run_workflow.py --workflow RecommendationWorkflow --thesis $ARGUMENTS
```

If no argument is provided, default to `examples/short_war_thesis.toml`.

The pipeline runs these steps in order:
MacroContext → HistoricalAnalog → InstrumentAnalysis → WebResearch → Backtest → FalsificationGeneration → Recommendation

Each step saves full output to `output/<workflow>_<timestamp>.txt`. Known limitations at Tier 1:
- WebResearch sleeps 65s before annotation to reset the rate limit window
- Queries 2 and 3 may be dropped by 429s (see TICKET-013c)
- HistoricalAnalog may log invalid JSON warning (see TICKET-011b)
