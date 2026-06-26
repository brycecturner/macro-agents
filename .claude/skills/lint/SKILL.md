---
description: Run Black formatter and Ruff linter over the project. Use when asked to lint, format, check style, or before committing.
allowed-tools: Bash
---

Run the project's two required code quality tools in order:

```
uv run black . && uv run ruff check .
```

Both must pass with zero errors before any commit (per CLAUDE.md).

If Black reports changes, show which files were reformatted.
If Ruff reports errors, show the full error list and fix them before reporting success.
Do not report the task as done until both commands exit with code 0.
