---
description: Finalize a completed ticket — mark it [DONE] in implementation_tickets.txt, run lint, commit, and push. Use when a ticket is fully implemented and tests pass.
allowed-tools: Read, Edit, Bash
---

Finalize the current ticket and push to remote. Steps in order:

1. **Identify the ticket.** Use $ARGUMENTS if provided (e.g. `/finalize TICKET-011b`). Otherwise infer from the current conversation — look at what was just implemented.

2. **Mark [DONE] in tickets/implementation_tickets.txt.** Find the ticket line and change its status tag to `[DONE]`. If it has `[NEXT]`, replace that too.

3. **Run lint.** Execute:
   ```
   uv run black . && uv run ruff check .
   ```
   Fix any errors before proceeding. Do not commit with lint failures.

4. **Run tests** for any files changed during the ticket:
   ```
   uv run pytest tests/ -q
   ```
   Do not commit if tests are failing.

5. **Commit.** Stage all modified files and commit using the project format from CLAUDE.md:
   ```
   <type>(TICKET-###): <ticket title> — <short description of change>
   ```
   Types: feat, fix, chore, test, refactor, docs.
   Always append: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`

6. **Push.**
   ```
   git push
   ```

7. Report what was committed and pushed, and what ticket is now `[NEXT]` if one exists.
