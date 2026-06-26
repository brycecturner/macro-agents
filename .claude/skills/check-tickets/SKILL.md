---
description: Read implementation_tickets.txt and report the current ticket status — what's NEXT, what's DONE, and what's in progress. Use when asked about ticket status, what to work on next, or implementation progress.
allowed-tools: Read, Bash
---

Read `tickets/implementation_tickets.txt` and report:

1. **Next up:** Any ticket marked `[NEXT]` — this is what should be worked on now
2. **Recent completions:** The last 3-5 tickets marked `[DONE]` to show recent progress
3. **Open bugs:** Any ticket with `BUG` in the name that is not marked `[DONE]`
4. **Overall progress:** Count of [DONE] vs total tickets

Keep the output concise. For the [NEXT] ticket, include its full acceptance criteria so we can start work immediately.
