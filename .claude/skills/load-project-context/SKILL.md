---
description: Load the three core project documents — PRD, CLAUDE.md, and ERD — into context. Use at the start of a session, before starting a new ticket, or when asked to load project context.
allowed-tools: Read
---

Read all three key project documents in full and summarize what was loaded:

1. Read `PRD.md` — product requirements, architecture decisions, and scope boundaries
2. Read `CLAUDE.md` — engineering standards and behavioral rules  
3. Read `ERD.md` — entity relationship diagram with all table definitions and field specs

After reading, confirm all three are loaded and note any sections most relevant to the current conversation if context is available (e.g. if a specific workflow or table is being discussed, call out the relevant PRD section and ERD table).

These are the source of truth hierarchy per CLAUDE.md:
  1. PRD.md — product requirements
  2. CLAUDE.md — engineering standards  
  3. Current ticket — implementation detail
