# TICKET 025 — Thesis Search & List View

**Section:** 5 — Trade Brief

## Acceptance Criteria

- Thesis list page shows all theses with: title, status, instrument,
  direction, created date, condition status (if active)
- Filterable by status, instrument, and date range
- Keyword search uses pgvector semantic search on thesis content
- Embeddings generated for each thesis on creation using Anthropic embeddings
- Tests confirm filtering, search results, and embedding generation
