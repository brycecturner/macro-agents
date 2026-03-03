# TICKET 001 — Project Scaffold

**Section:** 1 — Project Foundation

## Acceptance Criteria

- Python 3.11+ project initialized with uv; pyproject.toml and uv.lock committed
- Docker Compose file runs the app and a local PostgreSQL instance
- FastAPI app starts and returns 200 on GET /health
- Environment variables loaded from .env file (never hardcoded)
- .env.example committed with all required keys documented and no real values
- No requirements.txt exists in the repository
- README documents how to run the project locally from scratch
