# Macro Agents

An agent-powered macro hedge fund system. Generates, researches, validates, and monitors macro trade ideas, with execution through Interactive Brokers.

---

## Prerequisites

**To run with Docker:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Docker Compose

**To run locally:**
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd macro-agents
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values. At minimum, you need:

| Variable | Description |
|----------|-------------|
| `SECRET_KEY` | Random hex string — `python -c "import secrets; print(secrets.token_hex(32))"` |
| `DATABASE_URL` | PostgreSQL connection string |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Used by docker-compose to configure the local DB |
| `ANTHROPIC_API_KEY` | Required for all LLM workflows |
| `FRED_API_KEY` | Required for macro data workflows |
| `IBKR_*` | Required for execution and market data |
| `SMTP_*` / `ALERT_EMAIL` | Required for alert delivery |

See `.env.example` for the full list with descriptions.

---

## Running the app

### Option A — Docker Compose (recommended)

```bash
docker compose up --build
```

This starts:
- **app** on `http://localhost:8000` (with hot reload)
- **db** — PostgreSQL with pgvector on port `5432`

The `DATABASE_URL` in `.env` is overridden automatically by `docker-compose.yml` to use the internal `db` hostname.

### Option B — Local development

Install dependencies:

```bash
uv sync
```

Start a local PostgreSQL instance (or point `DATABASE_URL` at an existing one), then run:

```bash
uv run uvicorn app.main:app --reload
```

App is available at `http://localhost:8000`.

---

## Health check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## Running tests

```bash
uv run pytest
```

---

## Linting and formatting

```bash
uv run black . && uv run ruff check .
```

Both must pass with zero errors before committing.

---

## Database migrations

Migrations use Alembic and live in `migrations/`. After the first ticket that sets up the schema:

```bash
uv run alembic upgrade head
```
