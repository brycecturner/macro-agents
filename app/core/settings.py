from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    secret_key: str
    app_host: str = "http://localhost:8000"

    # Database
    database_url: str

    # Anthropic API — required for all LLM workflows
    anthropic_api_key: str | None = None

    # FRED API — required for MacroContext, HistoricalAnalog, Backtest workflows
    fred_api_key: str | None = None

    # IBKR Client Portal API — required for execution and market data
    ibkr_base_url: str | None = None
    ibkr_account_id: str | None = None
    ibkr_paper_account_id: str | None = None

    # Email / SMTP — required for alert delivery
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    alert_email: str | None = None

    # PostgreSQL vars (used by docker-compose; DATABASE_URL is used by the app)
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
