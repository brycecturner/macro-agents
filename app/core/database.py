from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import get_settings


@lru_cache
def get_engine():
    """Return the SQLAlchemy engine, created once and cached."""
    return create_engine(
        get_settings().database_url,
        pool_pre_ping=True,  # verify connections before use
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Return the session factory, created once and cached."""
    return sessionmaker(
        bind=get_engine(),
        autocommit=False,
        autoflush=False,
    )


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session per request."""
    db = get_session_factory()()
    try:
        yield db
    finally:
        db.close()
