import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.core.settings import Settings


class TestSettings:
    """Tests for the infrastructure Settings model.

    Settings loads from environment variables at startup. Missing required
    variables must raise clear errors rather than silently defaulting.
    """

    _base_env = {
        "SECRET_KEY": "test-secret",
        "DATABASE_URL": "postgresql://user:pass@localhost/db",
    }

    def test_loads_with_required_vars(self) -> None:
        with patch.dict(os.environ, self._base_env, clear=True):
            settings = Settings(_env_file=None)
        assert settings.secret_key == "test-secret"
        assert settings.database_url == "postgresql://user:pass@localhost/db"

    def test_raises_on_missing_secret_key(self) -> None:
        env = {"DATABASE_URL": "postgresql://user:pass@localhost/db"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings(_env_file=None)
        error_fields = [e["loc"][0] for e in exc_info.value.errors()]
        assert "secret_key" in error_fields

    def test_raises_on_missing_database_url(self) -> None:
        env = {"SECRET_KEY": "test-secret"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings(_env_file=None)
        error_fields = [e["loc"][0] for e in exc_info.value.errors()]
        assert "database_url" in error_fields

    def test_optional_fields_default_to_none(self) -> None:
        with patch.dict(os.environ, self._base_env, clear=True):
            settings = Settings(_env_file=None)
        assert settings.anthropic_api_key is None
        assert settings.fred_api_key is None
        assert settings.ibkr_base_url is None

    def test_smtp_port_defaults_to_587(self) -> None:
        with patch.dict(os.environ, self._base_env, clear=True):
            settings = Settings(_env_file=None)
        assert settings.smtp_port == 587
