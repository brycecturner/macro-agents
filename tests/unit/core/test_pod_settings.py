import uuid
from unittest.mock import MagicMock

import pytest

from app.core.pod_settings import PodSettings, get_pod_settings
from app.models.enums import KillAuthority, TradingMode


def _make_pod_config(**overrides: object) -> MagicMock:
    """Build a MagicMock that looks like a PodConfig ORM object."""
    defaults = {
        "id": uuid.uuid4(),
        "pod_id": uuid.uuid4(),
        "trading_mode": TradingMode.paper,
        "target_vol_per_position": 0.05,
        "max_position_pct": 0.25,
        "rebalance_threshold_pct": 0.01,
        "rebalance_day": 0,
        "intake_timeout_hours": 24,
        "kill_authority_default": KillAuthority.alert_only,
        "vol_lookback_days": 60,
    }
    defaults.update(overrides)
    config = MagicMock()
    for attr, val in defaults.items():
        setattr(config, attr, val)
    return config


class TestPodSettings:
    """Tests for PodSettings Pydantic model and its FastAPI dependency."""

    def test_from_orm_maps_all_fields(self) -> None:
        config = _make_pod_config()
        settings = PodSettings.from_orm(config)

        assert settings.pod_id == config.pod_id
        assert settings.trading_mode == TradingMode.paper
        assert settings.target_vol_per_position == 0.05
        assert settings.max_position_pct == 0.25
        assert settings.rebalance_threshold_pct == 0.01
        assert settings.rebalance_day == 0
        assert settings.intake_timeout_hours == 24
        assert settings.kill_authority_default == KillAuthority.alert_only
        assert settings.vol_lookback_days == 60

    def test_from_orm_real_trading_mode(self) -> None:
        config = _make_pod_config(trading_mode=TradingMode.real)
        settings = PodSettings.from_orm(config)
        assert settings.trading_mode == TradingMode.real

    def test_from_orm_auto_close_kill_authority(self) -> None:
        config = _make_pod_config(kill_authority_default=KillAuthority.auto_close)
        settings = PodSettings.from_orm(config)
        assert settings.kill_authority_default == KillAuthority.auto_close

    def test_get_pod_settings_dependency_returns_pod_settings(self) -> None:
        config = _make_pod_config()
        mock_db = MagicMock()
        mock_db.query.return_value.first.return_value = config

        settings = get_pod_settings(db=mock_db)

        assert isinstance(settings, PodSettings)
        assert settings.pod_id == config.pod_id

    def test_get_pod_settings_dependency_raises_when_no_config(self) -> None:
        mock_db = MagicMock()
        mock_db.query.return_value.first.return_value = None

        with pytest.raises(RuntimeError, match="No pod configuration found"):
            get_pod_settings(db=mock_db)
