import uuid
from unittest.mock import MagicMock

import pytest

from app.models.enums import KillAuthority, TradingMode
from app.models.log import AuditLog
from app.models.pod import PodConfig
from app.services.pod_config_service import PodConfigService


def _make_pod_config(pod_id: uuid.UUID) -> MagicMock:
    config = MagicMock(spec=PodConfig)
    config.id = uuid.uuid4()
    config.pod_id = pod_id
    config.trading_mode = TradingMode.paper
    config.target_vol_per_position = 0.05
    config.max_position_pct = 0.25
    config.rebalance_threshold_pct = 0.01
    config.rebalance_day = 0
    config.intake_timeout_hours = 24
    config.kill_authority_default = KillAuthority.alert_only
    config.vol_lookback_days = 60
    return config


def _make_db(config: MagicMock) -> MagicMock:
    """Build a mock session that returns config from the query chain."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = config
    return db


class TestPodConfigService:
    """Tests for PodConfigService — audited pod_configs mutations."""

    @pytest.fixture
    def pod_id(self) -> uuid.UUID:
        return uuid.uuid4()

    @pytest.fixture
    def service(self) -> PodConfigService:
        return PodConfigService()

    def test_update_returns_updated_config(
        self, service: PodConfigService, pod_id: uuid.UUID
    ) -> None:
        config = _make_pod_config(pod_id)
        db = _make_db(config)

        result = service.update(db, pod_id, "user-123", vol_lookback_days=90)

        assert result is config
        assert config.vol_lookback_days == 90

    def test_update_writes_audit_log(
        self, service: PodConfigService, pod_id: uuid.UUID
    ) -> None:
        config = _make_pod_config(pod_id)
        db = _make_db(config)
        added: list[object] = []
        db.add.side_effect = added.append

        service.update(db, pod_id, "user-123", vol_lookback_days=90)

        assert len(added) == 1
        audit = added[0]
        assert isinstance(audit, AuditLog)
        assert audit.entity_type == "pod_configs"
        assert audit.action == "config_updated"
        assert audit.changed_by == "user-123"
        assert audit.pod_id == pod_id
        assert audit.entity_id == config.id

    def test_update_records_previous_and_new_value(
        self, service: PodConfigService, pod_id: uuid.UUID
    ) -> None:
        config = _make_pod_config(pod_id)
        db = _make_db(config)
        added: list[AuditLog] = []
        db.add.side_effect = added.append

        service.update(db, pod_id, "system", intake_timeout_hours=48)

        audit = added[0]
        assert audit.previous_value == {"intake_timeout_hours": 24}
        assert audit.new_value == {"intake_timeout_hours": 48}

    def test_update_flushes_session(
        self, service: PodConfigService, pod_id: uuid.UUID
    ) -> None:
        config = _make_pod_config(pod_id)
        db = _make_db(config)

        service.update(db, pod_id, "user-123", rebalance_day=1)

        db.flush.assert_called_once()

    def test_update_multiple_fields_single_audit_entry(
        self, service: PodConfigService, pod_id: uuid.UUID
    ) -> None:
        config = _make_pod_config(pod_id)
        db = _make_db(config)
        added: list[object] = []
        db.add.side_effect = added.append

        service.update(
            db,
            pod_id,
            "user-123",
            rebalance_day=2,
            intake_timeout_hours=12,
        )

        # One audit entry covering both fields
        assert len(added) == 1
        audit = added[0]
        assert "rebalance_day" in audit.previous_value
        assert "intake_timeout_hours" in audit.previous_value

    def test_update_rejects_unknown_field(
        self, service: PodConfigService, pod_id: uuid.UUID
    ) -> None:
        config = _make_pod_config(pod_id)
        db = _make_db(config)

        with pytest.raises(ValueError, match="Unknown pod_configs field"):
            service.update(db, pod_id, "user-123", nonexistent_field=99)

    def test_update_raises_when_no_config_found(
        self, service: PodConfigService, pod_id: uuid.UUID
    ) -> None:
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(RuntimeError, match="No pod_configs row found"):
            service.update(db, pod_id, "user-123", vol_lookback_days=90)

    def test_update_raises_on_empty_kwargs(
        self, service: PodConfigService, pod_id: uuid.UUID
    ) -> None:
        config = _make_pod_config(pod_id)
        db = _make_db(config)

        with pytest.raises(ValueError, match="At least one field"):
            service.update(db, pod_id, "user-123")

    def test_update_enum_field_serialised_as_string_in_audit(
        self, service: PodConfigService, pod_id: uuid.UUID
    ) -> None:
        config = _make_pod_config(pod_id)
        db = _make_db(config)
        added: list[AuditLog] = []
        db.add.side_effect = added.append

        service.update(
            db, pod_id, "user-123", kill_authority_default=KillAuthority.auto_close
        )

        audit = added[0]
        # Enum values must be stored as plain strings for JSON serialisation
        assert audit.previous_value["kill_authority_default"] == "alert_only"
        assert audit.new_value["kill_authority_default"] == "auto_close"
