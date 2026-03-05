import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.models.log import AuditLog
from app.models.pod import PodConfig

# Fields that callers are permitted to update on pod_configs.
# Prevents arbitrary column writes and documents the public API surface.
_ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        "trading_mode",
        "target_vol_per_position",
        "max_position_pct",
        "rebalance_threshold_pct",
        "rebalance_day",
        "intake_timeout_hours",
        "kill_authority_default",
        "vol_lookback_days",
    }
)


class PodConfigService:
    """Service for updating pod_configs values with mandatory audit logging.

    Every change to a pod_configs field must go through this service so that
    an audit_log entry is written in the same transaction. Callers are
    responsible for committing the session.
    """

    def update(
        self,
        db: Session,
        pod_id: uuid.UUID,
        changed_by: str,
        **kwargs: Any,
    ) -> PodConfig:
        """Update one or more pod_configs fields and write an audit_log entry.

        Args:
            db: The active database session. Caller commits.
            pod_id: UUID of the pod whose config is being updated.
            changed_by: User UUID string or agent identifier (e.g. 'system').
            **kwargs: Field name → new value pairs. Must be in _ALLOWED_FIELDS.

        Returns:
            The updated PodConfig ORM object (not yet committed).

        Raises:
            ValueError: If any kwargs key is not in _ALLOWED_FIELDS.
            RuntimeError: If no pod_configs row exists for the given pod_id.
        """
        unknown = set(kwargs.keys()) - _ALLOWED_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown pod_configs field(s): {sorted(unknown)}. "
                f"Allowed fields: {sorted(_ALLOWED_FIELDS)}"
            )

        if not kwargs:
            raise ValueError("At least one field must be provided to update.")

        config = db.query(PodConfig).filter(PodConfig.pod_id == pod_id).first()
        if config is None:
            raise RuntimeError(f"No pod_configs row found for pod_id={pod_id}")

        # Snapshot previous values for the fields being changed
        previous_value: dict[str, Any] = {
            field: _serialise(getattr(config, field)) for field in kwargs
        }

        # Apply updates
        for field, value in kwargs.items():
            setattr(config, field, value)

        new_value: dict[str, Any] = {
            field: _serialise(getattr(config, field)) for field in kwargs
        }

        # Write audit entry in the same transaction
        audit_entry = AuditLog(
            id=uuid.uuid4(),
            pod_id=pod_id,
            entity_id=config.id,
            entity_type="pod_configs",
            action="config_updated",
            previous_value=previous_value,
            new_value=new_value,
            changed_by=changed_by,
        )
        db.add(audit_entry)

        # Flush so both the config update and audit entry are staged together;
        # the caller commits the transaction.
        db.flush()

        return config


def _serialise(value: Any) -> Any:
    """Convert ORM field values to JSON-serialisable types for audit storage."""
    if hasattr(value, "value"):
        # Enum — store the string value
        return value.value
    if isinstance(value, uuid.UUID):
        return str(value)
    return value
