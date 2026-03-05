import uuid
from typing import Annotated

from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.enums import KillAuthority, TradingMode
from app.models.pod import PodConfig


class PodSettings(BaseModel):
    """Operational parameters for a pod, loaded from pod_configs at request time.

    Services and workflows receive this model — they never read pod_configs
    directly. This ensures all operational parameters come from one place and
    can be injected in tests without a database.
    """

    pod_id: uuid.UUID
    trading_mode: TradingMode
    target_vol_per_position: float
    max_position_pct: float
    rebalance_threshold_pct: float
    rebalance_day: int
    intake_timeout_hours: int
    kill_authority_default: KillAuthority
    vol_lookback_days: int

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm(cls, config: PodConfig) -> "PodSettings":
        return cls(
            pod_id=config.pod_id,
            trading_mode=config.trading_mode,
            target_vol_per_position=config.target_vol_per_position,
            max_position_pct=config.max_position_pct,
            rebalance_threshold_pct=config.rebalance_threshold_pct,
            rebalance_day=config.rebalance_day,
            intake_timeout_hours=config.intake_timeout_hours,
            kill_authority_default=config.kill_authority_default,
            vol_lookback_days=config.vol_lookback_days,
        )


def get_pod_settings(db: Annotated[Session, Depends(get_db)]) -> PodSettings:
    """FastAPI dependency that loads PodSettings from the database.

    In v1 there is one pod, so this loads the first pod_configs row.
    Future multi-pod support would require a pod_id parameter.
    """
    config = db.query(PodConfig).first()
    if config is None:
        raise RuntimeError(
            "No pod configuration found. Run the seed script to initialize the pod."
        )
    return PodSettings.from_orm(config)
