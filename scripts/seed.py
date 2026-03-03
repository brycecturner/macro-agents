"""Seed script — creates the default pod, user, and pod_configs row.

Run once after migrations on a fresh database:

    uv run python scripts/seed.py

Safe to run multiple times — checks for existing data before inserting.
Default values are sourced from PRD Section 9.2.
"""

import sys
import uuid
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import get_session_factory
from app.models.enums import KillAuthority, PodMembershipRole, TradingMode
from app.models.pod import Pod, PodConfig, PodMembership, User

# Default pod_configs values from PRD Section 9.2
DEFAULT_POD_NAME = "Default Pod"
DEFAULT_USER_NAME = "Portfolio Manager"
DEFAULT_USER_EMAIL = "pm@localhost"

POD_CONFIG_DEFAULTS = {
    "trading_mode": TradingMode.paper,
    "target_vol_per_position": 0.05,  # 5% of portfolio volatility
    "max_position_pct": 0.25,  # 25% of NAV
    "rebalance_threshold_pct": 0.01,  # 1% of NAV minimum drift
    "rebalance_day": 0,  # 0 = Monday
    "intake_timeout_hours": 24,
    "kill_authority_default": KillAuthority.alert_only,
    "vol_lookback_days": 60,
}


def seed() -> None:
    session_factory = get_session_factory()
    with session_factory() as session:
        # Check if a pod already exists — skip if already seeded
        existing_pod = session.query(Pod).first()
        if existing_pod is not None:
            print(
                f"Database already seeded (pod '{existing_pod.name}' exists). Skipping."
            )
            return

        pod_id = uuid.uuid4()
        user_id = uuid.uuid4()

        pod = Pod(
            id=pod_id,
            name=DEFAULT_POD_NAME,
        )
        session.add(pod)

        user = User(
            id=user_id,
            name=DEFAULT_USER_NAME,
            email=DEFAULT_USER_EMAIL,
        )
        session.add(user)

        pod_config = PodConfig(
            id=uuid.uuid4(),
            pod_id=pod_id,
            **POD_CONFIG_DEFAULTS,
        )
        session.add(pod_config)

        membership = PodMembership(
            id=uuid.uuid4(),
            pod_id=pod_id,
            user_id=user_id,
            role=PodMembershipRole.pm,
        )
        session.add(membership)

        session.commit()

        print(f"Seeded pod '{DEFAULT_POD_NAME}' (id={pod_id})")
        print(
            f"Seeded user '{DEFAULT_USER_NAME}' <{DEFAULT_USER_EMAIL}> (id={user_id})"
        )
        print("Seeded pod_configs with defaults from PRD Section 9.2:")
        for key, value in POD_CONFIG_DEFAULTS.items():
            display = value.value if hasattr(value, "value") else value
            print(f"  {key} = {display}")


if __name__ == "__main__":
    seed()
