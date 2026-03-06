"""Job: sync FRED release dates into the economic_calendar table.

Fetches upcoming and recent release dates for all supported scheduled trigger
types and upserts them into economic_calendar. Safe to run repeatedly —
existing rows are updated in place rather than duplicated.

Intended to run on startup and then on a monthly schedule.
"""

import logging
import uuid
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.core.constants import (
    FRED_RELEASE_IDS,
    SYNC_LOOKAHEAD_DAYS,
    SYNC_LOOKBACK_DAYS,
)
from app.integrations.fred_client import FREDClient, FREDClientError
from app.models.monitoring import EconomicCalendar

logger = logging.getLogger(__name__)


def sync_economic_calendar(db: Session, fred_client: FREDClient) -> dict[str, int]:
    """Sync FRED release dates into the economic_calendar table.

    For each supported trigger type, fetches release dates from FRED within a
    rolling window of SYNC_LOOKBACK_DAYS in the past and SYNC_LOOKAHEAD_DAYS
    in the future. Upserts each date:
    - If the row does not exist, it is inserted.
    - If the row already exists, actual_date is updated if needed.

    Past dates (scheduled_date <= today) have actual_date set to scheduled_date.
    Future dates have actual_date = None.

    Args:
        db: Active database session. Caller is responsible for committing.
        fred_client: Initialised FREDClient instance.

    Returns:
        Dict with counts: {"inserted": N, "updated": N, "skipped": N}
    """
    today = date.today()
    window_start = today - timedelta(days=SYNC_LOOKBACK_DAYS)
    window_end = today + timedelta(days=SYNC_LOOKAHEAD_DAYS)

    counts = {"inserted": 0, "updated": 0, "skipped": 0}

    for trigger_type, release_id in FRED_RELEASE_IDS.items():
        try:
            result = fred_client.get_release_dates(release_id)
        except FREDClientError as exc:
            logger.error(
                "Failed to fetch release dates for %s (release_id=%d): %s",
                trigger_type,
                release_id,
                exc,
            )
            continue

        in_window = [d for d in result.dates if window_start <= d <= window_end]
        logger.debug(
            "%s: %d dates from FRED, %d in window",
            trigger_type,
            len(result.dates),
            len(in_window),
        )

        for release_date in in_window:
            actual_date = release_date if release_date <= today else None
            _upsert_release(db, trigger_type, release_date, actual_date, counts)

    db.flush()
    logger.info(
        "Economic calendar sync complete — inserted=%d updated=%d skipped=%d",
        counts["inserted"],
        counts["updated"],
        counts["skipped"],
    )
    return counts


def _upsert_release(
    db: Session,
    release_type: str,
    scheduled_date: date,
    actual_date: date | None,
    counts: dict[str, int],
) -> None:
    existing = (
        db.query(EconomicCalendar)
        .filter_by(release_type=release_type, scheduled_date=scheduled_date)
        .first()
    )

    if existing is None:
        db.add(
            EconomicCalendar(
                id=uuid.uuid4(),
                release_type=release_type,
                scheduled_date=scheduled_date,
                actual_date=actual_date,
            )
        )
        counts["inserted"] += 1
    elif existing.actual_date != actual_date:
        # A previously future date has now passed — populate actual_date
        existing.actual_date = actual_date
        counts["updated"] += 1
    else:
        counts["skipped"] += 1
