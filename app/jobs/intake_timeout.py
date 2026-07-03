"""APScheduler job: process timed-out intake messages."""

from __future__ import annotations

import logging

from app.core.database import get_session_factory
from app.models.pod import PodConfig
from app.services.intake_service import IntakeService

logger = logging.getLogger(__name__)


def check_intake_timeouts() -> None:
    """Find theses past their intake timeout window and process them.

    Scheduled hourly. Reads intake_timeout_hours from pod_configs so the
    threshold always reflects the current pod setting.
    """
    db = get_session_factory()()
    try:
        config = db.query(PodConfig).first()
        if config is None:
            logger.warning("No pod config found — skipping intake timeout check")
            return
        timeout_hours = config.intake_timeout_hours
        count = IntakeService.check_and_process_timeouts(db, timeout_hours)
        if count:
            logger.info("Intake timeout: processed %d thesis(es)", count)
    except Exception:
        logger.exception("Error in intake timeout job")
    finally:
        db.close()
