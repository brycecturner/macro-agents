"""Deep dive trigger route — Tier 2 (PRD Section 4.4).

User-initiated only — deep dives are never generated automatically. Runs a
single named deep dive workflow against the current thesis and returns an
HTML fragment appended to the brief page via HTMX, so triggering one never
reloads the page.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.pod_settings import PodSettings
from app.models.pod import PodConfig
from app.models.thesis import Thesis
from app.services.deep_dive_service import UnknownDeepDiveError, run_deep_dive

logger = logging.getLogger(__name__)

router = APIRouter(tags=["deep-dives"])

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


@router.post(
    "/theses/{thesis_id}/deep-dives/{workflow_name}/run",
    response_class=HTMLResponse,
)
def run_deep_dive_route(
    request: Request,
    thesis_id: uuid.UUID,
    workflow_name: str,
    db: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if thesis is None:
        raise HTTPException(status_code=404, detail="Thesis not found")

    config = db.query(PodConfig).filter(PodConfig.pod_id == thesis.pod_id).first()
    pod_settings = PodSettings.from_orm(config) if config else None

    try:
        result = run_deep_dive(thesis, workflow_name, db, pod_settings)
    except UnknownDeepDiveError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return templates.TemplateResponse(
        request,
        "theses/_deep_dive_result.html",
        {"result": result},
    )
