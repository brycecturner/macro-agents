"""Thesis routes — idea input form, detail page, and intake conversation."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.database import get_db, get_session_factory
from app.core.settings import get_settings
from app.models.enums import Direction, KillAuthority, ThesisStatus
from app.models.pod import Pod, PodConfig
from app.models.thesis import Thesis
from app.services.intake_service import IntakeService
from app.services.research_pipeline_service import run_research_pipeline_for_thesis
from app.services.thesis_decision_service import InvalidDecisionError, record_decision

logger = logging.getLogger(__name__)

router = APIRouter(tags=["theses"])

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _get_pod_context(db: Session) -> dict:
    """Fetch pod and pod_config for nav display. Included in every page context."""
    pod = db.query(Pod).first()
    pod_config = (
        db.query(PodConfig).filter(PodConfig.pod_id == pod.id).first() if pod else None
    )
    return {"pod": pod, "pod_config": pod_config}


def _run_intake_generation(thesis_id: uuid.UUID) -> None:
    """Background task: generate intake message for a newly created thesis."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not configured — intake skipped for thesis %s",
            thesis_id,
        )
        return
    db = get_session_factory()()
    try:
        thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
        if thesis is None:
            logger.error("Thesis %s not found for intake generation", thesis_id)
            return
        IntakeService().generate_intake_message(thesis, db, settings.anthropic_api_key)
    except Exception:
        logger.exception("Failed to generate intake message for thesis %s", thesis_id)
    finally:
        db.close()


# ---------------------------------------------------------------------------
# GET /theses/new — blank idea input form
# ---------------------------------------------------------------------------


@router.get("/theses/new", response_class=HTMLResponse)
def new_thesis_form(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "theses/new.html", {"errors": {}, "values": {}, **_get_pod_context(db)}
    )


# ---------------------------------------------------------------------------
# POST /theses — create thesis, trigger intake generation, redirect to detail
# ---------------------------------------------------------------------------


@router.post("/theses", response_class=HTMLResponse)
def create_thesis(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
    background_tasks: BackgroundTasks,
    thesis_title: Annotated[str, Form()] = "",
    time_horizon: Annotated[str, Form()] = "",
    direction: Annotated[str, Form()] = "",
    notes: Annotated[str, Form()] = "",
) -> HTMLResponse:
    errors: dict[str, str] = {}

    title = thesis_title.strip()
    horizon = time_horizon.strip()
    dir_val = direction.strip()
    notes_val = notes.strip()

    if not title:
        errors["thesis_title"] = "Title is required."
    if not horizon:
        errors["time_horizon"] = "Time horizon is required."
    if not dir_val:
        errors["direction"] = "Direction is required."
    elif dir_val not in ("long", "short"):
        errors["direction"] = "Direction must be 'long' or 'short'."
    if not notes_val:
        errors["notes"] = "Notes are required."

    pod_ctx = _get_pod_context(db)
    pod = pod_ctx["pod"]
    pod_config = pod_ctx["pod_config"]

    if pod is None:
        return templates.TemplateResponse(
            request,
            "theses/new.html",
            {
                **pod_ctx,
                "errors": {"__root__": "No pod configured. Run the seed script."},
                "values": {},
            },
            status_code=500,
        )

    if errors:
        return templates.TemplateResponse(
            request,
            "theses/new.html",
            {
                **pod_ctx,
                "errors": errors,
                "values": {
                    "thesis_title": thesis_title,
                    "time_horizon": time_horizon,
                    "direction": direction,
                    "notes": notes,
                },
            },
            status_code=422,
        )

    kill_authority = (
        pod_config.kill_authority_default if pod_config else KillAuthority.alert_only
    )

    thesis = Thesis(
        id=uuid.uuid4(),
        pod_id=pod.id,
        title=title,
        time_horizon=horizon,
        direction=Direction(dir_val),
        notes=notes_val,
        status=ThesisStatus.draft,
        kill_authority=kill_authority,
        thesis_confirmed=True,
    )
    db.add(thesis)
    db.commit()

    background_tasks.add_task(_run_intake_generation, thesis.id)

    return RedirectResponse(url=f"/theses/{thesis.id}", status_code=303)


# ---------------------------------------------------------------------------
# GET /theses/{thesis_id} — thesis detail page
# ---------------------------------------------------------------------------


@router.get("/theses/{thesis_id}", response_class=HTMLResponse)
def thesis_detail(
    request: Request,
    thesis_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if thesis is None:
        raise HTTPException(status_code=404, detail="Thesis not found")
    return templates.TemplateResponse(
        request, "theses/detail.html", {"thesis": thesis, **_get_pod_context(db)}
    )


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/intake-response — user responds to intake message
# ---------------------------------------------------------------------------


@router.post("/theses/{thesis_id}/intake-response", response_class=HTMLResponse)
def intake_response(
    request: Request,
    thesis_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    background_tasks: BackgroundTasks,
    user_response: Annotated[str, Form()] = "",
) -> HTMLResponse:
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if thesis is None:
        raise HTTPException(status_code=404, detail="Thesis not found")
    if thesis.status != ThesisStatus.intake_sent:
        raise HTTPException(
            status_code=409, detail="Thesis is not awaiting an intake response."
        )
    IntakeService().handle_intake_response(thesis, user_response, db)
    background_tasks.add_task(run_research_pipeline_for_thesis, thesis_id)
    return RedirectResponse(url=f"/theses/{thesis_id}", status_code=303)


# ---------------------------------------------------------------------------
# GET /theses/{thesis_id}/brief — structured Tier 1 trade brief (JSON)
# ---------------------------------------------------------------------------


@router.get("/theses/{thesis_id}/brief")
def get_thesis_brief(
    thesis_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if thesis is None:
        raise HTTPException(status_code=404, detail="Thesis not found")
    if thesis.brief is None:
        raise HTTPException(status_code=404, detail="Brief has not been generated yet")
    return thesis.brief


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/decision — human Go / No-Go / Hold decision
# ---------------------------------------------------------------------------


@router.post("/theses/{thesis_id}/decision", response_class=HTMLResponse)
def submit_thesis_decision(
    request: Request,
    thesis_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    decision: Annotated[str, Form()] = "",
) -> HTMLResponse:
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if thesis is None:
        raise HTTPException(status_code=404, detail="Thesis not found")
    if thesis.status != ThesisStatus.researched:
        raise HTTPException(
            status_code=409, detail="Thesis is not awaiting a decision."
        )
    try:
        record_decision(thesis, decision, db)
    except InvalidDecisionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    db.commit()
    return RedirectResponse(url=f"/theses/{thesis_id}", status_code=303)


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/acknowledge-intake — user dismisses the warning banner
# ---------------------------------------------------------------------------


@router.post("/theses/{thesis_id}/acknowledge-intake", response_class=HTMLResponse)
def acknowledge_intake(
    request: Request,
    thesis_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if thesis is None:
        raise HTTPException(status_code=404, detail="Thesis not found")
    IntakeService.acknowledge_intake(thesis, db)
    return RedirectResponse(url=f"/theses/{thesis_id}", status_code=303)
