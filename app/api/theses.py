"""Thesis routes — idea input form and detail page."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.enums import Direction, KillAuthority, ThesisStatus
from app.models.pod import Pod, PodConfig
from app.models.thesis import Thesis

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
# POST /theses — create thesis, redirect to detail on success
# ---------------------------------------------------------------------------


@router.post("/theses", response_class=HTMLResponse)
def create_thesis(
    request: Request,
    db: Annotated[Session, Depends(get_db)],
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
        intake_unconfirmed=False,
    )
    db.add(thesis)
    db.commit()

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
