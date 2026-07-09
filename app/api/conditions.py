"""Falsification condition CRUD routes and on-demand Test Now evaluation.

Per PRD Section 5.7, conditions are only editable while a thesis is
'approved' — the window between research completing and the thesis going
active. Once active (or any other status), create/update/delete are
rejected with a typed ConditionLockedError (409). Test Now is exempt from
the lock: it is a read-only, on-demand evaluation available at any thesis
status, rendered as an HTMX partial swap.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.thesis import FalsificationCondition, Thesis
from app.schemas.condition import (
    FalsificationConditionCreate,
    FalsificationConditionUpdate,
)
from app.services.condition_service import (
    ConditionLockedError,
    create_condition,
    delete_condition,
    test_now,
    update_condition,
)

router = APIRouter(tags=["conditions"])

_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _get_thesis_or_404(thesis_id: uuid.UUID, db: Session) -> Thesis:
    thesis = db.query(Thesis).filter(Thesis.id == thesis_id).first()
    if thesis is None:
        raise HTTPException(status_code=404, detail="Thesis not found")
    return thesis


def _get_condition_or_404(
    thesis_id: uuid.UUID, condition_id: uuid.UUID, db: Session
) -> FalsificationCondition:
    condition = (
        db.query(FalsificationCondition)
        .filter(
            FalsificationCondition.id == condition_id,
            FalsificationCondition.thesis_id == thesis_id,
        )
        .first()
    )
    if condition is None:
        raise HTTPException(status_code=404, detail="Condition not found")
    return condition


def _validation_error_detail(exc: ValidationError) -> str:
    return "; ".join(f"{e['loc'][-1]}: {e['msg']}" for e in exc.errors())


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/conditions — create
# ---------------------------------------------------------------------------


@router.post("/theses/{thesis_id}/conditions", response_class=HTMLResponse)
def create_condition_route(
    thesis_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    description: Annotated[str, Form()] = "",
    condition_type: Annotated[str, Form()] = "",
    trigger_type: Annotated[str, Form()] = "",
    measurable_proxy: Annotated[str, Form()] = "",
    evaluation_logic: Annotated[str, Form()] = "",
) -> HTMLResponse:
    thesis = _get_thesis_or_404(thesis_id, db)
    try:
        data = FalsificationConditionCreate(
            description=description,
            condition_type=condition_type,
            trigger_type=trigger_type,
            measurable_proxy=measurable_proxy,
            evaluation_logic=evaluation_logic,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=_validation_error_detail(exc)
        ) from exc

    try:
        create_condition(thesis, data, db)
    except ConditionLockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    return RedirectResponse(url=f"/theses/{thesis_id}", status_code=303)


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/conditions/{condition_id}/update
# ---------------------------------------------------------------------------


@router.post(
    "/theses/{thesis_id}/conditions/{condition_id}/update",
    response_class=HTMLResponse,
)
def update_condition_route(
    thesis_id: uuid.UUID,
    condition_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
    description: Annotated[str, Form()] = "",
    condition_type: Annotated[str, Form()] = "",
    trigger_type: Annotated[str, Form()] = "",
    measurable_proxy: Annotated[str, Form()] = "",
    evaluation_logic: Annotated[str, Form()] = "",
) -> HTMLResponse:
    thesis = _get_thesis_or_404(thesis_id, db)
    condition = _get_condition_or_404(thesis_id, condition_id, db)
    try:
        data = FalsificationConditionUpdate(
            description=description,
            condition_type=condition_type,
            trigger_type=trigger_type,
            measurable_proxy=measurable_proxy,
            evaluation_logic=evaluation_logic,
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=_validation_error_detail(exc)
        ) from exc

    try:
        update_condition(thesis, condition, data)
    except ConditionLockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    return RedirectResponse(url=f"/theses/{thesis_id}", status_code=303)


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/conditions/{condition_id}/delete
# ---------------------------------------------------------------------------


@router.post(
    "/theses/{thesis_id}/conditions/{condition_id}/delete",
    response_class=HTMLResponse,
)
def delete_condition_route(
    thesis_id: uuid.UUID,
    condition_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    thesis = _get_thesis_or_404(thesis_id, db)
    condition = _get_condition_or_404(thesis_id, condition_id, db)
    try:
        delete_condition(thesis, condition, db)
    except ConditionLockedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    db.commit()
    return RedirectResponse(url=f"/theses/{thesis_id}", status_code=303)


# ---------------------------------------------------------------------------
# POST /theses/{thesis_id}/conditions/{condition_id}/test-now — HTMX partial
# ---------------------------------------------------------------------------


@router.post(
    "/theses/{thesis_id}/conditions/{condition_id}/test-now",
    response_class=HTMLResponse,
)
def test_now_route(
    request: Request,
    thesis_id: uuid.UUID,
    condition_id: uuid.UUID,
    db: Annotated[Session, Depends(get_db)],
) -> HTMLResponse:
    _get_thesis_or_404(thesis_id, db)
    condition = _get_condition_or_404(thesis_id, condition_id, db)
    result = test_now(condition)
    return templates.TemplateResponse(
        request, "theses/_condition_test_result.html", {"result": result}
    )
