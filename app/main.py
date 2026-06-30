import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.debug import router as debug_router
from app.api.theses import router as theses_router
from app.core.database import get_db, get_session_factory
from app.workflows.runner import register_workflows

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = get_session_factory()()
    try:
        register_workflows(db)
    except Exception:
        logger.exception("Failed to register workflows at startup")
    finally:
        db.close()

    yield


app = FastAPI(title="Macro Agents", version="0.1.0", lifespan=lifespan)
app.include_router(debug_router)
app.include_router(theses_router)


@app.get("/health")
def health(db: Annotated[Session, Depends(get_db)]) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    return {"status": "ok"}
