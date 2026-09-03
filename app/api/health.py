from __future__ import annotations

"""GET /health — состояние сервиса и его внешних зависимостей."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.state import state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["service"])


class HealthResponse(BaseModel):
    status: str
    ready: bool
    opensearch: bool
    ollama: bool
    ingestion: bool
    database: bool


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    if not state.ready.is_set():
        return HealthResponse(
            status="starting", ready=False,
            opensearch=False, ollama=False, ingestion=False, database=False)

    try:
        opensearch_ok = await state.os_client.ping()
    except Exception:
        logger.warning("opensearch недоступен", exc_info=True)
        opensearch_ok = False

    ollama_ok = await state.llm.ping()
    ingestion_ok = await state.embed.ping()

    database_ok = False
    try:
        async with state.session_maker() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        logger.warning("postgres недоступен", exc_info=True)

    checks = (opensearch_ok, ollama_ok, ingestion_ok, database_ok)
    return HealthResponse(
        status="ok" if all(checks) else "degraded",
        ready=True,
        opensearch=opensearch_ok,
        ollama=ollama_ok,
        ingestion=ingestion_ok,
        database=database_ok,
    )


@router.get("/health/ready")
async def ready() -> JSONResponse:
    """Kubernetes readiness: 503 до окончания lifespan-подготовки, 200 после.
    Отдельная ручка, потому что 200 на /health не должен создавать иллюзии
    «сервис готов принимать трафик» — там degraded это тоже 200."""
    if state.ready.is_set():
        return JSONResponse({"ready": True, "port": settings.port})
    return JSONResponse({"ready": False}, status_code=503)
