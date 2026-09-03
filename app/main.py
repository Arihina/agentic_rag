from __future__ import annotations

"""Agentic RAG — сервис на одном порту (8020)."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app import errors
from app.api import health
from app.clients.embed import EmbedClient
from app.clients.ingest import IngestClient
from app.clients.llm import LLMClient
from app.clients.opensearch import make_opensearch_client
from app.config import settings
from app.db.session import dispose_db, make_engine, make_session_maker
from app.state import state

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Инициализация клиентов...")
    state.os_client = make_opensearch_client()
    state.llm = LLMClient()
    state.embed = EmbedClient()
    state.ingest = IngestClient()
    state.db_engine = make_engine()
    state.session_maker = make_session_maker(state.db_engine)

    state.ready.set()
    logger.info("Готов, порт %s", settings.port)
    yield

    logger.info("Останов, закрываю клиенты...")
    await asyncio.gather(
        state.os_client.close(),
        state.llm.close(),
        state.embed.close(),
        state.ingest.close(),
        dispose_db(state.db_engine),
        return_exceptions=True,
    )


app = FastAPI(title="Agentic RAG", version="2.0.0", lifespan=lifespan)
app.include_router(health.router)
errors.install(app)


def run() -> None:
    """Запуск сервиса: python -m app.main"""
    import uvicorn

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
    )

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
        timeout_keep_alive=settings.timeout_keep_alive,
    )


if __name__ == "__main__":
    run()
