from __future__ import annotations

"""Разделяемое состояние процесса, собирается в lifespan.

Все клиенты создаются одним экземпляром на процесс и переиспользуются во
всех обработчиках. Прямой доступ через `state.<field>` — вместо DI-инъекций
на каждую ручку; в тестах поля перекрываются точечно.
"""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opensearchpy import AsyncOpenSearch
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

    from app.clients.embed import EmbedClient
    from app.clients.ingest import IngestClient
    from app.clients.llm import LLMClient


class AppState:
    os_client: "AsyncOpenSearch"
    llm: "LLMClient"
    embed: "EmbedClient"
    ingest: "IngestClient"
    db_engine: "AsyncEngine"
    session_maker: "async_sessionmaker[AsyncSession]"
    ready: asyncio.Event


state = AppState()
state.ready = asyncio.Event()
