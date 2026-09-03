from __future__ import annotations

"""Async engine и фабрика сессий."""

from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine,
)

from app.config import settings
from app.db.models import Base


def make_engine(url: str | None = None) -> AsyncEngine:
    url = url or settings.database_url
    kwargs: dict = {"echo": False, "future": True}
    if not url.startswith("sqlite"):
        kwargs["pool_size"] = settings.db_pool_size
        kwargs["max_overflow"] = settings.db_max_overflow
    return create_async_engine(url, **kwargs)


def make_session_maker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db(engine: AsyncEngine) -> None:
    await engine.dispose()


async def get_session() -> AsyncIterator[AsyncSession]:
    from app.state import state
    async with state.session_maker() as session:
        yield session
