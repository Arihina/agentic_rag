from __future__ import annotations

"""Заглушки клиентов для STUB-режима.

Идея: подменяем поля state.* тонкими фейками ДО первого HTTP-запроса к
TestClient, чтобы lifespan не тянул реальный ollama/opensearch/ingestion.
Подмена — через override в test setUp; lifespan сам ничего не создаст,
если поля уже проставлены (в теку lifespan этой проверки нет — добавим
если понадобится, а пока просто оставляем реальные клиенты чтобы они
падали ping, что и покажет тесту сети).
"""

from dataclasses import dataclass


@dataclass
class FakeOpenSearch:
    ping_ok: bool = True

    async def ping(self) -> bool:
        return self.ping_ok

    async def close(self) -> None:
        pass


@dataclass
class FakeLLM:
    ping_ok: bool = True

    async def ping(self) -> bool:
        return self.ping_ok

    async def close(self) -> None:
        pass


@dataclass
class FakeEmbed:
    ping_ok: bool = True

    async def ping(self) -> bool:
        return self.ping_ok

    async def close(self) -> None:
        pass


@dataclass
class FakeIngest:
    async def close(self) -> None:
        pass


class _FakeExecuteResult:
    def scalar_one(self):
        return 1

    def scalar_one_or_none(self):
        return None

    def scalars(self):
        return self

    def all(self):
        return []


class _FakeSession:
    """Минимальный совместимый с AsyncSession API — только то, что зовёт
    health-check (`session.execute(text("SELECT 1"))`) и тесты, которые
    хотят подменить БД без поднятого Postgres."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *args, **kwargs):
        return _FakeExecuteResult()

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def close(self):
        pass


class FakeSessionMaker:
    """Замена async_sessionmaker: возвращает _FakeSession как async CM.
    Тесты кладут в state.session_maker; SELECT 1 в health отвечает всегда OK,
    без поднятой БД."""
    ping_ok: bool = True

    def __init__(self, ping_ok: bool = True):
        self.ping_ok = ping_ok

    def __call__(self):
        if not self.ping_ok:
            class _Broken(_FakeSession):
                async def execute(self, *args, **kwargs):
                    raise RuntimeError("fake БД недоступна")
            return _Broken()
        return _FakeSession()
