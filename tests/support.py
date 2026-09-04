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
    """Fake IngestClient для тестов сервисного слоя.

    По умолчанию get_rag возвращает валидный ready-набор с id, который
    попросили. Кастомизация — через set_rag() для конкретных id или
    set_error() для эмуляции сбоев ingestion. Реальный HTTP не идёт.
    """

    def __init__(self):
        # rag_id -> RagConfig-like dict, из которого get_rag соберёт объект.
        self._configs: dict[str, dict] = {}
        # rag_id -> исключение, которое надо бросить вместо возврата.
        self._errors: dict[str, Exception] = {}
        # Глобальный сбой (например, IngestError на все запросы).
        self._global_error: Exception | None = None
        self.get_rag_calls: list[tuple[str, str]] = []

    def set_rag(self, rag_id, *, name="Тестовый набор", status="ready",
                prompt=None, temperature=0.3, top_k=10,
                score_threshold=0.0):
        """Задать возврат get_rag для конкретного rag_id."""
        self._configs[str(rag_id)] = {
            "name": name, "status": status, "prompt": prompt,
            "temperature": temperature, "top_k": top_k,
            "score_threshold": score_threshold,
        }

    def set_error(self, rag_id, exc: Exception):
        """Задать исключение для конкретного rag_id."""
        self._errors[str(rag_id)] = exc

    def set_global_error(self, exc: Exception | None):
        self._global_error = exc

    async def close(self) -> None:
        pass

    async def get_rag(self, rag_id, user_id):
        from app.clients.ingest import RagConfig
        import uuid as _uuid

        self.get_rag_calls.append((str(rag_id), str(user_id)))

        if self._global_error is not None:
            raise self._global_error
        if str(rag_id) in self._errors:
            raise self._errors[str(rag_id)]

        cfg = self._configs.get(str(rag_id))
        if cfg is None:
            from app.clients.ingest import RagNotFound
            raise RagNotFound(str(rag_id))

        return RagConfig(
            id=_uuid.UUID(str(rag_id)),
            name=cfg["name"], status=cfg["status"],
            prompt=cfg["prompt"], temperature=cfg["temperature"],
            top_k=cfg["top_k"], score_threshold=cfg["score_threshold"],
        )


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
            # Эмулируем недоступность БД — execute падает.
            class _Broken(_FakeSession):
                async def execute(self, *args, **kwargs):
                    raise RuntimeError("fake БД недоступна")
            return _Broken()
        return _FakeSession()
