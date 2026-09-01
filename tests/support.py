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
