from __future__ import annotations

"""Клиент к ingestion /embed на внутреннем порту 8012.

Батч всех вариантов multi-query уходит одним вызовом — ingestion под это
и заточен (max_length=256, отдельный query-embedder с независимым
семафором). Разбивать батч на per-query вызовы нельзя: sparse-вектора
bge-m3 генерируются иначе, если модель работает батчем vs по-одному, и
результаты чуть разъезжаются — а нам нужна воспроизводимость.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EmbedItem:
    dense: list[float]
    sparse: dict[str, float]


class EmbedError(RuntimeError):
    """Ingestion недоступен или ответил ошибкой."""


class EmbedClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=settings.ingest_internal_url,
            timeout=settings.ingest_timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def ping(self) -> bool:
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            logger.warning("ingestion недоступен (health)", exc_info=True)
            return False

    async def embed(
        self,
        texts: list[str],
        pool: Literal["query", "ingest"] = "query",
    ) -> list[EmbedItem]:
        """Возвращает dense+sparse по каждому тексту в том же порядке."""
        if not texts:
            return []
        try:
            response = await self._client.post(
                "/embed", json={"texts": texts, "pool": pool})
        except httpx.TransportError as e:
            raise EmbedError(f"ingestion недоступен: {e}")

        if response.status_code != 200:
            raise EmbedError(
                f"ingestion /embed вернул {response.status_code}: "
                f"{response.text[:500]}")

        payload = response.json()
        return [EmbedItem(dense=e["dense"], sparse=e["sparse"])
                for e in payload["embeddings"]]
