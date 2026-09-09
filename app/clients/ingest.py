from __future__ import annotations

"""Клиент к ingestion /v1/internal/* на внутреннем порту 8012.

Две ручки:
- `GET /v1/internal/rags/{id}` — резолв конфига набора (2.4.a).
- `POST /v1/internal/documents/lookup` — batch-подстановка filename по
  списку document_id.

Инвариант: user_id уходит query-параметром, НЕ заголовком — таков
контракт /v1/internal/*. Это отличается от платформенных ручек, где
`X-User-Id` — заголовок. Разница осознанная: платформенные вызывает
мастер и ставит user_id из своего auth, внутренние вызывают сервисы
платформы и явно указывают, от чьего имени спрашивают.
"""

import logging
import uuid
from dataclasses import dataclass

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


_LOOKUP_BATCH_MAX = 100


@dataclass(frozen=True, slots=True)
class RagConfig:
    id: uuid.UUID
    name: str
    status: str
    prompt: str | None
    temperature: float
    top_k: int
    score_threshold: float


class RagNotFound(LookupError):
    """Набор с таким id либо не существует, либо не принадлежит user_id.
    Ingestion не различает эти случаи (чтобы не утекала информация о чужих
    id), и мы тоже не различаем."""


class IngestError(RuntimeError):
    """Ingestion недоступен или ответил внезапной ошибкой (не 404)."""


class IngestClient:
    def __init__(self):
        self._client = httpx.AsyncClient(
            base_url=settings.ingest_internal_url,
            timeout=settings.ingest_timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def get_rag(self, rag_id: uuid.UUID,
                      user_id: uuid.UUID) -> RagConfig:
        try:
            response = await self._client.get(
                f"/v1/internal/rags/{rag_id}",
                params={"user_id": str(user_id)},
            )
        except httpx.TransportError as e:
            raise IngestError(f"ingestion недоступен: {e}")

        if response.status_code == 404:
            raise RagNotFound(str(rag_id))
        if response.status_code != 200:
            raise IngestError(
                f"ingestion /v1/internal/rags/{rag_id} вернул "
                f"{response.status_code}: {response.text[:500]}")

        payload = response.json()
        return RagConfig(
            id=uuid.UUID(payload["id"]),
            name=payload["name"],
            status=payload["status"],
            prompt=payload["prompt"],
            temperature=payload["temperature"],
            top_k=payload["top_k"],
            score_threshold=payload["score_threshold"],
        )

    async def lookup_documents(
        self,
        user_id: uuid.UUID,
        document_ids: list[uuid.UUID],
    ) -> dict[uuid.UUID, str]:
        """Batch-подстановка filename по массиву document_id."""
        if not document_ids:
            return {}

        seen: set[uuid.UUID] = set()
        unique_ids: list[uuid.UUID] = []
        for did in document_ids:
            if did not in seen:
                seen.add(did)
                unique_ids.append(did)

        if len(unique_ids) > _LOOKUP_BATCH_MAX:
            raise IngestError(
                f"lookup_documents: batch превышает лимит "
                f"({len(unique_ids)} > {_LOOKUP_BATCH_MAX}). "
                "Разбейте вызов на несколько.")

        try:
            response = await self._client.post(
                "/v1/internal/documents/lookup",
                params={"user_id": str(user_id)},
                json={"document_ids": [str(d) for d in unique_ids]},
            )
        except httpx.TransportError as e:
            raise IngestError(f"ingestion недоступен: {e}")

        if response.status_code != 200:
            raise IngestError(
                f"ingestion /v1/internal/documents/lookup вернул "
                f"{response.status_code}: {response.text[:500]}")

        payload = response.json()
        return {
            uuid.UUID(entry["document_id"]): entry["filename"]
            for entry in payload.get("documents", [])
        }
