from __future__ import annotations

"""Резолв конфига набора для одного хода агента."""

import re
import uuid
from dataclasses import dataclass

from app.clients.ingest import IngestClient, IngestError, RagNotFound
from app.config import settings


class InvalidModelForm(ValueError):
    """`model` не в формате `rag/<uuid>`."""


class ModelDoesNotMatchConversation(ValueError):
    """rag_id из `model` не совпадает с rag_id, зафиксированным при
    создании conversation."""


class RagUnavailable(RuntimeError):
    """Набор существует, но не готов принимать вопросы: `empty` (нет
    документов), `ingesting` (загрузка ещё идёт, ready=0), `failed`
    (все документы упали при обработке)."""

    def __init__(self, rag_id: uuid.UUID, status: str):
        super().__init__(
            f"Набор {rag_id} в статусе {status!r}, отвечать пока нельзя")
        self.rag_id = rag_id
        self.status = status


class RagLookupFailed(RuntimeError):
    """Ingestion не ответил (сетевой сбой, 5xx)"""


@dataclass(frozen=True, slots=True)
class ResolvedRag:
    rag_id: uuid.UUID
    name: str
    top_k: int
    score_threshold: float
    answer_temperature: float
    answer_system_prompt: str


_MODEL_RE = re.compile(
    r"^rag/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$",
    re.IGNORECASE,
)


def parse_model(model: str) -> uuid.UUID:
    """`"rag/<uuid>"` → UUID. Иначе InvalidModelForm."""
    if not isinstance(model, str) or not model:
        raise InvalidModelForm(
            "поле model обязательно и должно быть непустой строкой")
    m = _MODEL_RE.match(model)
    if m is None:
        raise InvalidModelForm(
            f"model={model!r} — ожидается форма 'rag/<uuid>'")
    return uuid.UUID(m.group(1))


async def resolve_rag_for_turn(
    model: str,
    user_id: uuid.UUID,
    ingest: IngestClient,
    *,
    conversation_rag_id: uuid.UUID | None = None,
) -> ResolvedRag:
    """Полный резолв одним вызовом:
    1) распарсить `model`,
    2) сверить с зафиксированным в conversation (если она есть),
    3) спросить конфиг у ingestion,
    4) проверить, что status='ready',
    5) склеить system-prompt.
    """
    rag_id = parse_model(model)

    if conversation_rag_id is not None and rag_id != conversation_rag_id:
        raise ModelDoesNotMatchConversation(
            f"model требует набор {rag_id}, а диалог привязан "
            f"к {conversation_rag_id} — набор нельзя сменить в середине "
            "диалога, начните новый чат")

    try:
        cfg = await ingest.get_rag(rag_id, user_id)
    except RagNotFound:
        raise
    except IngestError as e:
        raise RagLookupFailed(
            f"Не удалось получить конфиг набора {rag_id}: {e}")

    if cfg.status != "ready":
        raise RagUnavailable(cfg.id, cfg.status)

    return ResolvedRag(
        rag_id=cfg.id,
        name=cfg.name,
        top_k=cfg.top_k,
        score_threshold=cfg.score_threshold,
        answer_temperature=cfg.temperature,
        answer_system_prompt=compose_answer_system_prompt(cfg.prompt),
    )


def compose_answer_system_prompt(rag_prompt: str | None) -> str:
    """Склейка: дефолтный ANSWER_SYSTEM_PROMPT + опциональный rag_set.prompt
    отдельным блоком.
    """
    base = settings.answer_system_prompt
    if rag_prompt is None or not rag_prompt.strip():
        return base
    extra = rag_prompt.strip()
    return f"{base}\n\nДополнительные инструкции для этого набора:\n{extra}"
