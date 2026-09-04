from __future__ import annotations

"""Схемы Responses API."""

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResponsesRequest(BaseModel):
    """Тело POST /v1/responses."""
    model_config = ConfigDict(extra="forbid")

    model: str = Field(
        min_length=1,
        description="`rag/<uuid>` — идентификатор набора")
    input: str = Field(
        min_length=1,
        description="Текст запроса пользователя")
    conversation_id: uuid.UUID = Field(
        description="UUID диалога — обязателен: набор зафиксирован при "
                    "создании conversation, model должна совпасть с ним")
    stream: bool = Field(
        default=True,
        description="У нас всегда SSE; поле оставлено для контракт-"
                    "совместимости с OpenAI SDK. stream=false пока не "
                    "поддерживается — сервис в MVP не собирает ответ в "
                    "буфер и не отдаёт одним response'ом.")


class OutputTextItem(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class OutputMessage(BaseModel):
    type: Literal["message"] = "message"
    id: str = Field(
        description="msg_<uuid> — вложенный id, отличается от response.id "
                    "префиксом; на практике UI ссылается на response.id")
    role: Literal["assistant"] = "assistant"
    status: Literal["completed", "failed"] = "completed"
    content: list[OutputTextItem]


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseObject(BaseModel):
    """Тело GET /v1/responses/{id} и того, что отдаём в SSE completed"""
    id: str = Field(
        description="resp_<uuid> — префиксированный id ассистентского "
                    "сообщения")
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["completed", "failed", "in_progress"]
    model: str
    conversation_id: uuid.UUID
    output: list[OutputMessage] = Field(
        default_factory=list,
        description="Пустой список, если status=failed без частичного "
                    "ответа; в остальном ровно один message с одним "
                    "output_text")
    usage: UsageOut | None = Field(
        default=None,
        description="None для in_progress / failed без usage; заполняется "
                    "в completed")
    error: dict | None = Field(
        default=None,
        description="Заполнен при status=failed: {message, type, code}")
