from __future__ import annotations

"""Схемы Platform-ручек conversations/messages."""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreateIn(BaseModel):
    """Тело POST /v1/platform/conversations."""
    model_config = ConfigDict(extra="forbid")

    rag_id: uuid.UUID = Field(
        description="UUID набора; должен принадлежать текущему user'у")
    title: str | None = Field(
        default=None, max_length=500,
        description="Название чата. Опционально; при None UI подставляет "
                    "плейсхолдер")


class ConversationUpdateIn(BaseModel):
    """Тело PATCH /v1/platform/conversations/{id}."""
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=500)


class ConversationOut(BaseModel):
    """Метаданные одного чата."""
    id: uuid.UUID
    rag_id: uuid.UUID
    title: str | None
    created_at: datetime
    updated_at: datetime


class ConversationListOut(BaseModel):
    """Обёртка для GET /v1/platform/conversations. data: [...]"""
    data: list[ConversationOut]


class MessageOut(BaseModel):
    """Одно сообщение в истории чата."""
    id: str = Field(description="msg_<uuid> — префиксированный id")
    object: Literal["message"] = "message"
    role: Literal["user", "assistant"]
    content: str
    status: Literal["ok", "pending", "failed"]
    error: str | None = None
    created_at: datetime


class MessagesListOut(BaseModel):
    """Обёртка для GET /v1/platform/conversations/{id}/messages."""
    data: list[MessageOut]
    has_more: bool = False
