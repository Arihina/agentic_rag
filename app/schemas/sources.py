from __future__ import annotations

"""Схемы sources endpoint'а."""

import uuid

from pydantic import BaseModel


class SourceOut(BaseModel):
    """Один чанк-источник ответа. Порядок в SourcesListOut — по `order`,
    он же соответствует нумерации [1], [2]... в тексте answer'а. Клиент
    подсвечивает `[N]` в тексте и подтягивает n-й элемент этого списка."""

    order: int
    chunk_id: str
    document_id: uuid.UUID
    chunk_index: int
    filename: str


class SourcesListOut(BaseModel):
    """Обёртка {data: [...]}."""
    data: list[SourceOut]
