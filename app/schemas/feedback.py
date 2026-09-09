from __future__ import annotations

"""Схемы feedback endpoint'ов."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, RootModel


class FeedbackIn(RootModel[dict[str, Any]]):
    root: dict[str, Any] = Field(min_length=1)


class FeedbackOut(BaseModel):
    """Возврат при POST и GET (когда feedback есть). GET без feedback
    возвращает `null` — не эту схему."""
    message_id: uuid.UUID
    data: dict[str, Any]
    created_at: datetime
    updated_at: datetime
