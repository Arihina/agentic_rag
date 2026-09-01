from __future__ import annotations

"""Аутентификация. current_user — точка расширения под Keycloak: при
переезде меняется тело функции, не сигнатуры ручек.

Пока — доверенный заголовок X-User-Id от мастера. Осознанный временный
компромисс: сервис доступен только на 127.0.0.1, единственный клиент —
мастер, у которого есть свой заголовок. Пробрасывается тем же именем и
в исходящих вызовах к ingestion (там пока такой же контракт).
"""

import uuid

from fastapi import Header, HTTPException


async def current_user(
    x_user_id: str | None = Header(default=None),
) -> uuid.UUID:
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Требуется X-User-Id")
    try:
        return uuid.UUID(x_user_id)
    except ValueError:
        raise HTTPException(
            status_code=401, detail="X-User-Id должен быть UUID")
