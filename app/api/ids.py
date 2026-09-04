from __future__ import annotations

"""Парсер id сообщений во всех формах, которые встречаются в контракте.

Клиенты присылают message_id в трёх видах — зависит от того, из какого
события они его сохранили или из какого другого агента платформы перешли:

- `resp_<uuid>` — формат Responses API;
- `chatcmpl-<uuid>` — OpenAI Chat Completions формат, используют другие
  агенты платформы (epoz, tech_rag, document_chat, slave_chat) и все
  клиенты, работающие через OpenAI SDK на chat.completions;
- голый `<uuid>` — некоторые клиенты сохраняют id без префикса
  (например, из тела ответа `id` уже пришёл с префиксом, а они
  сохраняют distilled форму).

Всегда возвращается `resp_<uuid>`
"""

import re
import uuid


class InvalidMessageId(ValueError):
    """Строка не парсится ни в одну из трёх форм."""


_UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_MESSAGE_ID_RE = re.compile(
    rf"^(?:resp_|chatcmpl-)?({_UUID_RE})$", re.IGNORECASE)


def parse_message_id(raw: str) -> uuid.UUID:
    """`resp_<uuid>` / `chatcmpl-<uuid>` / `<uuid>` → UUID. Регистр UUID
    не важен, префикса — тоже (принимаем `Resp_`, `RESP_`, ...)."""
    if not isinstance(raw, str) or not raw:
        raise InvalidMessageId(
            "id обязателен и должен быть непустой строкой")
    m = _MESSAGE_ID_RE.match(raw)
    if m is None:
        raise InvalidMessageId(
            f"id={raw!r} — ожидается 'resp_<uuid>', "
            f"'chatcmpl-<uuid>' или голый UUID")
    return uuid.UUID(m.group(1))


def format_message_id(message_id: uuid.UUID) -> str:
    """Единый формат наружу — `resp_<uuid>`. Использовать во всех
    JSON-ответах и SSE-frame'ах."""
    return f"resp_{message_id}"
