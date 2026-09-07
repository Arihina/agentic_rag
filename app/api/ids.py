from __future__ import annotations

"""Парсер id сообщений во всех формах, которые встречаются в контракте.

Клиенты присылают message_id в четырёх видах — зависит от того, из
какого события они его сохранили или через какой endpoint читали:

- `resp_<uuid>` — Responses API (POST /v1/responses, GET /v1/responses/{id});
- `msg_<uuid>` — platform listings (GET /v1/platform/conversations/{id}/messages);
- `chatcmpl-<uuid>` — OpenAI Chat Completions формат, используют другие
  агенты платформы и все клиенты, работающие через OpenAI SDK на
  chat.completions;
- голый `<uuid>` — некоторые клиенты сохраняют id без префикса.

Наружу формат зависит от endpoint'а:
- Responses API → `resp_<uuid>` (`format_message_id`);
- Platform listings → `msg_<uuid>` (`format_platform_message_id`).
"""

import re
import uuid


class InvalidMessageId(ValueError):
    """Строка не парсится ни в одну из известных форм."""


_UUID_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_MESSAGE_ID_RE = re.compile(
    rf"^(?:resp_|msg_|chatcmpl-)?({_UUID_RE})$", re.IGNORECASE)


def parse_message_id(raw: str) -> uuid.UUID:
    """`resp_<uuid>` / `msg_<uuid>` / `chatcmpl-<uuid>` / `<uuid>` → UUID."""
    if not isinstance(raw, str) or not raw:
        raise InvalidMessageId(
            "id обязателен и должен быть непустой строкой")
    m = _MESSAGE_ID_RE.match(raw)
    if m is None:
        raise InvalidMessageId(
            f"id={raw!r} — ожидается 'resp_<uuid>', 'msg_<uuid>', "
            f"'chatcmpl-<uuid>' или голый UUID")
    return uuid.UUID(m.group(1))


def format_message_id(message_id: uuid.UUID) -> str:
    """Формат для Responses API — `resp_<uuid>`. Для
    JSON-ответов и SSE-frame'ов POST /v1/responses и GET
    /v1/responses/{id}."""
    return f"resp_{message_id}"


def format_platform_message_id(message_id: uuid.UUID) -> str:
    """Формат для platform listings — `msg_<uuid>`. Для
    GET /v1/platform/conversations/{id}/messages."""
    return f"msg_{message_id}"
