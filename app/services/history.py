from __future__ import annotations

"""Скользящее окно истории для rewriter"""

import logging
import uuid
from functools import lru_cache
from typing import Literal, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Message
from app.db.repository import get_conversation

logger = logging.getLogger(__name__)

_MESSAGE_OVERHEAD_TOKENS = 5

_DROP_TARGET_RATIO = 0.65

Overflow = Literal["truncate", "strict"]


class HistoryOverflowError(RuntimeError):
    ...


class TokenizerProto(Protocol):

    def encode(self, text: str) -> object:  # returns sequence-like
        ...


@lru_cache(maxsize=4)
def get_tokenizer(repo: str) -> TokenizerProto:
    """Ленивая загрузка. Первое обращение — тянет с HuggingFace hub
    (или из локального кеша HF_HOME). Кеш на процесс — токенайзер
    thread-safe и переиспользуется всеми запросами."""
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained(repo)


def count_tokens(text: str, tokenizer: TokenizerProto) -> int:
    encoded = tokenizer.encode(text)

    ids = getattr(encoded, "ids", encoded)
    return len(ids)


def _message_tokens(msg: dict, tokenizer: TokenizerProto) -> int:
    return count_tokens(msg["content"], tokenizer) + _MESSAGE_OVERHEAD_TOKENS


async def build_history_for_rewriter(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    tokenizer: TokenizerProto | None = None,
    token_limit: int | None = None,
    overflow: Overflow | None = None,
) -> list[dict[str, str]]:
    tokenizer = tokenizer or get_tokenizer(settings.tokenizer_repo)
    token_limit = token_limit or settings.history_token_limit

    overflow = overflow or settings.history_overflow
    if overflow not in ("truncate", "strict"):
        raise ValueError(
            f"history_overflow должен быть 'truncate' или 'strict', "
            f"получено: {overflow!r}")

    await get_conversation(session, conversation_id, user_id)

    stmt = (select(Message)
            .where(Message.conversation_id == conversation_id,
                   Message.status == "ok")
            .order_by(Message.created_at))
    messages = list((await session.execute(stmt)).scalars().all())

    history = [{"role": m.role, "content": m.content} for m in messages]
    return _apply_sliding_window(history, tokenizer, token_limit, overflow)


def _apply_sliding_window(
    history: list[dict[str, str]],
    tokenizer: TokenizerProto,
    token_limit: int,
    overflow: Overflow,
) -> list[dict[str, str]]:
    if not history:
        return []

    working = list(history)
    per_message = [_message_tokens(m, tokenizer) for m in working]
    total = sum(per_message)

    if total <= token_limit:
        return _strip_leading_assistant(working)

    if overflow == "strict":
        raise HistoryOverflowError(
            f"история {total} токенов > лимит {token_limit}; включите "
            f"overflow=truncate или начните новый чат")

    target = int(token_limit * _DROP_TARGET_RATIO)
    dropped_count = 0
    while total > target and len(working) > 1:
        total -= per_message.pop(0)
        working.pop(0)
        dropped_count += 1

    logger.info(
        "sliding_window: дропнули %d старых сообщений, осталось %d "
        "(≈%d токенов, лимит %d, target %d)",
        dropped_count, len(working), total, token_limit, target)

    return _strip_leading_assistant(working)


def _strip_leading_assistant(
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    while history and history[0]["role"] == "assistant":
        history.pop(0)
    return history
