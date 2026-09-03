from __future__ import annotations

"""Типизированный репозиторий поверх моделей."""

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    Conversation, Message, MessageFeedback, MessageSource, MessageUsage,
)


class NotFoundOrForbidden(LookupError):
    """Ресурса нет — или он есть, но принадлежит другому пользователю.
    Единая ошибка: различать нельзя, иначе через 404/403 утекает
    информация о существовании чужих id."""


async def create_conversation(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    rag_id: uuid.UUID,
    title: str | None = None,
) -> Conversation:
    conv = Conversation(user_id=user_id, rag_id=rag_id, title=title)
    session.add(conv)
    await session.flush()
    return conv


async def get_conversation(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Conversation:
    conv = await session.get(Conversation, conversation_id)
    if conv is None or conv.user_id != user_id:
        raise NotFoundOrForbidden(str(conversation_id))
    return conv


async def list_conversations(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    limit: int = 100,
) -> list[Conversation]:
    stmt = (select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit))
    return list((await session.execute(stmt)).scalars().all())


async def update_conversation_title(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    title: str,
) -> Conversation:
    conv = await get_conversation(session, conversation_id, user_id)
    conv.title = title
    conv.updated_at = _utcnow()
    await session.flush()
    return conv


async def touch_conversation(
    session: AsyncSession, conversation_id: uuid.UUID,
) -> None:
    stmt = (update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=_utcnow()))
    await session.execute(stmt)


async def delete_conversation(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
) -> None:
    conv = await get_conversation(session, conversation_id, user_id)
    await session.delete(conv)
    await session.flush()


async def add_user_message(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    content: str,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role="user", content=content, status="ok")
    session.add(msg)
    await touch_conversation(session, conversation_id)
    await session.flush()
    return msg


async def add_pending_assistant_message(
    session: AsyncSession,
    conversation_id: uuid.UUID,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role="assistant", content="", status="pending")
    session.add(msg)
    await touch_conversation(session, conversation_id)
    await session.flush()
    return msg


async def mark_message_ok(
    session: AsyncSession,
    message_id: uuid.UUID,
    content: str,
) -> None:
    stmt = (update(Message)
            .where(Message.id == message_id)
            .values(content=content, status="ok", error=None))
    await session.execute(stmt)


async def mark_message_failed(
    session: AsyncSession,
    message_id: uuid.UUID,
    error: str,
    content: str = "",
) -> None:
    stmt = (update(Message)
            .where(Message.id == message_id)
            .values(content=content, status="failed", error=error))
    await session.execute(stmt)


async def get_message(
    session: AsyncSession,
    message_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Message:
    stmt = (select(Message).join(Conversation)
            .where(Message.id == message_id,
                   Conversation.user_id == user_id))
    msg = (await session.execute(stmt)).scalar_one_or_none()
    if msg is None:
        raise NotFoundOrForbidden(str(message_id))
    return msg


async def list_messages(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
) -> list[Message]:
    await get_conversation(session, conversation_id, user_id)
    stmt = (select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at))
    return list((await session.execute(stmt)).scalars().all())


@dataclass(frozen=True, slots=True)
class SourceIn:
    chunk_id: str
    document_id: uuid.UUID
    chunk_index: int
    filename: str
    rag_id: uuid.UUID
    order: int


async def add_sources(
    session: AsyncSession,
    message_id: uuid.UUID,
    sources: list[SourceIn],
) -> None:
    if not sources:
        return
    session.add_all([
        MessageSource(
            message_id=message_id,
            chunk_id=s.chunk_id,
            document_id=s.document_id,
            chunk_index=s.chunk_index,
            filename=s.filename,
            rag_id=s.rag_id,
            order=s.order,
        ) for s in sources
    ])
    await session.flush()


async def list_sources(
    session: AsyncSession,
    message_id: uuid.UUID,
    user_id: uuid.UUID,
) -> list[MessageSource]:
    await get_message(session, message_id, user_id)
    stmt = (select(MessageSource)
            .where(MessageSource.message_id == message_id)
            .order_by(MessageSource.order))
    return list((await session.execute(stmt)).scalars().all())


async def set_usage(
    session: AsyncSession,
    message_id: uuid.UUID,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> None:
    total = prompt_tokens + completion_tokens
    dialect = session.bind.dialect.name if session.bind else ""

    if dialect == "postgresql":
        stmt = pg_insert(MessageUsage).values(
            message_id=message_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            model=model,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[MessageUsage.message_id],
            set_={
                "prompt_tokens": stmt.excluded.prompt_tokens,
                "completion_tokens": stmt.excluded.completion_tokens,
                "total_tokens": stmt.excluded.total_tokens,
                "model": stmt.excluded.model,
            },
        )
        await session.execute(stmt)
    else:
        existing = (await session.execute(
            select(MessageUsage).where(MessageUsage.message_id == message_id)
        )).scalar_one_or_none()
        if existing is None:
            session.add(MessageUsage(
                message_id=message_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total,
                model=model,
            ))
        else:
            existing.prompt_tokens = prompt_tokens
            existing.completion_tokens = completion_tokens
            existing.total_tokens = total
            existing.model = model
        await session.flush()


async def upsert_feedback(
    session: AsyncSession,
    message_id: uuid.UUID,
    user_id: uuid.UUID,
    patch: dict,
) -> MessageFeedback:
    await get_message(session, message_id, user_id)

    dialect = session.bind.dialect.name if session.bind else ""

    if dialect == "postgresql":
        stmt = pg_insert(MessageFeedback).values(
            message_id=message_id, data=patch)

        stmt = stmt.on_conflict_do_update(
            index_elements=[MessageFeedback.message_id],
            set_={
                "data": MessageFeedback.data.op("||")(stmt.excluded.data),
                "updated_at": func.now(),
            },
        )
        await session.execute(stmt)
    else:
        existing = (await session.execute(
            select(MessageFeedback)
            .where(MessageFeedback.message_id == message_id)
        )).scalar_one_or_none()
        if existing is None:
            session.add(MessageFeedback(message_id=message_id, data=patch))
        else:
            existing.data = {**(existing.data or {}), **patch}
            existing.updated_at = _utcnow()
        await session.flush()

    fresh = (await session.execute(
        select(MessageFeedback)
        .where(MessageFeedback.message_id == message_id)
    )).scalar_one()
    return fresh


async def get_feedback(
    session: AsyncSession,
    message_id: uuid.UUID,
    user_id: uuid.UUID,
) -> MessageFeedback | None:
    await get_message(session, message_id, user_id)
    stmt = (select(MessageFeedback)
            .where(MessageFeedback.message_id == message_id))
    return (await session.execute(stmt)).scalar_one_or_none()


async def delete_feedback(
    session: AsyncSession,
    message_id: uuid.UUID,
    user_id: uuid.UUID,
) -> bool:
    await get_message(session, message_id, user_id)
    stmt = (delete(MessageFeedback)
            .where(MessageFeedback.message_id == message_id))
    result = await session.execute(stmt)
    return (result.rowcount or 0) > 0


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)
