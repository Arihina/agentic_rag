from __future__ import annotations

"""Platform ручки управления conversations."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import current_user
from app.clients.ingest import RagNotFound
from app.db import repository as repo
from app.db.repository import NotFoundOrForbidden
from app.db.session import get_session
from app.schemas.conversations import (
    ConversationCreateIn, ConversationListOut, ConversationOut,
    ConversationUpdateIn,
)
from app.services.rag_config import RagLookupFailed, validate_rag_exists
from app.state import state

router = APIRouter(
    prefix="/v1/platform/conversations", tags=["conversations"])


@router.post("", response_model=ConversationOut,
             status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreateIn,
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ConversationOut:
    """Создать новый чат."""
    try:
        await validate_rag_exists(body.rag_id, user_id, state.ingest)
    except RagNotFound:
        raise HTTPException(404, "Набор не найден")
    except RagLookupFailed as e:
        raise HTTPException(502, str(e))

    conv = await repo.create_conversation(
        session, user_id=user_id, rag_id=body.rag_id, title=body.title)
    await session.commit()
    return ConversationOut.model_validate(conv, from_attributes=True)


@router.get("", response_model=ConversationListOut)
async def list_conversations(
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ConversationListOut:
    """Список чатов пользователя"""
    convs = await repo.list_conversations(session, user_id)
    return ConversationListOut(
        data=[ConversationOut.model_validate(c, from_attributes=True)
              for c in convs])


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: uuid.UUID = Path(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ConversationOut:
    try:
        conv = await repo.get_conversation(
            session, conversation_id, user_id)
    except NotFoundOrForbidden:
        raise HTTPException(404, "Диалог не найден")
    return ConversationOut.model_validate(conv, from_attributes=True)


@router.patch("/{conversation_id}", response_model=ConversationOut)
async def update_conversation(
    body: ConversationUpdateIn,
    conversation_id: uuid.UUID = Path(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ConversationOut:
    """Переименование. rag_id менять нельзя — extra='forbid' в схеме
    ловит попытку послать `rag_id` в теле."""
    try:
        conv = await repo.update_conversation_title(
            session, conversation_id, user_id, body.title)
    except NotFoundOrForbidden:
        raise HTTPException(404, "Диалог не найден")
    await session.commit()
    return ConversationOut.model_validate(conv, from_attributes=True)


@router.delete("/{conversation_id}",
               status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID = Path(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """CASCADE удалит messages, sources, usage, feedback"""
    try:
        await repo.delete_conversation(
            session, conversation_id, user_id)
    except NotFoundOrForbidden:
        raise HTTPException(404, "Диалог не найден")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
