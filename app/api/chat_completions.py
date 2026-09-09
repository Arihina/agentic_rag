from __future__ import annotations

"""Ручки под префиксом /v1/chat/completions/{id}/*."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.ids import InvalidMessageId, parse_message_id
from app.auth import current_user
from app.db import repository as repo
from app.db.repository import NotFoundOrForbidden
from app.db.session import get_session
from app.schemas.feedback import FeedbackIn, FeedbackOut
from app.schemas.sources import SourceOut, SourcesListOut

router = APIRouter(
    prefix="/v1/chat/completions", tags=["chat_completions"])


@router.post("/{message_id}/feedback",
             response_model=FeedbackOut,
             status_code=status.HTTP_200_OK)
async def upsert_feedback(
    body: FeedbackIn,
    message_id: str = Path(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> FeedbackOut:
    mid = _parse_id(message_id)
    try:
        fb = await repo.upsert_feedback(session, mid, user_id, body.root)
    except NotFoundOrForbidden:
        raise HTTPException(404, "Сообщение не найдено")
    await session.commit()
    return _to_out(fb)


@router.get("/{message_id}/feedback",
            response_model=FeedbackOut | None)
async def get_feedback(
    message_id: str = Path(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> FeedbackOut | None:
    mid = _parse_id(message_id)
    try:
        fb = await repo.get_feedback(session, mid, user_id)
    except NotFoundOrForbidden:
        raise HTTPException(404, "Сообщение не найдено")
    if fb is None:
        return None
    return _to_out(fb)


@router.delete("/{message_id}/feedback",
               status_code=status.HTTP_204_NO_CONTENT)
async def delete_feedback(
    message_id: str = Path(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> Response:
    mid = _parse_id(message_id)
    try:
        await repo.delete_feedback(session, mid, user_id)
    except NotFoundOrForbidden:
        raise HTTPException(404, "Сообщение не найдено")
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{message_id}/sources", response_model=SourcesListOut)
async def list_sources(
    message_id: str = Path(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> SourcesListOut:
    mid = _parse_id(message_id)
    try:
        rows = await repo.list_sources(session, mid, user_id)
    except NotFoundOrForbidden:
        raise HTTPException(404, "Сообщение не найдено")

    return SourcesListOut(
        data=[
            SourceOut(
                order=r.order,
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                chunk_index=r.chunk_index,
                filename=r.filename,
            )
            for r in rows
        ]
    )


def _parse_id(raw: str) -> uuid.UUID:
    try:
        return parse_message_id(raw)
    except InvalidMessageId as e:
        raise HTTPException(400, str(e))


def _to_out(fb) -> FeedbackOut:
    return FeedbackOut(
        message_id=fb.message_id,
        data=fb.data,
        created_at=fb.created_at,
        updated_at=fb.updated_at,
    )
