from __future__ import annotations

"""HTTP-ручки Responses API."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Path, Request
from fastapi.responses import StreamingResponse

from app.api.ids import InvalidMessageId, format_message_id, parse_message_id
from app.auth import current_user
from app.config import settings
from app.db import repository as repo
from app.db.repository import NotFoundOrForbidden
from app.db.session import get_session
from app.schemas.responses import (
    OutputMessage, OutputTextItem, ResponseObject, ResponsesRequest, UsageOut,
)
from app.services.run_turn import run_turn
from app.sse.stream import with_heartbeat
from app.state import state
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/v1/responses", tags=["responses"])


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.post("")
async def create_response(
    request: Request,
    body: ResponsesRequest,
    user_id: uuid.UUID = Depends(current_user),
) -> StreamingResponse:
    """SSE-стрим одного хода"""

    async def _is_disconnected() -> bool:
        return await request.is_disconnected()

    source = run_turn(
        request_body=body,
        user_id=user_id,
        session_maker=state.session_maker,
        ingest=state.ingest,
        llm=state.llm,
        embed=state.embed,
        os_client=state.os_client,
        is_disconnected=_is_disconnected,
    )
    return StreamingResponse(
        with_heartbeat(source, interval=settings.sse_heartbeat_interval),
        media_type="text/event-stream; charset=utf-8",
        headers=_SSE_HEADERS,
    )


@router.get("/{response_id}", response_model=ResponseObject)
async def get_response(
    response_id: str = Path(...),
    user_id: uuid.UUID = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ResponseObject:
    try:
        message_id = parse_message_id(response_id)
    except InvalidMessageId as e:
        raise HTTPException(400, str(e))

    try:
        msg = await repo.get_message(session, message_id, user_id)
    except NotFoundOrForbidden:
        raise HTTPException(404, "Ответ не найден")

    await session.refresh(msg, attribute_names=["usage", "sources",
                                                "conversation"])

    return _to_response_object(msg)


def _to_response_object(msg) -> ResponseObject:
    api_status: str
    output: list[OutputMessage] = []
    error: dict | None = None

    if msg.status == "ok":
        api_status = "completed"
        output = [OutputMessage(
            id=f"msg_{msg.id}",
            status="completed",
            content=[OutputTextItem(text=msg.content or "")],
        )]
    elif msg.status == "failed":
        api_status = "failed"
        if msg.content:
            output = [OutputMessage(
                id=f"msg_{msg.id}",
                status="failed",
                content=[OutputTextItem(text=msg.content)],
            )]
        error = {
            "message": msg.error or "unknown error",
            "type": "server_error",
            "code": None,
        }
    else:
        api_status = "in_progress"

    usage = None
    if msg.usage is not None:
        usage = UsageOut(
            prompt_tokens=msg.usage.prompt_tokens,
            completion_tokens=msg.usage.completion_tokens,
            total_tokens=msg.usage.total_tokens,
        )

    model_str = f"rag/{msg.conversation.rag_id}"

    return ResponseObject(
        id=format_message_id(msg.id),
        created_at=int(msg.created_at.timestamp()),
        status=api_status,
        model=model_str,
        conversation_id=msg.conversation_id,
        output=output,
        usage=usage,
        error=error,
    )
