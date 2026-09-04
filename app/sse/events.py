from __future__ import annotations

"""Конструкторы SSE-событий Responses API."""

import json
import time
import uuid


HEARTBEAT: bytes = b": ping\n\n"

DONE: bytes = b"data: [DONE]\n\n"


def _sse(event: str, data: dict) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


def response_created(
    *, response_id: uuid.UUID, model: str, conversation_id: uuid.UUID,
) -> bytes:
    return _sse("response.created", {
        "id": _to_resp_id(response_id),
        "object": "response",
        "status": "in_progress",
        "model": model,
        "created_at": int(time.time()),
        "conversation_id": str(conversation_id),
    })


def response_in_progress(*, response_id: uuid.UUID) -> bytes:
    return _sse("response.in_progress", {
        "id": _to_resp_id(response_id),
        "status": "in_progress",
    })


def output_text_delta(*, response_id: uuid.UUID, delta: str) -> bytes:
    return _sse("response.output_text.delta", {
        "id": _to_resp_id(response_id),
        "delta": delta,
    })


def response_completed(
    *,
    response_id: uuid.UUID,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> bytes:
    return _sse("response.completed", {
        "id": _to_resp_id(response_id),
        "status": "completed",
        "model": model,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


def response_error(
    *,
    response_id: uuid.UUID | None,
    message: str,
    error_type: str = "server_error",
    code: str | None = None,
) -> bytes:
    body: dict = {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
        },
    }
    if response_id is not None:
        body["id"] = _to_resp_id(response_id)
    return _sse("response.error", body)


def _to_resp_id(message_id: uuid.UUID) -> str:
    return f"resp_{message_id}"
