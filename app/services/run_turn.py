from __future__ import annotations

"""Оркестратор одного хода Responses API.

Async-генератор SSE-байтов. Собирает воедино:
- валидацию (parse_model, get_conversation);
- резолв конфига набора (services/rag_config);
- persistence-часть (create user + pending assistant, mark ok/failed);
- сборку истории (services/history);
- агентский цикл (core/agent) с cancel_hook;
- финализацию (add_sources + set_usage) одной транзакцией.

Транзакции короткие. Ход агента (LLM + OpenSearch + ingestion) может
идти 30+ секунд — держать открытой транзакцию всё это время значит
удерживать connection из пула. При 10 конкурентных ходах пул исчерпан.
Значит:
  - session_1: create user + pending assistant + commit → генерим
    response.created;
  - (без сессии): агентский цикл;
  - session_2: mark_ok + sources + usage + commit → генерим completed.

cancel hook:
- **Ранний обрыв** (клиент отвалился ДО того, как мы залипли на yield):
  cancel_hook внутри run_agent видит is_disconnected=True → бросает
  ClientDisconnected → мы ловим здесь → mark_failed → return без yield.
- **Поздний обрыв** (клиент отвалился, пока мы на yield): with_heartbeat
  вызывает aclose() → GeneratorExit внутри нашего yield. try/finally
  запускает cleanup через asyncio.shield — best-effort mark_failed без
  await'а (иначе получим RuntimeError на closing generator).
"""

import asyncio
import logging
import uuid
from typing import Awaitable, Callable, AsyncIterator

from opensearchpy import AsyncOpenSearch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.clients.embed import EmbedClient
from app.clients.ingest import IngestClient, RagNotFound
from app.clients.llm import LLMClient
from app.core.agent import ClientDisconnected, run_agent
from app.db import repository as repo
from app.db.repository import NotFoundOrForbidden, SourceIn
from app.schemas.responses import ResponsesRequest
from app.services.history import build_history_for_rewriter, count_tokens, get_tokenizer
from app.services.rag_config import (
    InvalidModelForm, ModelDoesNotMatchConversation, RagLookupFailed,
    RagUnavailable, resolve_rag_for_turn,
)
from app.sse import events
from app.config import settings

logger = logging.getLogger(__name__)


async def run_turn(
    *,
    request_body: ResponsesRequest,
    user_id: uuid.UUID,
    session_maker: async_sessionmaker[AsyncSession],
    ingest: IngestClient,
    llm: LLMClient,
    embed: EmbedClient,
    os_client: AsyncOpenSearch,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[bytes]:
    """SSE-стрим одного хода. Ответственность: полный жизненный цикл
    ассистентского сообщения — от валидации до сохранения ответа или
    отметки failed при любом сбое."""

    try:
        conversation = await _load_conversation(
            session_maker, request_body.conversation_id, user_id)
    except NotFoundOrForbidden:
        yield events.response_error(
            response_id=None,
            message="Диалог не найден",
            error_type="not_found_error")
        return

    try:
        resolved = await resolve_rag_for_turn(
            request_body.model, user_id, ingest,
            conversation_rag_id=conversation.rag_id)
    except InvalidModelForm as e:
        yield events.response_error(
            response_id=None, message=str(e),
            error_type="invalid_request_error")
        return
    except ModelDoesNotMatchConversation as e:
        yield events.response_error(
            response_id=None, message=str(e),
            error_type="invalid_request_error")
        return
    except RagNotFound:
        yield events.response_error(
            response_id=None, message="Набор не найден",
            error_type="not_found_error")
        return
    except RagUnavailable as e:
        yield events.response_error(
            response_id=None, message=str(e),
            error_type="invalid_request_error")
        return
    except RagLookupFailed as e:
        yield events.response_error(
            response_id=None, message=str(e),
            error_type="bad_gateway_error")
        return

    async with session_maker() as session:
        await repo.add_user_message(
            session, conversation.id, request_body.input)
        assistant_msg = await repo.add_pending_assistant_message(
            session, conversation.id)
        await session.commit()
        msg_id = assistant_msg.id

    yield events.response_created(
        response_id=msg_id, model=request_body.model,
        conversation_id=conversation.id)
    yield events.response_in_progress(response_id=msg_id)

    async with session_maker() as session:
        history = await build_history_for_rewriter(
            session, conversation.id, user_id)
        if history and history[-1]["content"] == request_body.input:
            history = history[:-1]

    finalized = False
    try:
        trace = await run_agent(
            str(resolved.rag_id), os_client, llm, embed,
            request_body.input, history,
            top_k=resolved.top_k,
            score_threshold=resolved.score_threshold,
            answer_system_prompt=resolved.answer_system_prompt,
            answer_temperature=resolved.answer_temperature,
            cancel_hook=is_disconnected,
        )

        answer_text = trace.answer.answer if trace.answer else ""
        prompt_tokens, completion_tokens = _estimate_usage(
            history, request_body.input, resolved.answer_system_prompt,
            trace.final_chunks, answer_text)

        async with session_maker() as session:
            await repo.mark_message_ok(session, msg_id, answer_text)
            await repo.add_sources(session, msg_id, [
                SourceIn(
                    chunk_id=hit["_id"],
                    document_id=uuid.UUID(hit["_source"]["document_id"]),
                    chunk_index=int(hit["_source"]["chunk_index"]),
                    filename=hit["_source"].get("filename", "(документ)"),
                    rag_id=resolved.rag_id,
                    order=i + 1,
                )
                for i, hit in enumerate(trace.final_chunks)
            ])
            await repo.set_usage(
                session, msg_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=settings.llm_model_answer)
            await session.commit()
        finalized = True

        yield events.output_text_delta(
            response_id=msg_id, delta=answer_text)
        yield events.response_completed(
            response_id=msg_id, model=request_body.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens)

    except ClientDisconnected:
        await _mark_failed_bg(session_maker, msg_id, "client_disconnected")
        finalized = True

    except Exception as exc:
        logger.exception("run_turn: агентский цикл упал")
        await _mark_failed_bg(
            session_maker, msg_id,
            f"{type(exc).__name__}: {exc}")
        finalized = True
        yield events.response_error(
            response_id=msg_id,
            message=f"{type(exc).__name__}: {exc}",
            error_type="server_error")

    finally:
        if not finalized:
            try:
                await asyncio.shield(_mark_failed_bg(
                    session_maker, msg_id, "client_disconnected"))
            except (asyncio.CancelledError, RuntimeError):
                logger.warning(
                    "run_turn: не смогли записать client_disconnected "
                    "для msg %s — event loop закрывается", msg_id)


async def _load_conversation(
    session_maker: async_sessionmaker[AsyncSession],
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
):
    async with session_maker() as session:
        return await repo.get_conversation(
            session, conversation_id, user_id)


async def _mark_failed_bg(
    session_maker: async_sessionmaker[AsyncSession],
    msg_id: uuid.UUID,
    error: str,
) -> None:
    try:
        async with session_maker() as session:
            await repo.mark_message_failed(session, msg_id, error)
            await session.commit()
    except Exception:
        logger.exception(
            "run_turn: не смогли записать failed для msg %s", msg_id)


def _estimate_usage(
    history: list[dict[str, str]],
    user_input: str,
    system_prompt: str,
    chunks: list[dict],
    answer_text: str,
) -> tuple[int, int]:
    """Оценка usage через HF-токенайзер"""
    tokenizer = get_tokenizer(settings.tokenizer_repo)
    prompt_parts = [system_prompt, user_input]
    for msg in history:
        prompt_parts.append(msg.get("content", ""))
    for hit in chunks:
        prompt_parts.append(hit.get("_source", {}).get("content", ""))

    prompt_tokens = sum(count_tokens(p, tokenizer) for p in prompt_parts)
    completion_tokens = count_tokens(answer_text, tokenizer)
    return prompt_tokens, completion_tokens
