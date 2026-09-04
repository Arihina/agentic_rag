from __future__ import annotations

"""Тесты оркестратора run_turn.

Собираем реальные компоненты (Postgres → SQLite, ingest → FakeIngest,
core → Fake*), гоняем end-to-end и проверяем как последовательность
SSE-событий, так и состояние БД по итогу.

Ключевые сценарии:
- happy path: правильные события, ok в БД, sources и usage записаны;
- каждая ветка ранней ошибки (mismatch, unavailable, unknown model)
  → error event + БД не тронута (нет pending сообщения);
- ошибка в run_agent → failed в БД + response.error;
- cancel через is_disconnected → failed в БД БЕЗ response.error/completed.

Тест-хелпер: FakeTokenizer уже используется в test_history.py, но для
run_turn прокидывать его через все слои неудобно — используем реальный
lru_cache get_tokenizer с мок-подменой на CharTokenizer через monkey
патч. Это единственное место, где не через DI: get_tokenizer — глобальный
кеш по дизайну (см. 2.3.c).
"""

import unittest
import uuid
from unittest.mock import patch

from app.db import repository as repo
from app.db.session import init_db, make_engine, make_session_maker
from app.schemas.responses import ResponsesRequest
from app.services import run_turn as run_turn_mod
from app.services.run_turn import run_turn
from tests.core_fakes import FakeEmbed, FakeLLM, FakeOpenSearch, hit
from tests.support import FakeIngest


USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER = uuid.UUID("22222222-2222-2222-2222-222222222222")
RAG = uuid.UUID("33333333-3333-3333-3333-333333333333")
OTHER_RAG = uuid.UUID("44444444-4444-4444-4444-444444444444")
DOC = uuid.UUID("55555555-5555-5555-5555-555555555555")


class _CharTokenizer:
    """Для тестов usage-оценки. 1 char = 1 token."""

    def encode(self, text: str) -> list[str]:
        return list(text)


class _RunTurnBase(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # SQLite in-memory + FK ON.
        self.engine = make_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("PRAGMA foreign_keys=ON"))
        await init_db(self.engine)
        self.session_maker = make_session_maker(self.engine)

        self.ingest = FakeIngest()
        self.ingest.set_rag(RAG, status="ready", top_k=5,
                            score_threshold=0.0, temperature=0.3)

        self.llm = FakeLLM()
        self.embed = FakeEmbed()
        self.os_client = FakeOpenSearch(responses=[[hit(
            "chunk-1",
            content="Ответ находится в документе.",
            document_id=str(DOC),
            chunk_index=0,
            headings=["Раздел"],
            pages=[1])]])

        # get_tokenizer — глобальный lru_cache; мокаем на CharTokenizer,
        # чтобы _estimate_usage не тянул HF-модель в CI.
        self._tokenizer_patch = patch.object(
            run_turn_mod, "get_tokenizer",
            return_value=_CharTokenizer())
        self._tokenizer_patch.start()

    async def asyncTearDown(self):
        self._tokenizer_patch.stop()
        await self.engine.dispose()

    async def _make_conversation(self, rag_id=RAG, user=USER) -> uuid.UUID:
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=user, rag_id=rag_id)
            await s.commit()
            return conv.id

    async def _collect(
        self, req, user=USER, is_disconnected=None,
    ) -> list[bytes]:
        gen = run_turn(
            request_body=req, user_id=user,
            session_maker=self.session_maker,
            ingest=self.ingest, llm=self.llm,
            embed=self.embed, os_client=self.os_client,
            is_disconnected=is_disconnected)
        return [chunk async for chunk in gen]


class HappyPathTests(_RunTurnBase):

    async def test_full_event_sequence(self):
        """Правильные события в правильном порядке."""
        conv_id = await self._make_conversation()
        req = ResponsesRequest(
            model=f"rag/{RAG}", input="как настроить X",
            conversation_id=conv_id)

        chunks = await self._collect(req)
        types = [_event_type(c) for c in chunks]

        self.assertEqual(types, [
            "response.created",
            "response.in_progress",
            "response.output_text.delta",
            "response.completed",
        ])

    async def test_message_persisted_as_ok(self):
        """После happy path сообщение в БД — status=ok, content = ответ."""
        conv_id = await self._make_conversation()
        req = ResponsesRequest(
            model=f"rag/{RAG}", input="q",
            conversation_id=conv_id)
        await self._collect(req)

        async with self.session_maker() as s:
            msgs = await repo.list_messages(s, conv_id, USER)
        # user + assistant
        self.assertEqual(len(msgs), 2)
        assistant = msgs[1]
        self.assertEqual(assistant.role, "assistant")
        self.assertEqual(assistant.status, "ok")
        self.assertEqual(assistant.content, "ответ")  # default from FakeLLM

    async def test_sources_persisted(self):
        """Найденные чанки записываются в message_sources с order по
        порядку из финального пула."""
        conv_id = await self._make_conversation()
        req = ResponsesRequest(
            model=f"rag/{RAG}", input="q",
            conversation_id=conv_id)
        await self._collect(req)

        async with self.session_maker() as s:
            msgs = await repo.list_messages(s, conv_id, USER)
            sources = await repo.list_sources(s, msgs[1].id, USER)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].chunk_id, "chunk-1")
        self.assertEqual(sources[0].document_id, DOC)
        self.assertEqual(sources[0].order, 1)
        self.assertEqual(sources[0].rag_id, RAG)

    async def test_usage_persisted(self):
        """Usage записан, prompt/completion > 0 (не нули)."""
        conv_id = await self._make_conversation()
        req = ResponsesRequest(
            model=f"rag/{RAG}", input="как настроить X",
            conversation_id=conv_id)
        await self._collect(req)

        async with self.session_maker() as s:
            msgs = await repo.list_messages(s, conv_id, USER)
            msg = await repo.get_message(s, msgs[1].id, USER)
            await s.refresh(msg, attribute_names=["usage"])

        self.assertIsNotNone(msg.usage)
        self.assertGreater(msg.usage.prompt_tokens, 0)
        self.assertGreater(msg.usage.completion_tokens, 0)
        self.assertEqual(msg.usage.total_tokens,
                         msg.usage.prompt_tokens + msg.usage.completion_tokens)


class EarlyValidationErrorTests(_RunTurnBase):

    async def test_conversation_not_found(self):
        req = ResponsesRequest(
            model=f"rag/{RAG}", input="q",
            conversation_id=uuid.uuid4())  # несуществующий

        chunks = await self._collect(req)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(_event_type(chunks[0]), "response.error")

    async def test_conversation_belongs_to_other_user(self):
        """Чужой чат — тот же error, что и «нет такого». Не даём отличить."""
        conv_id = await self._make_conversation(user=OTHER)
        req = ResponsesRequest(
            model=f"rag/{RAG}", input="q",
            conversation_id=conv_id)

        chunks = await self._collect(req, user=USER)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(_event_type(chunks[0]), "response.error")

    async def test_model_mismatch_with_conversation(self):
        """conv.rag_id != model.rag_id → 400 без обращения к ingest."""
        conv_id = await self._make_conversation(rag_id=OTHER_RAG)
        self.ingest.set_rag(OTHER_RAG, status="ready")
        req = ResponsesRequest(
            # model просит RAG, conv на OTHER_RAG
            model=f"rag/{RAG}", input="q",
            conversation_id=conv_id)

        chunks = await self._collect(req)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(_event_type(chunks[0]), "response.error")
        # Ingestion не спрашивали — проверка сразу на сопоставлении.
        self.assertEqual(self.ingest.get_rag_calls, [])

    async def test_rag_not_found(self):
        conv_id = await self._make_conversation()
        # НЕ вызываем set_rag(RAG) — FakeIngest вернёт RagNotFound
        self.ingest = FakeIngest()  # чистый, без наших наборов
        req = ResponsesRequest(
            model=f"rag/{RAG}", input="q",
            conversation_id=conv_id)

        chunks = await self._collect(req)
        self.assertEqual(_event_type(chunks[0]), "response.error")

    async def test_rag_unavailable_status(self):
        """Набор существует, но status != 'ready' → error event."""
        conv_id = await self._make_conversation()
        self.ingest.set_rag(RAG, status="ingesting")
        req = ResponsesRequest(
            model=f"rag/{RAG}", input="q",
            conversation_id=conv_id)

        chunks = await self._collect(req)
        self.assertEqual(_event_type(chunks[0]), "response.error")

    async def test_no_pending_message_on_early_error(self):
        """Регрессия: если валидация упала до create_pending_assistant,
        в БД не должно оставаться сирот-сообщений."""
        conv_id = await self._make_conversation()
        req = ResponsesRequest(
            model="rag/00000000-0000-0000-0000-000000000000",  # unknown rag
            input="q",
            conversation_id=conv_id)

        await self._collect(req)

        async with self.session_maker() as s:
            msgs = await repo.list_messages(s, conv_id, USER)
        self.assertEqual(msgs, [], "pending не должен создаваться до "
                         "успешной валидации")


class AgentFailureTests(_RunTurnBase):

    async def test_agent_exception_marks_failed_and_sends_error(self):
        """Исключение из run_agent → status=failed в БД + response.error."""
        conv_id = await self._make_conversation()
        # Заставляем evaluate падать через custom callback.
        from app.core.evaluation import EvalResult

        def failing_eval(prompt, chunks):
            raise RuntimeError("ollama таймаут")

        self.llm = FakeLLM(evaluator=failing_eval)

        req = ResponsesRequest(
            model=f"rag/{RAG}", input="q",
            conversation_id=conv_id)

        chunks = await self._collect(req)
        types = [_event_type(c) for c in chunks]

        # response.created + in_progress уже ушли (pending был создан),
        # потом на eval упало — error.
        self.assertIn("response.created", types)
        self.assertIn("response.error", types)
        self.assertNotIn("response.completed", types)

        # БД: сообщение помечено failed, error содержит текст.
        async with self.session_maker() as s:
            msgs = await repo.list_messages(s, conv_id, USER)
        assistant = msgs[1]
        self.assertEqual(assistant.status, "failed")
        self.assertIn("ollama таймаут", assistant.error or "")


class ClientDisconnectTests(_RunTurnBase):

    async def test_disconnect_before_completed_marks_failed(self):
        """is_disconnected возвращает True → cancel_hook в run_agent
        бросает ClientDisconnected → mark_failed 'client_disconnected'.

        Ключевой инвариант: НЕТ response.completed и НЕТ response.error
        (клиента уже нет — некому слушать)."""
        conv_id = await self._make_conversation()

        # Возвращаем True со второго вызова — первый (в самом начале
        # run_agent, до rewriter) True, но без разницы: результат один.
        call_count = {"n": 0}

        async def is_disc():
            call_count["n"] += 1
            return True

        req = ResponsesRequest(
            model=f"rag/{RAG}", input="q",
            conversation_id=conv_id)

        chunks = await self._collect(req, is_disconnected=is_disc)
        types = [_event_type(c) for c in chunks]

        # created + in_progress ушли ДО того, как мы обратились к cancel_hook
        # (они yield'ятся до run_agent). Дальше — тишина.
        self.assertIn("response.created", types)
        self.assertNotIn("response.completed", types)
        self.assertNotIn("response.error", types)

        # БД: failed с 'client_disconnected'.
        async with self.session_maker() as s:
            msgs = await repo.list_messages(s, conv_id, USER)
        assistant = msgs[1]
        self.assertEqual(assistant.status, "failed")
        self.assertEqual(assistant.error, "client_disconnected")


def _event_type(frame: bytes) -> str:
    """Первая строка SSE-frame'а: `event: <type>`."""
    return frame.decode("utf-8").split("\n", 1)[0].removeprefix("event: ")


if __name__ == "__main__":
    unittest.main()
