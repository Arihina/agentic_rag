from __future__ import annotations

"""Тесты HTTP-эндпоинтов Responses API.

Проверяем HTTP-слой изолированно от бизнес-логики (последняя покрыта
test_run_turn):
- POST /v1/responses — что заворот в StreamingResponse работает, хедеры
  правильные, поток SSE-фреймов доходит до клиента;
- GET /v1/responses/{id} — снапшот собирается в ResponseObject правильно
  для всех трёх статусов; парсинг id в трёх формах; скоупинг по user.

Не проверяем здесь: cancel через is_disconnected — TestClient не
эмулирует обрыв trans портом; всё покрыто test_run_turn.
"""

import json
import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.db import repository as repo
from app.db.session import init_db, make_engine, make_session_maker
from app.state import state
from tests import base
from tests.core_fakes import FakeEmbed, FakeLLM, FakeOpenSearch, hit
from tests.support import FakeIngest


USER_ID = uuid.UUID(base.USER_ID)
OTHER_ID = uuid.UUID(base.OTHER_USER_ID)
RAG = uuid.UUID("33333333-3333-3333-3333-333333333333")
DOC = uuid.UUID("44444444-4444-4444-4444-444444444444")


class _CharTokenizer:
    def encode(self, text: str) -> list[str]:
        return list(text)


class _ResponsesHttpBase(unittest.TestCase):

    def setUp(self):
        # Реальный SQLite in-memory + FK ON.
        import asyncio
        self.loop = asyncio.new_event_loop()

        self.engine = self.loop.run_until_complete(self._init_engine())
        self.session_maker = make_session_maker(self.engine)

        # Fakes для всего остального.
        self.ingest = FakeIngest()
        self.ingest.set_rag(RAG, status="ready", top_k=5,
                            score_threshold=0.0, temperature=0.3)
        self.llm = FakeLLM()
        self.embed = FakeEmbed()
        self.os_client = FakeOpenSearch(responses=[[hit(
            "chunk-1",
            content="Ответ найден.",
            document_id=str(DOC),
            chunk_index=0,
            headings=["Раздел"],
            pages=[1])]])

        # Мокаем токенайзер, чтобы не тянуть HF-модель.
        from app.services import run_turn as run_turn_mod
        self._tokenizer_patch = patch.object(
            run_turn_mod, "get_tokenizer",
            return_value=_CharTokenizer())
        self._tokenizer_patch.start()

        # Импортируем main поздно — settings уже прочитаны.
        from app import main
        self.main = main

        # Подмена state — уже ПОСЛЕ входа в контекст TestClient (в тестах).
        self._originals: dict = {}

    async def _init_engine(self):
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("PRAGMA foreign_keys=ON"))
        await init_db(engine)
        return engine

    def _override_state(self, **fakes):
        for name, obj in fakes.items():
            self._originals[name] = getattr(state, name, None)
            setattr(state, name, obj)

    def _restore_state(self):
        for name, obj in self._originals.items():
            if obj is not None:
                setattr(state, name, obj)
        self._originals.clear()

    def _override_all(self):
        self._override_state(
            os_client=self.os_client, llm=self.llm,
            embed=self.embed, ingest=self.ingest,
            session_maker=self.session_maker)

    def tearDown(self):
        self._tokenizer_patch.stop()
        self.loop.run_until_complete(self.engine.dispose())
        self.loop.close()

    def _make_conversation(self, rag_id=RAG, user=USER_ID) -> uuid.UUID:
        async def _mk():
            async with self.session_maker() as s:
                conv = await repo.create_conversation(
                    s, user_id=user, rag_id=rag_id)
                await s.commit()
                return conv.id
        return self.loop.run_until_complete(_mk())


class PostResponsesTests(_ResponsesHttpBase):

    def test_streaming_headers(self):
        """SSE-хедеры: text/event-stream, no-cache, X-Accel-Buffering off.
        Без последнего nginx буферизует потоки по 4К и стрим ломается."""
        conv_id = self._make_conversation()
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                with client.stream(
                    "POST", "/v1/responses",
                    headers={"X-User-Id": str(USER_ID)},
                    json={
                        "model": f"rag/{RAG}",
                        "input": "как настроить X",
                        "conversation_id": str(conv_id),
                    },
                ) as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("text/event-stream",
                                  response.headers["content-type"])
                    self.assertEqual(response.headers["cache-control"],
                                     "no-cache")
                    self.assertEqual(response.headers["x-accel-buffering"],
                                     "no")
                    # Прочитаем до конца, чтобы фон отработал.
                    _ = b"".join(response.iter_bytes())
            finally:
                self._restore_state()

    def test_sse_frames_delivered(self):
        """POST реально отдаёт все ожидаемые события — по факту через
        TestClient видим стрим."""
        conv_id = self._make_conversation()
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                with client.stream(
                    "POST", "/v1/responses",
                    headers={"X-User-Id": str(USER_ID)},
                    json={
                        "model": f"rag/{RAG}",
                        "input": "q",
                        "conversation_id": str(conv_id),
                    },
                ) as response:
                    body = b"".join(response.iter_bytes())
            finally:
                self._restore_state()

        types = _parse_event_types(body)
        self.assertIn("response.created", types)
        self.assertIn("response.in_progress", types)
        self.assertIn("response.output_text.delta", types)
        self.assertIn("response.completed", types)

    def test_requires_x_user_id(self):
        """Без X-User-Id — 401."""
        conv_id = self._make_conversation()
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                response = client.post(
                    "/v1/responses",
                    json={
                        "model": f"rag/{RAG}",
                        "input": "q",
                        "conversation_id": str(conv_id),
                    })
            finally:
                self._restore_state()

        self.assertEqual(response.status_code, 401)

    def test_pydantic_validation_400(self):
        """Пустой input → 400 от pydantic (extra='forbid', min_length=1)."""
        conv_id = self._make_conversation()
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                response = client.post(
                    "/v1/responses",
                    headers={"X-User-Id": str(USER_ID)},
                    json={
                        "model": f"rag/{RAG}",
                        "input": "",  # пустой
                        "conversation_id": str(conv_id),
                    })
            finally:
                self._restore_state()

        self.assertEqual(response.status_code, 400)

    def test_error_in_stream_for_missing_conversation(self):
        """Несуществующий conversation_id → не 404 на HTTP-уровне, а
        SSE поток с одним response.error. Так контракт согласован: клиент
        уже установил стрим и не должен по 404 отваливаться."""
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                with client.stream(
                    "POST", "/v1/responses",
                    headers={"X-User-Id": str(USER_ID)},
                    json={
                        "model": f"rag/{RAG}",
                        "input": "q",
                        "conversation_id": str(uuid.uuid4()),  # нет такого
                    },
                ) as response:
                    self.assertEqual(response.status_code, 200)
                    body = b"".join(response.iter_bytes())
            finally:
                self._restore_state()

        types = _parse_event_types(body)
        self.assertEqual(types, ["response.error"])


class GetResponsesTests(_ResponsesHttpBase):

    def _run_turn_and_get_message_id(self, conv_id: uuid.UUID) -> str:
        """Утилита: провернуть POST /v1/responses, вытащить id из
        первого события и вернуть его (уже в форме resp_<uuid>)."""
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                with client.stream(
                    "POST", "/v1/responses",
                    headers={"X-User-Id": str(USER_ID)},
                    json={
                        "model": f"rag/{RAG}",
                        "input": "q",
                        "conversation_id": str(conv_id),
                    },
                ) as response:
                    body = b"".join(response.iter_bytes())
            finally:
                self._restore_state()

        # response.created содержит id.
        for frame in body.split(b"\n\n"):
            if frame.startswith(b"event: response.created"):
                data_line = [l for l in frame.split(b"\n")
                             if l.startswith(b"data: ")][0]
                data = json.loads(data_line[len(b"data: "):].decode())
                return data["id"]
        raise AssertionError("нет response.created в стриме")

    def test_snapshot_completed(self):
        """После happy path GET возвращает completed с output и usage."""
        conv_id = self._make_conversation()
        resp_id = self._run_turn_and_get_message_id(conv_id)

        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                response = client.get(
                    f"/v1/responses/{resp_id}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(response.status_code, 200)
        body = response.json()

        self.assertEqual(body["id"], resp_id)
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["model"], f"rag/{RAG}")
        self.assertEqual(body["conversation_id"], str(conv_id))
        self.assertEqual(len(body["output"]), 1)
        self.assertEqual(body["output"][0]["content"][0]["text"], "ответ")
        self.assertIsNotNone(body["usage"])
        self.assertGreater(body["usage"]["total_tokens"], 0)

    def test_snapshot_accepts_chatcmpl_prefix(self):
        """Клиент, пришедший из chat/completions-контекста, может
        сохранить id как chatcmpl-<uuid>. GET должен принимать все три
        формы."""
        conv_id = self._make_conversation()
        resp_id = self._run_turn_and_get_message_id(conv_id)
        # Обрубаем префикс resp_ и подставляем chatcmpl-.
        bare_uuid = resp_id.removeprefix("resp_")

        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r1 = client.get(
                    f"/v1/responses/chatcmpl-{bare_uuid}",
                    headers={"X-User-Id": str(USER_ID)})
                r2 = client.get(
                    f"/v1/responses/{bare_uuid}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        # id на выходе всегда resp_<uuid>, независимо от формы на входе.
        self.assertEqual(r1.json()["id"], resp_id)
        self.assertEqual(r2.json()["id"], resp_id)

    def test_invalid_id_400(self):
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                response = client.get(
                    "/v1/responses/not-a-valid-id",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(response.status_code, 400)

    def test_snapshot_other_user_404(self):
        """Чужой ответ = 404 (то же, что несуществующий). Не даём
        отличить."""
        conv_id = self._make_conversation()
        resp_id = self._run_turn_and_get_message_id(conv_id)

        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                response = client.get(
                    f"/v1/responses/{resp_id}",
                    headers={"X-User-Id": str(OTHER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(response.status_code, 404)

    def test_snapshot_missing_404(self):
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                response = client.get(
                    f"/v1/responses/resp_{uuid.uuid4()}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(response.status_code, 404)


def _parse_event_types(body: bytes) -> list[str]:
    """Из SSE-стрима байтов вытащить упорядоченный список типов событий."""
    types = []
    for frame in body.split(b"\n\n"):
        if frame.startswith(b"event: "):
            first_line = frame.split(b"\n", 1)[0]
            event_type = first_line.removeprefix(b"event: ").decode()
            types.append(event_type)
    return types


if __name__ == "__main__":
    unittest.main()
