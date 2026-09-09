from __future__ import annotations

"""HTTP-тесты Platform conversations CRUD.

Проверяем каждую ручку по трём осям:
- happy path — работает как обещано в схеме;
- скоупинг по X-User-Id — чужой ресурс = 404 (тот же, что и «нет»);
- негативы — 401 без X-User-Id, 400 на битом теле, 502 на сбое ingestion.

Тесты через TestClient; sessions/ingest подменяем на fakes в setUp
(паттерн из test_health и test_api_responses).
"""

import unittest
import uuid

from fastapi.testclient import TestClient

from app.clients.ingest import IngestError
from app.db import repository as repo
from app.db.session import init_db, make_engine, make_session_maker
from app.state import state
from tests import base
from tests.support import FakeIngest


USER_ID = uuid.UUID(base.USER_ID)
OTHER_ID = uuid.UUID(base.OTHER_USER_ID)
RAG = uuid.UUID("33333333-3333-3333-3333-333333333333")


class _ConversationsHttpBase(unittest.TestCase):
    """Общий setUp: реальный SQLite + FakeIngest, override в state."""

    def setUp(self):
        import asyncio
        self.loop = asyncio.new_event_loop()

        self.engine = self.loop.run_until_complete(self._init_engine())
        self.session_maker = make_session_maker(self.engine)

        self.ingest = FakeIngest()
        self.ingest.set_rag(RAG, status="ready")

        from app import main
        self.main = main
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
            ingest=self.ingest, session_maker=self.session_maker)

    def tearDown(self):
        self.loop.run_until_complete(self.engine.dispose())
        self.loop.close()

    def _seed_conversation(self, user=USER_ID, title="X",
                           rag_id=RAG) -> uuid.UUID:
        async def _mk():
            async with self.session_maker() as s:
                conv = await repo.create_conversation(
                    s, user_id=user, rag_id=rag_id, title=title)
                await s.commit()
                return conv.id
        return self.loop.run_until_complete(_mk())


class CreateConversationTests(_ConversationsHttpBase):

    def test_happy_path_creates(self):
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.post(
                    "/v1/platform/conversations",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"rag_id": str(RAG), "title": "Первый чат"})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 201)
        body = r.json()
        self.assertEqual(body["rag_id"], str(RAG))
        self.assertEqual(body["title"], "Первый чат")
        self.assertIn("id", body)
        self.assertIn("created_at", body)
        self.assertIn("updated_at", body)

    def test_without_title(self):
        """title опционален — None по умолчанию, в БД null."""
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.post(
                    "/v1/platform/conversations",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"rag_id": str(RAG)})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 201)
        self.assertIsNone(r.json()["title"])

    def test_rag_not_found_404(self):
        """Ingest вернул RagNotFound → 404."""
        with TestClient(self.main.app) as client:
            # RAG не зарегистрирован в fake ingest.
            self.ingest = FakeIngest()
            self._override_all()
            try:
                r = client.post(
                    "/v1/platform/conversations",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"rag_id": str(RAG)})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 404)

    def test_empty_rag_accepted(self):
        """Ключевой инвариант 2.5.a: чат на пустой набор — можно создать."""
        with TestClient(self.main.app) as client:
            self.ingest = FakeIngest()
            self.ingest.set_rag(RAG, status="empty")
            self._override_all()
            try:
                r = client.post(
                    "/v1/platform/conversations",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"rag_id": str(RAG)})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 201,
                         "empty набор не должен блокировать создание чата")

    def test_ingest_down_502(self):
        """Ingestion не ответил → 502."""
        with TestClient(self.main.app) as client:
            self.ingest = FakeIngest()
            self.ingest.set_error(RAG, IngestError("connection refused"))
            self._override_all()
            try:
                r = client.post(
                    "/v1/platform/conversations",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"rag_id": str(RAG)})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 502)

    def test_requires_x_user_id(self):
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.post(
                    "/v1/platform/conversations",
                    json={"rag_id": str(RAG)})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 401)

    def test_invalid_body_400(self):
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                # Пустое тело — нет обязательного rag_id.
                r = client.post(
                    "/v1/platform/conversations",
                    headers={"X-User-Id": str(USER_ID)},
                    json={})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 400)

    def test_extra_field_400(self):
        """extra='forbid' в схеме: клиент, попытавшийся передать
        conversation_id/model/etc, получает 400 сразу."""
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.post(
                    "/v1/platform/conversations",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"rag_id": str(RAG), "model": "rag/foo"})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 400)


class ListConversationsTests(_ConversationsHttpBase):

    def test_empty_list(self):
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.get(
                    "/v1/platform/conversations",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"data": []})

    def test_only_own_conversations(self):
        """Скоупинг: чужие в список не попадают."""
        self._seed_conversation(user=USER_ID, title="Мой 1")
        self._seed_conversation(user=USER_ID, title="Мой 2")
        self._seed_conversation(user=OTHER_ID, title="Чужой")

        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.get(
                    "/v1/platform/conversations",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        titles = {c["title"] for c in r.json()["data"]}
        self.assertEqual(titles, {"Мой 1", "Мой 2"})

    def test_requires_x_user_id(self):
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.get("/v1/platform/conversations")
            finally:
                self._restore_state()
        self.assertEqual(r.status_code, 401)


class GetConversationTests(_ConversationsHttpBase):

    def test_get_own(self):
        cid = self._seed_conversation(title="Один")
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.get(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["id"], str(cid))
        self.assertEqual(r.json()["title"], "Один")

    def test_get_other_404(self):
        """Чужой = 404. Не даём отличить от «нет такого»."""
        cid = self._seed_conversation(user=OTHER_ID)
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.get(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()
        self.assertEqual(r.status_code, 404)

    def test_get_missing_404(self):
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.get(
                    f"/v1/platform/conversations/{uuid.uuid4()}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()
        self.assertEqual(r.status_code, 404)

    def test_invalid_uuid_422_or_400(self):
        """Path'овая валидация UUID — FastAPI по умолчанию 422, наш
        RequestValidationError-handler мапит в 400."""
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.get(
                    "/v1/platform/conversations/not-a-uuid",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()
        self.assertEqual(r.status_code, 400)


class UpdateConversationTests(_ConversationsHttpBase):

    def test_rename_own(self):
        cid = self._seed_conversation(title="Старое")
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.patch(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"title": "Новое"})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["title"], "Новое")

    def test_rename_other_404(self):
        cid = self._seed_conversation(user=OTHER_ID, title="Чужой")
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.patch(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"title": "hijack"})
            finally:
                self._restore_state()
        self.assertEqual(r.status_code, 404)

    def test_cannot_change_rag_id(self):
        """Инвариант: rag_id read-only. extra='forbid' на PATCH."""
        cid = self._seed_conversation()
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.patch(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"title": "ok", "rag_id": str(uuid.uuid4())})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 400)

    def test_empty_title_400(self):
        cid = self._seed_conversation()
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.patch(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)},
                    json={"title": ""})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 400)


class DeleteConversationTests(_ConversationsHttpBase):

    def test_delete_own_204(self):
        cid = self._seed_conversation()
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.delete(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 204)
        self.assertEqual(r.content, b"", "204 — без тела")

    def test_deleted_conversation_not_gettable(self):
        """После DELETE — GET того же id вернёт 404."""
        cid = self._seed_conversation()
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r_del = client.delete(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)})
                r_get = client.get(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(r_del.status_code, 204)
        self.assertEqual(r_get.status_code, 404)

    def test_delete_other_404(self):
        cid = self._seed_conversation(user=OTHER_ID)
        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.delete(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()

        self.assertEqual(r.status_code, 404)

    def test_delete_cascades_messages(self):
        """CASCADE вниз: удаление чата уносит все его сообщения.
        Регрессия: без ondelete='CASCADE' в моделях остались бы сироты."""
        cid = self._seed_conversation()

        async def _seed_messages():
            async with self.session_maker() as s:
                await repo.add_user_message(s, cid, "q1")
                m = await repo.add_pending_assistant_message(s, cid)
                await repo.mark_message_ok(s, m.id, "a1")
                await s.commit()

        async def _count_messages():
            async with self.session_maker() as s:
                from sqlalchemy import func, select
                from app.db.models import Message
                r = await s.execute(
                    select(func.count()).select_from(Message)
                    .where(Message.conversation_id == cid))
                return r.scalar()

        self.loop.run_until_complete(_seed_messages())
        before = self.loop.run_until_complete(_count_messages())
        self.assertEqual(before, 2)

        with TestClient(self.main.app) as client:
            self._override_all()
            try:
                r = client.delete(
                    f"/v1/platform/conversations/{cid}",
                    headers={"X-User-Id": str(USER_ID)})
            finally:
                self._restore_state()
        self.assertEqual(r.status_code, 204)

        after = self.loop.run_until_complete(_count_messages())
        self.assertEqual(after, 0, "CASCADE не удалил messages")


if __name__ == "__main__":
    unittest.main()
