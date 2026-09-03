from __future__ import annotations

"""Тесты репозиторных CRUD-функций.

Ядро проверяемого поведения:
- скоупинг по user_id: чужие ресурсы всегда → NotFoundOrForbidden;
- partial update feedback: ключи из data, не упомянутые в patch,
  сохраняются (JSONB `||` в Postgres / python-merge в SQLite);
- транзакционность: репозиторий НЕ коммитит, только flush; тест сам
  коммитит когда нужно.

Тесты через in-memory SQLite. JSONB `||` в SQLite fallback идёт через
python-merge — семантически то же самое, но проверить синтаксис
Postgres-специфичного SQL мы здесь не можем. Отдельный @live_only тест
против реального Postgres — задел на 2.9.
"""

import unittest
import uuid

from app.db import repository as repo
from app.db.models import Conversation, Message
from app.db.repository import NotFoundOrForbidden, SourceIn
from app.db.session import init_db, make_engine, make_session_maker


USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER = uuid.UUID("22222222-2222-2222-2222-222222222222")
RAG = uuid.UUID("33333333-3333-3333-3333-333333333333")
DOC = uuid.UUID("44444444-4444-4444-4444-444444444444")


class _RepoBase(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.engine = make_engine("sqlite+aiosqlite:///:memory:")
        # FK constraint в SQLite не enforcened по умолчанию.
        async with self.engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("PRAGMA foreign_keys=ON"))
        await init_db(self.engine)
        self.session_maker = make_session_maker(self.engine)

    async def asyncTearDown(self):
        await self.engine.dispose()


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

class ConversationsRepoTests(_RepoBase):

    async def test_create_and_get(self):
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG, title="тест")
            await s.commit()

        async with self.session_maker() as s:
            fetched = await repo.get_conversation(s, conv.id, USER)
            self.assertEqual(fetched.title, "тест")
            self.assertEqual(fetched.rag_id, RAG)

    async def test_get_by_other_user_raises(self):
        """Чужая conversation — не пустой ответ, а исключение с одним и
        тем же текстом, чтобы 404 в API-слое не отличался от 'нет такой'."""
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG)
            await s.commit()

        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.get_conversation(s, conv.id, OTHER)

    async def test_get_missing_raises(self):
        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.get_conversation(s, uuid.uuid4(), USER)

    async def test_list_scoped_to_user(self):
        """Пользователь видит только свои. Чужие даже не считаются."""
        async with self.session_maker() as s:
            await repo.create_conversation(s, user_id=USER, rag_id=RAG,
                                           title="мой 1")
            await repo.create_conversation(s, user_id=USER, rag_id=RAG,
                                           title="мой 2")
            await repo.create_conversation(s, user_id=OTHER, rag_id=RAG,
                                           title="чужой")
            await s.commit()

        async with self.session_maker() as s:
            mine = await repo.list_conversations(s, USER)
            self.assertEqual({c.title for c in mine}, {"мой 1", "мой 2"})

    async def test_list_ordered_by_updated_at_desc(self):
        """Недавно активные — сверху. Порядок определяется updated_at,
        не created_at, чтобы старый чат с новым сообщением поднимался."""
        async with self.session_maker() as s:
            first = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG, title="старый")
            second = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG, title="новее")
            await s.commit()
            # Обновляем первый — он должен подняться в списке.
            await repo.touch_conversation(s, first.id)
            await s.commit()

        async with self.session_maker() as s:
            listed = await repo.list_conversations(s, USER)
            self.assertEqual(listed[0].id, first.id,
                             "touch должен поднять чат в списке")

    async def test_update_title_scoped(self):
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG, title="старое")
            await s.commit()

        async with self.session_maker() as s:
            await repo.update_conversation_title(s, conv.id, USER, "новое")
            await s.commit()

        async with self.session_maker() as s:
            fresh = await repo.get_conversation(s, conv.id, USER)
            self.assertEqual(fresh.title, "новое")

    async def test_update_title_by_other_raises(self):
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG)
            await s.commit()

        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.update_conversation_title(
                    s, conv.id, OTHER, "hijack")

    async def test_delete_scoped(self):
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG)
            await s.commit()

        # Чужой удалить не может.
        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.delete_conversation(s, conv.id, OTHER)

        # Свой — может.
        async with self.session_maker() as s:
            await repo.delete_conversation(s, conv.id, USER)
            await s.commit()

        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.get_conversation(s, conv.id, USER)


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

class MessagesRepoTests(_RepoBase):

    async def _new_conv(self) -> uuid.UUID:
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG)
            await s.commit()
            return conv.id

    async def test_add_user_message_touches_conversation(self):
        """Добавление сообщения обновляет updated_at чата."""
        conv_id = await self._new_conv()
        async with self.session_maker() as s:
            before = (await repo.get_conversation(s, conv_id, USER)).updated_at

        async with self.session_maker() as s:
            await repo.add_user_message(s, conv_id, "привет")
            await s.commit()

        async with self.session_maker() as s:
            after = (await repo.get_conversation(s, conv_id, USER)).updated_at
            self.assertGreaterEqual(after, before)

    async def test_pending_message_lifecycle(self):
        """pending → mark_ok меняет content и статус."""
        conv_id = await self._new_conv()
        async with self.session_maker() as s:
            msg = await repo.add_pending_assistant_message(s, conv_id)
            await s.commit()
            self.assertEqual(msg.status, "pending")
            self.assertEqual(msg.content, "")

        async with self.session_maker() as s:
            await repo.mark_message_ok(s, msg.id, "готовый ответ")
            await s.commit()

        async with self.session_maker() as s:
            fresh = await repo.get_message(s, msg.id, USER)
            self.assertEqual(fresh.status, "ok")
            self.assertEqual(fresh.content, "готовый ответ")
            self.assertIsNone(fresh.error)

    async def test_pending_to_failed_records_error(self):
        conv_id = await self._new_conv()
        async with self.session_maker() as s:
            msg = await repo.add_pending_assistant_message(s, conv_id)
            await s.commit()

        async with self.session_maker() as s:
            await repo.mark_message_failed(
                s, msg.id, "ollama не ответил")
            await s.commit()

        async with self.session_maker() as s:
            fresh = await repo.get_message(s, msg.id, USER)
            self.assertEqual(fresh.status, "failed")
            self.assertEqual(fresh.error, "ollama не ответил")

    async def test_get_message_scoped(self):
        conv_id = await self._new_conv()
        async with self.session_maker() as s:
            msg = await repo.add_user_message(s, conv_id, "q")
            await s.commit()

        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.get_message(s, msg.id, OTHER)

    async def test_list_messages_includes_failed(self):
        """list_messages — для UI, показывает все включая failed. Фильтр
        `status='ok'` — только при сборке истории для rewriter (2.3.c)."""
        conv_id = await self._new_conv()
        async with self.session_maker() as s:
            await repo.add_user_message(s, conv_id, "q1")
            await s.commit()

            msg = await repo.add_pending_assistant_message(s, conv_id)
            await s.commit()
            await repo.mark_message_failed(s, msg.id, "err")
            await s.commit()

        async with self.session_maker() as s:
            listed = await repo.list_messages(s, conv_id, USER)
            statuses = [m.status for m in listed]
            self.assertIn("failed", statuses,
                          "failed сообщения обязаны быть видны в UI")

    async def test_list_messages_scoped(self):
        conv_id = await self._new_conv()
        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.list_messages(s, conv_id, OTHER)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

class SourcesRepoTests(_RepoBase):

    async def _new_message(self) -> uuid.UUID:
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG)
            msg = await repo.add_pending_assistant_message(s, conv.id)
            await s.commit()
            return msg.id

    async def test_add_and_list_ordered(self):
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            await repo.add_sources(s, msg_id, [
                SourceIn(chunk_id="h1", document_id=DOC,
                         chunk_index=0, filename="a.pdf",
                         rag_id=RAG, order=1),
                SourceIn(chunk_id="h3", document_id=DOC,
                         chunk_index=2, filename="a.pdf",
                         rag_id=RAG, order=3),
                SourceIn(chunk_id="h2", document_id=DOC,
                         chunk_index=1, filename="a.pdf",
                         rag_id=RAG, order=2),
            ])
            await s.commit()

        async with self.session_maker() as s:
            listed = await repo.list_sources(s, msg_id, USER)
            # По order, не по порядку вставки — [1] в тексте ответа
            # должен ссылаться на source с order=1.
            self.assertEqual([r.order for r in listed], [1, 2, 3])
            self.assertEqual([r.chunk_id for r in listed],
                             ["h1", "h2", "h3"])

    async def test_add_empty_noop(self):
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            await repo.add_sources(s, msg_id, [])
            await s.commit()
        async with self.session_maker() as s:
            listed = await repo.list_sources(s, msg_id, USER)
            self.assertEqual(listed, [])

    async def test_list_scoped(self):
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.list_sources(s, msg_id, OTHER)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

class UsageRepoTests(_RepoBase):

    async def _new_message(self) -> uuid.UUID:
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG)
            msg = await repo.add_pending_assistant_message(s, conv.id)
            await s.commit()
            return msg.id

    async def test_set_usage_writes_total(self):
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            await repo.set_usage(
                s, msg_id, prompt_tokens=100,
                completion_tokens=50, model="qwen3:8b")
            await s.commit()

        async with self.session_maker() as s:
            msg = await repo.get_message(s, msg_id, USER)
            # relationship usage подтягиваем через refresh, иначе
            # lazy-load в async вызывает MissingGreenlet.
            await s.refresh(msg, attribute_names=["usage"])
            self.assertEqual(msg.usage.prompt_tokens, 100)
            self.assertEqual(msg.usage.completion_tokens, 50)
            self.assertEqual(msg.usage.total_tokens, 150)
            self.assertEqual(msg.usage.model, "qwen3:8b")

    async def test_set_usage_upsert_updates(self):
        """Повторный вызов на том же message_id — перезаписывает, не
        создаёт дубль. Защита от race при retry API-слоя."""
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            await repo.set_usage(s, msg_id, prompt_tokens=100,
                                 completion_tokens=50, model="qwen3:8b")
            await s.commit()
            await repo.set_usage(s, msg_id, prompt_tokens=200,
                                 completion_tokens=75, model="qwen3:8b")
            await s.commit()

        async with self.session_maker() as s:
            msg = await repo.get_message(s, msg_id, USER)
            await s.refresh(msg, attribute_names=["usage"])
            self.assertEqual(msg.usage.prompt_tokens, 200)
            self.assertEqual(msg.usage.total_tokens, 275)


# ---------------------------------------------------------------------------
# Feedback (partial update)
# ---------------------------------------------------------------------------

class FeedbackRepoTests(_RepoBase):

    async def _new_message(self) -> uuid.UUID:
        async with self.session_maker() as s:
            conv = await repo.create_conversation(
                s, user_id=USER, rag_id=RAG)
            msg = await repo.add_pending_assistant_message(s, conv.id)
            await s.commit()
            return msg.id

    async def test_first_upsert_creates(self):
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            fb = await repo.upsert_feedback(
                s, msg_id, USER, {"rating": 5})
            await s.commit()
            self.assertEqual(fb.data, {"rating": 5})

    async def test_second_upsert_merges_keys(self):
        """Инвариант partial update: existing ключи, не в patch, остаются."""
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            await repo.upsert_feedback(
                s, msg_id, USER, {"rating": 5, "helpful": True})
            await s.commit()

        async with self.session_maker() as s:
            fb = await repo.upsert_feedback(
                s, msg_id, USER, {"comment": "хорошо"})
            await s.commit()
            # rating и helpful должны сохраниться, comment — добавиться.
            self.assertEqual(fb.data, {
                "rating": 5, "helpful": True, "comment": "хорошо",
            })

    async def test_upsert_overwrites_same_key(self):
        """Ключ из patch перезаписывает существующий, а не мержит вглубь."""
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            await repo.upsert_feedback(
                s, msg_id, USER, {"rating": 1})
            await s.commit()

        async with self.session_maker() as s:
            fb = await repo.upsert_feedback(
                s, msg_id, USER, {"rating": 5})
            await s.commit()
            self.assertEqual(fb.data["rating"], 5)

    async def test_upsert_by_other_user_forbidden(self):
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.upsert_feedback(
                    s, msg_id, OTHER, {"rating": 5})

    async def test_get_feedback_returns_none_when_absent(self):
        """Отсутствие feedback ≠ нет доступа. get не бросает, а даёт None —
        UI это переваривает как «не оценивали»."""
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            fb = await repo.get_feedback(s, msg_id, USER)
            self.assertIsNone(fb)

    async def test_get_feedback_by_other_user_forbidden(self):
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            await repo.upsert_feedback(s, msg_id, USER, {"rating": 5})
            await s.commit()

        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await repo.get_feedback(s, msg_id, OTHER)

    async def test_delete_returns_true_only_when_deleted(self):
        """Идемпотентный delete: True — было и удалили, False — не было."""
        msg_id = await self._new_message()
        async with self.session_maker() as s:
            deleted = await repo.delete_feedback(s, msg_id, USER)
            self.assertFalse(deleted, "ещё не создан — удалять нечего")

        async with self.session_maker() as s:
            await repo.upsert_feedback(s, msg_id, USER, {"rating": 5})
            await s.commit()

        async with self.session_maker() as s:
            deleted = await repo.delete_feedback(s, msg_id, USER)
            await s.commit()
            self.assertTrue(deleted)

        async with self.session_maker() as s:
            deleted = await repo.delete_feedback(s, msg_id, USER)
            self.assertFalse(deleted, "повторный delete идемпотентен")


if __name__ == "__main__":
    unittest.main()
