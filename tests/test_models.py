from __future__ import annotations

"""Модели: базовые CRUD и инварианты FK/CASCADE.

Тесты через in-memory SQLite (aiosqlite). Всерьёз тестировать модели
без БД нельзя — SQLAlchemy может думать одно, а движок делать другое
(особенно с JSONB и CASCADE). SQLite не Postgres, но общие инварианты
(FK, UNIQUE, autoincrement, default значения) проверяет.

Для JSONB-специфичных вещей (partial update через `||`) будут отдельные
integration-тесты в 2.3.b, где мы уже сможем прогонять против реального
Postgres.
"""

import unittest
import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.db.models import (
    Conversation, Message, MessageFeedback, MessageSource, MessageUsage,
)
from app.db.session import init_db, make_engine, make_session_maker


USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
RAG = uuid.UUID("22222222-2222-2222-2222-222222222222")
DOC = uuid.UUID("33333333-3333-3333-3333-333333333333")


class ModelsCrudTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # SQLite in-memory, свой на каждый тест — иначе тесты видят друг
        # друга через shared состояние engine.
        # foreign_keys=ON в SQLite не по умолчанию; без PRAGMA cascade не
        # работает. asyncpg-Postgres такой ключ соблюдает нативно.
        self.engine = make_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("PRAGMA foreign_keys=ON"))
        await init_db(self.engine)
        self.session_maker = make_session_maker(self.engine)

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def _mk_conversation(self) -> Conversation:
        async with self.session_maker() as s:
            conv = Conversation(user_id=USER, rag_id=RAG, title="test")
            s.add(conv)
            await s.commit()
            await s.refresh(conv)
            return conv

    async def test_create_conversation_defaults(self):
        conv = await self._mk_conversation()
        self.assertIsInstance(conv.id, uuid.UUID)
        self.assertEqual(conv.user_id, USER)
        self.assertEqual(conv.rag_id, RAG)
        self.assertIsNotNone(conv.created_at)
        self.assertIsNotNone(conv.updated_at)

    async def test_message_status_defaults_to_ok(self):
        conv = await self._mk_conversation()
        async with self.session_maker() as s:
            msg = Message(conversation_id=conv.id, role="user",
                          content="привет")
            s.add(msg)
            await s.commit()
            await s.refresh(msg)
            self.assertEqual(msg.status, "ok")
            self.assertIsNone(msg.error)

    async def test_pending_and_failed_status_stored(self):
        """Assistant-сообщение создаётся 'pending', на ошибке становится
        'failed' с текстом; оба значения должны переживать round-trip."""
        conv = await self._mk_conversation()
        async with self.session_maker() as s:
            msg = Message(conversation_id=conv.id, role="assistant",
                          content="", status="pending")
            s.add(msg)
            await s.commit()

            msg.status = "failed"
            msg.error = "ingestion не ответил"
            msg.content = ""
            await s.commit()
            await s.refresh(msg)

            self.assertEqual(msg.status, "failed")
            self.assertEqual(msg.error, "ingestion не ответил")

    async def test_cascade_from_conversation_removes_messages(self):
        """Удаление conversation → все её messages уходят с ней."""
        conv = await self._mk_conversation()
        async with self.session_maker() as s:
            s.add(Message(conversation_id=conv.id, role="user", content="q"))
            s.add(Message(conversation_id=conv.id, role="assistant",
                          content="a"))
            await s.commit()

        async with self.session_maker() as s:
            fresh = await s.get(Conversation, conv.id)
            await s.delete(fresh)
            await s.commit()

        async with self.session_maker() as s:
            rows = (await s.execute(
                select(Message).where(Message.conversation_id == conv.id))
            ).scalars().all()
            self.assertEqual(rows, [],
                             "CASCADE не сработал: messages остались "
                             "сиротами после удаления conversation")

    async def test_cascade_from_message_removes_sources_usage_feedback(self):
        """Удаление message → уходят sources, usage, feedback."""
        conv = await self._mk_conversation()
        async with self.session_maker() as s:
            msg = Message(conversation_id=conv.id, role="assistant",
                          content="ответ")
            s.add(msg)
            await s.commit()
            await s.refresh(msg)

            s.add(MessageSource(
                message_id=msg.id, chunk_id="hash1",
                document_id=DOC, chunk_index=0,
                filename="doc.pdf", rag_id=RAG, order=1))
            s.add(MessageUsage(
                message_id=msg.id, prompt_tokens=100,
                completion_tokens=50, total_tokens=150, model="qwen3:8b"))
            s.add(MessageFeedback(message_id=msg.id, data={"rating": 5}))
            await s.commit()

            await s.delete(msg)
            await s.commit()

        async with self.session_maker() as s:
            self.assertEqual((await s.execute(
                select(MessageSource))).scalars().all(), [])
            self.assertEqual((await s.execute(
                select(MessageUsage))).scalars().all(), [])
            self.assertEqual((await s.execute(
                select(MessageFeedback))).scalars().all(), [])

    async def test_unique_order_per_message(self):
        """Два source c одинаковым (message_id, order) — конфликт: иначе
        [1] в тексте ответа не сможет однозначно указать на источник."""
        conv = await self._mk_conversation()
        async with self.session_maker() as s:
            msg = Message(conversation_id=conv.id, role="assistant",
                          content="a")
            s.add(msg)
            await s.commit()
            await s.refresh(msg)

            s.add(MessageSource(
                message_id=msg.id, chunk_id="h1", document_id=DOC,
                chunk_index=0, filename="d.pdf", rag_id=RAG, order=1))
            await s.commit()

            s.add(MessageSource(
                message_id=msg.id, chunk_id="h2", document_id=DOC,
                chunk_index=1, filename="d.pdf", rag_id=RAG, order=1))
            with self.assertRaises(IntegrityError):
                await s.commit()

    async def test_unique_feedback_per_message(self):
        """Один feedback на сообщение (message_id UNIQUE)."""
        conv = await self._mk_conversation()
        async with self.session_maker() as s:
            msg = Message(conversation_id=conv.id, role="assistant",
                          content="a")
            s.add(msg)
            await s.commit()
            await s.refresh(msg)

            s.add(MessageFeedback(message_id=msg.id, data={"r": 5}))
            await s.commit()

            s.add(MessageFeedback(message_id=msg.id, data={"r": 1}))
            with self.assertRaises(IntegrityError):
                await s.commit()

    async def test_feedback_data_round_trips_as_dict(self):
        """JSONB/JSON: dict → БД → dict, ключи сохраняются."""
        conv = await self._mk_conversation()
        async with self.session_maker() as s:
            msg = Message(conversation_id=conv.id, role="assistant",
                          content="a")
            s.add(msg)
            await s.commit()
            await s.refresh(msg)

            payload = {"rating": 5, "helpful": True,
                       "tags": ["accurate", "concise"]}
            fb = MessageFeedback(message_id=msg.id, data=payload)
            s.add(fb)
            await s.commit()

        async with self.session_maker() as s:
            fresh = (await s.execute(
                select(MessageFeedback))).scalar_one()
            self.assertEqual(fresh.data, payload)

    async def test_messages_ordered_by_created_at(self):
        """conversation.messages — упорядочены по created_at (relationship
        order_by). Иначе при чтении диалога UI получит перемешанные роли."""
        conv = await self._mk_conversation()
        async with self.session_maker() as s:
            s.add(Message(conversation_id=conv.id, role="user", content="1"))
            await s.commit()
            s.add(Message(conversation_id=conv.id, role="assistant",
                          content="2"))
            await s.commit()
            s.add(Message(conversation_id=conv.id, role="user", content="3"))
            await s.commit()

        async with self.session_maker() as s:
            fresh = await s.get(Conversation, conv.id)
            # Явно тянем relationship через refresh, чтобы не полагаться
            # на lazy-loading в async-контексте.
            await s.refresh(fresh, attribute_names=["messages"])
            contents = [m.content for m in fresh.messages]
            self.assertEqual(contents, ["1", "2", "3"])


if __name__ == "__main__":
    unittest.main()
