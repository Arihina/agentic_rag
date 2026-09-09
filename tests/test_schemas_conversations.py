from __future__ import annotations

"""Схемы Platform-ручек — валидация формы.

Проверяем: обязательные поля, extra='forbid', дефолты, литералы.
"""

import unittest
import uuid
from datetime import datetime, timezone

from pydantic import ValidationError

from app.schemas.conversations import (
    ConversationCreateIn, ConversationListOut, ConversationOut,
    ConversationUpdateIn, MessageOut, MessagesListOut,
)


RAG = uuid.UUID("22222222-2222-2222-2222-222222222222")
CID = uuid.UUID("33333333-3333-3333-3333-333333333333")
MID = uuid.UUID("11111111-1111-1111-1111-111111111111")


class ConversationCreateInTests(unittest.TestCase):

    def test_minimal_valid(self):
        req = ConversationCreateIn(rag_id=RAG)
        self.assertEqual(req.rag_id, RAG)
        self.assertIsNone(req.title)

    def test_with_title(self):
        req = ConversationCreateIn(rag_id=RAG, title="Тестовый чат")
        self.assertEqual(req.title, "Тестовый чат")

    def test_rag_id_required(self):
        with self.assertRaises(ValidationError):
            ConversationCreateIn()  # type: ignore[call-arg]

    def test_rag_id_from_string(self):
        req = ConversationCreateIn(rag_id=str(RAG))  # type: ignore[arg-type]
        self.assertEqual(req.rag_id, RAG)

    def test_rag_id_invalid_string(self):
        with self.assertRaises(ValidationError):
            ConversationCreateIn(rag_id="not-uuid")  # type: ignore[arg-type]

    def test_title_max_length(self):
        with self.assertRaises(ValidationError):
            ConversationCreateIn(rag_id=RAG, title="a" * 501)

    def test_extra_forbidden(self):
        """model / conversation_id / user_id — распространённые опечатки.
        extra='forbid' ловит любые: клиент не должен передавать
        неизвестные поля молча."""
        with self.assertRaises(ValidationError):
            ConversationCreateIn(
                rag_id=RAG, model="rag/x")  # type: ignore[call-arg]


class ConversationUpdateInTests(unittest.TestCase):

    def test_title_required(self):
        """Меняем ТОЛЬКО title. rag_id — нельзя (фиксирован при
        создании). Значит PATCH без title = 400."""
        with self.assertRaises(ValidationError):
            ConversationUpdateIn()  # type: ignore[call-arg]

    def test_empty_title_rejected(self):
        """Пустой title — тоже 400. Хочешь без названия — не PATCH'и,
        оставь как есть."""
        with self.assertRaises(ValidationError):
            ConversationUpdateIn(title="")

    def test_title_max_length(self):
        with self.assertRaises(ValidationError):
            ConversationUpdateIn(title="a" * 501)

    def test_rag_id_not_updatable(self):
        """Инвариант: rag_id — read-only после создания. extra='forbid'
        режет попытку сменить набор в PATCH."""
        with self.assertRaises(ValidationError):
            ConversationUpdateIn(
                title="ok", rag_id=RAG)  # type: ignore[call-arg]


class ConversationOutTests(unittest.TestCase):

    def test_shape(self):
        now = datetime.now(timezone.utc)
        out = ConversationOut(
            id=CID, rag_id=RAG, title="X",
            created_at=now, updated_at=now)
        self.assertEqual(out.id, CID)
        self.assertEqual(out.rag_id, RAG)

    def test_title_can_be_null(self):
        now = datetime.now(timezone.utc)
        out = ConversationOut(
            id=CID, rag_id=RAG, title=None,
            created_at=now, updated_at=now)
        self.assertIsNone(out.title)


class ConversationListOutTests(unittest.TestCase):

    def test_empty_list(self):
        out = ConversationListOut(data=[])
        self.assertEqual(out.data, [])

    def test_wraps_items(self):
        now = datetime.now(timezone.utc)
        item = ConversationOut(
            id=CID, rag_id=RAG, title="X",
            created_at=now, updated_at=now)
        out = ConversationListOut(data=[item])
        self.assertEqual(len(out.data), 1)


class MessageOutTests(unittest.TestCase):

    def test_user_message(self):
        now = datetime.now(timezone.utc)
        msg = MessageOut(
            id=f"msg_{MID}", role="user",
            content="привет", status="ok",
            created_at=now)
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.object, "message")
        self.assertIsNone(msg.error)

    def test_assistant_failed_carries_error(self):
        now = datetime.now(timezone.utc)
        msg = MessageOut(
            id=f"msg_{MID}", role="assistant",
            content="", status="failed",
            error="ollama таймаут",
            created_at=now)
        self.assertEqual(msg.status, "failed")
        self.assertEqual(msg.error, "ollama таймаут")

    def test_role_literal_enforced(self):
        """Регресс: не даём проскочить system/tool/etc. — только user
        и assistant пишутся в диалог, остальное сервисное."""
        now = datetime.now(timezone.utc)
        with self.assertRaises(ValidationError):
            MessageOut(
                id=f"msg_{MID}", role="system",  # type: ignore[arg-type]
                content="x", status="ok",
                created_at=now)

    def test_status_literal_enforced(self):
        now = datetime.now(timezone.utc)
        with self.assertRaises(ValidationError):
            MessageOut(
                id=f"msg_{MID}", role="user",
                content="x", status="in_progress",  # type: ignore[arg-type]
                created_at=now)


class MessagesListOutTests(unittest.TestCase):

    def test_empty(self):
        out = MessagesListOut(data=[])
        self.assertEqual(out.data, [])


if __name__ == "__main__":
    unittest.main()
