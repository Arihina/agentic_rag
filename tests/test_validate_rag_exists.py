from __future__ import annotations

"""Тесты validate_rag_exists (2.5.a).

Ключевое отличие от resolve_rag_for_turn: НЕ требует status='ready'.
Пользователь должен иметь возможность создать чат на пустой / загружающийся
набор — «зарезервировать» диалог до того, как ingestion загрузит первые
документы.
"""

import unittest
import uuid

from app.clients.ingest import IngestError, RagNotFound
from app.services.rag_config import (
    RagLookupFailed, validate_rag_exists,
)
from tests.support import FakeIngest


USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
RAG = uuid.UUID("22222222-2222-2222-2222-222222222222")


class ValidateRagExistsTests(unittest.IsolatedAsyncioTestCase):

    async def test_ready_status_accepted(self):
        ingest = FakeIngest()
        ingest.set_rag(RAG, status="ready")
        cfg = await validate_rag_exists(RAG, USER, ingest)
        self.assertEqual(cfg.id, RAG)
        self.assertEqual(cfg.status, "ready")

    async def test_empty_status_accepted(self):
        """Ключевой инвариант 2.5.a: пустой набор — валидный для
        создания чата. Отвечать всё равно не даст POST /v1/responses,
        но UX не должен блокировать заведение диалога.

        Регрессия: если случайно скопируют логику resolve_rag_for_turn
        и добавят проверку `status='ready'` — этот тест падает."""
        ingest = FakeIngest()
        ingest.set_rag(RAG, status="empty")
        cfg = await validate_rag_exists(RAG, USER, ingest)
        self.assertEqual(cfg.status, "empty")

    async def test_ingesting_status_accepted(self):
        """Ingestion в процессе — можно заводить чат заранее."""
        ingest = FakeIngest()
        ingest.set_rag(RAG, status="ingesting")
        cfg = await validate_rag_exists(RAG, USER, ingest)
        self.assertEqual(cfg.status, "ingesting")

    async def test_failed_status_accepted(self):
        """Даже failed набор — не блокируем создание. Пользователь
        может решить перезалить документы и продолжить чат."""
        ingest = FakeIngest()
        ingest.set_rag(RAG, status="failed")
        cfg = await validate_rag_exists(RAG, USER, ingest)
        self.assertEqual(cfg.status, "failed")

    async def test_not_found_propagates(self):
        """RagNotFound (набор не существует ИЛИ чужой) — API мапит в 404."""
        ingest = FakeIngest()  # без set_rag
        with self.assertRaises(RagNotFound):
            await validate_rag_exists(RAG, USER, ingest)

    async def test_lookup_error_becomes_lookup_failed(self):
        """Сетевой сбой ingestion → RagLookupFailed (API мапит в 502)."""
        ingest = FakeIngest()
        ingest.set_error(RAG, IngestError("connection refused"))
        with self.assertRaises(RagLookupFailed):
            await validate_rag_exists(RAG, USER, ingest)

    async def test_returns_config_for_caller(self):
        """Возвращаем RagConfig — вызывающий может использовать cfg.name,
        например, для дефолтного title чата."""
        ingest = FakeIngest()
        ingest.set_rag(RAG, name="База знаний по продукту", status="ready")
        cfg = await validate_rag_exists(RAG, USER, ingest)
        self.assertEqual(cfg.name, "База знаний по продукту")


if __name__ == "__main__":
    unittest.main()
