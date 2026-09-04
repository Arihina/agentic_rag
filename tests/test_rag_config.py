from __future__ import annotations

"""Тесты резолва конфига набора для одного хода.

Три вещи проверяем изолированно:
- parse_model: только `rag/<uuid>` проходит, всё остальное — InvalidModelForm;
- compose_answer_system_prompt: базовый промпт + опциональный per-set — в
  правильной склейке, никогда не заменяется целиком;
- resolve_rag_for_turn: сочетание валидаций и обращение к FakeIngest —
  правильные исключения на каждый сценарий.
"""

import unittest
import uuid

from app.clients.ingest import IngestError, RagNotFound
from app.config import settings
from app.services.rag_config import (
    InvalidModelForm, ModelDoesNotMatchConversation, RagLookupFailed,
    RagUnavailable, compose_answer_system_prompt, parse_model,
    resolve_rag_for_turn,
)
from tests.support import FakeIngest


USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
RAG = uuid.UUID("22222222-2222-2222-2222-222222222222")
OTHER_RAG = uuid.UUID("33333333-3333-3333-3333-333333333333")


class ParseModelTests(unittest.TestCase):

    def test_valid_lowercase(self):
        self.assertEqual(parse_model(f"rag/{RAG}"), RAG)

    def test_valid_uppercase_uuid(self):
        """UUID регистронезависимый — принимаем и aBcD..., и abcd..."""
        self.assertEqual(parse_model(f"rag/{str(RAG).upper()}"), RAG)

    def test_wrong_prefix_rejected(self):
        """`foo/<uuid>`, `epoz/<uuid>` — не наш формат."""
        for m in (f"foo/{RAG}", f"epoz/{RAG}", f"/rag/{RAG}", f"{RAG}"):
            with self.subTest(m=m):
                with self.assertRaises(InvalidModelForm):
                    parse_model(m)

    def test_not_a_uuid_rejected(self):
        for m in ("rag/", "rag/abc", "rag/12345", "rag/not-a-uuid"):
            with self.subTest(m=m):
                with self.assertRaises(InvalidModelForm):
                    parse_model(m)

    def test_empty_and_none_rejected(self):
        with self.assertRaises(InvalidModelForm):
            parse_model("")
        with self.assertRaises(InvalidModelForm):
            parse_model(None)  # type: ignore[arg-type]

    def test_extra_after_uuid_rejected(self):
        """`rag/<uuid>/foo` — не наш формат; мастер такое режет ещё раньше,
        но валидация тут страхует прямое обращение в обход мастера."""
        with self.assertRaises(InvalidModelForm):
            parse_model(f"rag/{RAG}/foo")


class ComposeAnswerSystemPromptTests(unittest.TestCase):

    def test_no_rag_prompt_returns_base(self):
        result = compose_answer_system_prompt(None)
        self.assertEqual(result, settings.answer_system_prompt)

    def test_empty_string_treated_as_none(self):
        """Пустой prompt в конфиге набора = «нет доп. инструкций», а не
        «замени пустотой»."""
        for empty in ("", "   ", "\n\t"):
            with self.subTest(rag_prompt=repr(empty)):
                result = compose_answer_system_prompt(empty)
                self.assertEqual(result, settings.answer_system_prompt)

    def test_rag_prompt_appended_not_replaces(self):
        """Ключевой инвариант: базовый промпт остаётся, per-set добавляется."""
        rag_prompt = "Ты работаешь с юридическими документами."
        result = compose_answer_system_prompt(rag_prompt)

        self.assertIn(settings.answer_system_prompt, result,
                      "базовый ANSWER_SYSTEM_PROMPT обязан остаться целиком")
        self.assertIn(rag_prompt, result,
                      "per-set prompt обязан появиться в результате")
        # Разделитель — двойной перевод + явный маркер, чтобы LLM видела
        # иерархию: base rules → per-set additions.
        self.assertIn("Дополнительные инструкции для этого набора:", result)

    def test_rag_prompt_stripped(self):
        result = compose_answer_system_prompt("  hello  \n\n")
        self.assertIn("hello", result)
        self.assertNotIn("  hello  ", result)


class ResolveRagForTurnTests(unittest.IsolatedAsyncioTestCase):

    async def test_happy_path_ready(self):
        ingest = FakeIngest()
        ingest.set_rag(RAG, name="Тест", status="ready",
                       prompt="Юр. документы.", temperature=0.5,
                       top_k=8, score_threshold=0.25)

        resolved = await resolve_rag_for_turn(
            f"rag/{RAG}", USER, ingest, conversation_rag_id=RAG)

        self.assertEqual(resolved.rag_id, RAG)
        self.assertEqual(resolved.top_k, 8)
        self.assertEqual(resolved.score_threshold, 0.25)
        self.assertEqual(resolved.answer_temperature, 0.5)
        self.assertIn("Юр. документы.", resolved.answer_system_prompt)
        self.assertIn(settings.answer_system_prompt,
                      resolved.answer_system_prompt)

    async def test_model_mismatch_with_conversation(self):
        """`model="rag/A"` при conversation.rag_id=B → 400 БЕЗ обращения
        к ingestion. Экономим сетевой вызов и не даём таймингом
        различать наличие чужого набора."""
        ingest = FakeIngest()
        ingest.set_rag(RAG, status="ready")  # существует, но не тот
        ingest.set_rag(OTHER_RAG, status="ready")

        with self.assertRaises(ModelDoesNotMatchConversation):
            await resolve_rag_for_turn(
                f"rag/{RAG}", USER, ingest,
                conversation_rag_id=OTHER_RAG)

        # Ingestion не спрашивали.
        self.assertEqual(ingest.get_rag_calls, [])

    async def test_model_ok_without_conversation_check(self):
        """Резолв без conversation (для validation-only сценариев) —
        проверка соответствия conversation не выполняется."""
        ingest = FakeIngest()
        ingest.set_rag(RAG, status="ready")

        resolved = await resolve_rag_for_turn(
            f"rag/{RAG}", USER, ingest, conversation_rag_id=None)

        self.assertEqual(resolved.rag_id, RAG)

    async def test_not_found_propagates(self):
        """404 от ingestion → RagNotFound наружу (API-слой мапит в 404).
        Не превращаем в свой RagLookupFailed — семантика разная."""
        ingest = FakeIngest()  # без set_rag → set вернёт RagNotFound

        with self.assertRaises(RagNotFound):
            await resolve_rag_for_turn(
                f"rag/{RAG}", USER, ingest, conversation_rag_id=RAG)

    async def test_ingestion_error_becomes_lookup_failed(self):
        """5xx / сетевой сбой → RagLookupFailed (API-слой мапит в 502).
        Не 404, потому что набор может существовать — временная ошибка."""
        ingest = FakeIngest()
        ingest.set_error(RAG, IngestError("connection refused"))

        with self.assertRaises(RagLookupFailed):
            await resolve_rag_for_turn(
                f"rag/{RAG}", USER, ingest, conversation_rag_id=RAG)

    async def test_status_ingesting_rejected(self):
        """`ingesting` = ready=0, все документы в processing → отвечать
        не по чему. 409, не 404, — набор реально есть."""
        ingest = FakeIngest()
        ingest.set_rag(RAG, status="ingesting")

        with self.assertRaises(RagUnavailable) as ctx:
            await resolve_rag_for_turn(
                f"rag/{RAG}", USER, ingest, conversation_rag_id=RAG)
        self.assertEqual(ctx.exception.status, "ingesting")

    async def test_status_empty_rejected(self):
        """`empty` = документов вообще нет. Отвечать не по чему."""
        ingest = FakeIngest()
        ingest.set_rag(RAG, status="empty")

        with self.assertRaises(RagUnavailable) as ctx:
            await resolve_rag_for_turn(
                f"rag/{RAG}", USER, ingest, conversation_rag_id=RAG)
        self.assertEqual(ctx.exception.status, "empty")

    async def test_status_failed_rejected(self):
        ingest = FakeIngest()
        ingest.set_rag(RAG, status="failed")

        with self.assertRaises(RagUnavailable) as ctx:
            await resolve_rag_for_turn(
                f"rag/{RAG}", USER, ingest, conversation_rag_id=RAG)
        self.assertEqual(ctx.exception.status, "failed")

    async def test_invalid_model_form_before_ingest(self):
        """Битый model — 400 без обращения к ingestion."""
        ingest = FakeIngest()

        with self.assertRaises(InvalidModelForm):
            await resolve_rag_for_turn(
                "foo/bar", USER, ingest, conversation_rag_id=RAG)
        self.assertEqual(ingest.get_rag_calls, [])


if __name__ == "__main__":
    unittest.main()
