from __future__ import annotations

"""Схемы ResponsesRequest / ResponseObject.

Проверяем: обязательные поля, extra='forbid' на входе, дефолты, форма
OpenAI-совместимого output.
"""

import unittest
import uuid

from pydantic import ValidationError

from app.schemas.responses import (
    OutputMessage, OutputTextItem, ResponseObject, ResponsesRequest, UsageOut,
)


CID = uuid.UUID("22222222-2222-2222-2222-222222222222")
MID = uuid.UUID("11111111-1111-1111-1111-111111111111")


class ResponsesRequestTests(unittest.TestCase):

    def test_minimal_valid(self):
        req = ResponsesRequest(
            model=f"rag/{CID}", input="привет",
            conversation_id=CID)
        self.assertEqual(req.model, f"rag/{CID}")
        self.assertEqual(req.input, "привет")
        self.assertEqual(req.conversation_id, CID)
        self.assertTrue(req.stream, "у нас stream=True по умолчанию")

    def test_model_required(self):
        with self.assertRaises(ValidationError):
            ResponsesRequest(input="x", conversation_id=CID)  # type: ignore

    def test_input_required(self):
        with self.assertRaises(ValidationError):
            # type: ignore
            ResponsesRequest(model="rag/x", conversation_id=CID)

    def test_conversation_id_required(self):
        with self.assertRaises(ValidationError):
            ResponsesRequest(model="rag/x", input="q")  # type: ignore

    def test_empty_input_rejected(self):
        """Пустой input — 400. Не даём агентскому циклу запуститься на
        пустом запросе (rewriter/multi_query смогут, но результат будет
        бесполезный + расход бюджета LLM)."""
        with self.assertRaises(ValidationError):
            ResponsesRequest(
                model=f"rag/{CID}", input="", conversation_id=CID)

    def test_empty_model_rejected(self):
        with self.assertRaises(ValidationError):
            ResponsesRequest(
                model="", input="q", conversation_id=CID)

    def test_extra_field_rejected(self):
        """`extra='forbid'`: клиент с опечаткой не должен молчаливо
        получить дефолт вместо того, что задумывал послать."""
        with self.assertRaises(ValidationError):
            ResponsesRequest(
                model=f"rag/{CID}", input="q",
                conversation_id=CID, temerature=0.5)  # опечатка

    def test_conversation_id_string_coerced(self):
        """UUID приходит из JSON как строка, pydantic конвертит."""
        req = ResponsesRequest(
            model=f"rag/{CID}", input="q",
            conversation_id=str(CID))  # type: ignore[arg-type]
        self.assertEqual(req.conversation_id, CID)

    def test_conversation_id_invalid_string(self):
        with self.assertRaises(ValidationError):
            ResponsesRequest(
                model=f"rag/{CID}", input="q",
                conversation_id="not-a-uuid")  # type: ignore[arg-type]

    def test_stream_can_be_false(self):
        """Схема допускает stream=false; отказ (501/400) уже уровнем
        API-слоя — сейчас MVP не поддерживает non-stream, но контракт
        совместим."""
        req = ResponsesRequest(
            model=f"rag/{CID}", input="q",
            conversation_id=CID, stream=False)
        self.assertFalse(req.stream)


class OutputMessageTests(unittest.TestCase):

    def test_shape(self):
        msg = OutputMessage(
            id=f"msg_{MID}",
            content=[OutputTextItem(text="Ответ по [1].")],
        )
        self.assertEqual(msg.type, "message")
        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.status, "completed")
        self.assertEqual(msg.content[0].type, "output_text")

    def test_type_literal_enforced(self):
        with self.assertRaises(ValidationError):
            OutputMessage(
                id="msg_x", type="tool_call",  # type: ignore
                content=[OutputTextItem(text="x")])

    def test_role_literal_enforced(self):
        with self.assertRaises(ValidationError):
            OutputMessage(
                id="msg_x", role="user",  # type: ignore
                content=[OutputTextItem(text="x")])


class ResponseObjectTests(unittest.TestCase):

    def _valid(self, **overrides) -> ResponseObject:
        defaults = {
            "id": f"resp_{MID}",
            "created_at": 1700000000,
            "status": "completed",
            "model": f"rag/{CID}",
            "conversation_id": CID,
            "output": [OutputMessage(
                id=f"msg_{MID}",
                content=[OutputTextItem(text="ответ")])],
            "usage": UsageOut(
                prompt_tokens=100, completion_tokens=50, total_tokens=150),
        }
        defaults.update(overrides)
        return ResponseObject(**defaults)

    def test_happy_completed(self):
        r = self._valid()
        self.assertEqual(r.object, "response")
        self.assertEqual(r.status, "completed")
        self.assertEqual(r.output[0].content[0].text, "ответ")
        self.assertEqual(r.usage.total_tokens, 150)
        self.assertIsNone(r.error)

    def test_in_progress_without_usage(self):
        """Для in_progress usage может быть None — токены ещё не
        подсчитаны."""
        r = self._valid(status="in_progress", usage=None, output=[])
        self.assertIsNone(r.usage)
        self.assertEqual(r.output, [])

    def test_failed_carries_error(self):
        r = self._valid(
            status="failed", output=[], usage=None,
            error={"message": "ollama таймаут",
                   "type": "timeout_error", "code": None})
        self.assertEqual(r.status, "failed")
        self.assertEqual(r.error["message"], "ollama таймаут")

    def test_status_literal_enforced(self):
        with self.assertRaises(ValidationError):
            self._valid(status="pending")  # у нас snapshot не отдаёт pending


if __name__ == "__main__":
    unittest.main()
