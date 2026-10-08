from __future__ import annotations

"""Тесты парсинга структурированных ответов Ollama.

Покрывают две известные причуды format="json":
  - обёртка {"properties": {...}} вокруг полезной нагрузки;
  - пустой ответ (EOF) → повтор внутри retry-бюджета.
"""

import unittest
from unittest import mock

from pydantic import BaseModel

from app.clients.llm import (
    LLMClient, LLMError, _parse_structured, _unwrap_properties)
from app.config import settings
from app.core.answer import GeneratedAnswer


class UnwrapPropertiesTests(unittest.TestCase):

    def test_unwrap_plain_wrapper(self):
        data = {"properties": {"answer": "x", "grounded": True}}
        self.assertEqual(
            _unwrap_properties(data, GeneratedAnswer),
            {"answer": "x", "grounded": True})

    def test_plain_payload_untouched(self):
        data = {"answer": "x", "grounded": True}
        self.assertEqual(_unwrap_properties(data, GeneratedAnswer), data)

    def test_non_dict_untouched(self):
        self.assertEqual(
            _unwrap_properties([1, 2], GeneratedAnswer), [1, 2])

    def test_does_not_strip_legit_properties_field(self):
        class WithProps(BaseModel):
            properties: dict

        data = {"properties": {"a": 1}}
        self.assertEqual(_unwrap_properties(data, WithProps), data)


class ParseStructuredTests(unittest.TestCase):

    def test_wrapped(self):
        raw = '{"properties": {"answer": "ok", "grounded": true}}'
        res = _parse_structured(raw, GeneratedAnswer)
        self.assertEqual(res.answer, "ok")
        self.assertTrue(res.grounded)

    def test_plain(self):
        raw = '{"answer": "ok", "grounded": false}'
        res = _parse_structured(raw, GeneratedAnswer)
        self.assertEqual(res.answer, "ok")
        self.assertFalse(res.grounded)

    def test_empty_raises_value_error(self):
        with self.assertRaises(ValueError):
            _parse_structured("   ", GeneratedAnswer)

    def test_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            _parse_structured("not json", GeneratedAnswer)


class _ScriptedLLM(LLMClient):
    """Отдаёт заготовленные сырые ответы по одному за вызов."""

    def __init__(self, responses: list[str]):
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    async def generate(self, **kwargs) -> str:
        self.calls += 1
        return self._responses.pop(0)


class GenerateStructuredRetryTests(unittest.IsolatedAsyncioTestCase):

    async def test_retries_empty_then_wrapped(self):
        llm = _ScriptedLLM([
            "",
            '{"properties": {"answer": "готово", "grounded": true}}',
        ])
        with mock.patch.object(settings, "llm_retry_backoff", 0.0):
            res = await llm.generate_structured(
                model="m", system="s", prompt="p",
                response_model=GeneratedAnswer, temperature=0.0)

        self.assertEqual(res.answer, "готово")
        self.assertTrue(res.grounded)
        self.assertEqual(llm.calls, 2)

    async def test_raises_after_exhausting_retries(self):
        llm = _ScriptedLLM(["", "", ""])
        with mock.patch.object(settings, "llm_retry_backoff", 0.0):
            with self.assertRaises(LLMError):
                await llm.generate_structured(
                    model="m", system="s", prompt="p",
                    response_model=GeneratedAnswer, temperature=0.0)

        self.assertEqual(llm.calls, settings.llm_retry_attempts)


if __name__ == "__main__":
    unittest.main()
