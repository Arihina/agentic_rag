from __future__ import annotations

"""Тесты конструкторов SSE-событий.

Snapshot-стиль: правильный формат frame'а (event/data/двойной перенос),
UTF-8 без \\u-эскейпов, id всегда с префиксом resp_.
"""

import json
import unittest
import uuid

from app.sse import events


MID = uuid.UUID("11111111-1111-1111-1111-111111111111")
CID = uuid.UUID("22222222-2222-2222-2222-222222222222")


def _parse(frame: bytes) -> tuple[str, dict]:
    """SSE-frame → (event_type, parsed_data)."""
    text = frame.decode("utf-8")
    assert text.endswith("\n\n"), f"нет двойного \\n\\n в конце: {text!r}"
    lines = text.strip().split("\n")
    event_line, data_line = lines[0], lines[1]
    assert event_line.startswith("event: ")
    assert data_line.startswith("data: ")
    return event_line[len("event: "):], json.loads(data_line[len("data: "):])


class SseFrameFormatTests(unittest.TestCase):

    def test_ends_with_double_newline(self):
        """Двойной \\n\\n — граница SSE-frame. Без него клиент не поймёт,
        что событие завершено, и будет копить следующее в буфере."""
        frame = events.response_in_progress(response_id=MID)
        self.assertTrue(frame.endswith(b"\n\n"))
        # Ровно два перевода в конце, не больше и не меньше.
        self.assertFalse(frame.endswith(b"\n\n\n"))

    def test_utf8_not_escaped(self):
        """Русский текст идёт как есть, не \\u-эскейпами. Экономит
        трафик и не ломается на клиентах, которые ожидают чистый UTF-8."""
        frame = events.output_text_delta(
            response_id=MID, delta="привет, мир")
        text = frame.decode("utf-8")
        self.assertIn("привет, мир", text)
        self.assertNotIn(r"\u043f", text)

    def test_data_is_single_line_json(self):
        """`data:` — ровно одна строка. Многострочный JSON превратился бы
        в два `data:`-пункта на стороне клиента, которые он склеит
        по-своему и получит не то."""
        frame = events.response_completed(
            response_id=MID, model="rag/x",
            prompt_tokens=10, completion_tokens=20)
        text = frame.decode("utf-8")
        data_lines = [l for l in text.split("\n") if l.startswith("data: ")]
        self.assertEqual(len(data_lines), 1)

    def test_heartbeat_is_sse_comment(self):
        """Клиент (EventSource, httpx) должен игнорировать. Начинается с
        `:` — по спеке SSE это comment-line."""
        self.assertTrue(events.HEARTBEAT.startswith(b":"))
        self.assertTrue(events.HEARTBEAT.endswith(b"\n\n"))


class ResponseCreatedTests(unittest.TestCase):

    def test_shape(self):
        frame = events.response_created(
            response_id=MID, model="rag/abc",
            conversation_id=CID)
        event_type, data = _parse(frame)

        self.assertEqual(event_type, "response.created")
        self.assertEqual(data["id"], f"resp_{MID}")
        self.assertEqual(data["object"], "response")
        self.assertEqual(data["status"], "in_progress")
        self.assertEqual(data["model"], "rag/abc")
        self.assertEqual(data["conversation_id"], str(CID))
        self.assertIsInstance(data["created_at"], int)

    def test_id_prefixed_with_resp(self):
        """Инвариант формата id для клиента: всегда `resp_<uuid>`."""
        frame = events.response_created(
            response_id=MID, model="rag/x", conversation_id=CID)
        _, data = _parse(frame)
        self.assertTrue(data["id"].startswith("resp_"))


class ResponseInProgressTests(unittest.TestCase):

    def test_minimal_shape(self):
        frame = events.response_in_progress(response_id=MID)
        event_type, data = _parse(frame)
        self.assertEqual(event_type, "response.in_progress")
        self.assertEqual(data, {
            "id": f"resp_{MID}", "status": "in_progress"})


class OutputTextDeltaTests(unittest.TestCase):

    def test_delta_content(self):
        frame = events.output_text_delta(
            response_id=MID, delta="Ответ по [1] и [2].")
        event_type, data = _parse(frame)
        self.assertEqual(event_type, "response.output_text.delta")
        self.assertEqual(data["id"], f"resp_{MID}")
        self.assertEqual(data["delta"], "Ответ по [1] и [2].")

    def test_delta_with_special_json_chars(self):
        """Кавычки, обратные слэши, переводы строки внутри delta —
        json.dumps должен всё это заэскейпить, чтобы результат остался
        валидным JSON на одной строке."""
        tricky = 'Он сказал: "привет"\nа она: "пока"'
        frame = events.output_text_delta(response_id=MID, delta=tricky)
        event_type, data = _parse(frame)  # если не парсится — тест упадёт
        self.assertEqual(data["delta"], tricky)


class ResponseCompletedTests(unittest.TestCase):

    def test_usage_totals(self):
        frame = events.response_completed(
            response_id=MID, model="rag/x",
            prompt_tokens=100, completion_tokens=50)
        event_type, data = _parse(frame)

        self.assertEqual(event_type, "response.completed")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["usage"], {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        })


class ResponseErrorTests(unittest.TestCase):

    def test_error_with_response_id(self):
        frame = events.response_error(
            response_id=MID, message="ollama недоступен",
            error_type="bad_gateway_error", code="upstream")
        event_type, data = _parse(frame)

        self.assertEqual(event_type, "response.error")
        self.assertEqual(data["id"], f"resp_{MID}")
        self.assertEqual(data["error"]["message"], "ollama недоступен")
        self.assertEqual(data["error"]["type"], "bad_gateway_error")
        self.assertEqual(data["error"]["code"], "upstream")

    def test_error_without_response_id(self):
        """Для ошибок ДО создания pending-сообщения (например,
        InvalidModelForm) response_id ещё не существует — не рисуем."""
        frame = events.response_error(
            response_id=None, message="model не в формате rag/<uuid>",
            error_type="invalid_request_error")
        _, data = _parse(frame)
        self.assertNotIn("id", data)
        self.assertEqual(data["error"]["type"], "invalid_request_error")


if __name__ == "__main__":
    unittest.main()
