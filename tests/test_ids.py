from __future__ import annotations

"""parse_message_id / format_message_id.

Три валидные формы на входе, всегда `resp_<uuid>` на выходе. Регистр
префикса и UUID неважен.
"""

import unittest
import uuid

from app.api.ids import (
    InvalidMessageId, format_message_id, format_platform_message_id,
    parse_message_id,
)


MID = uuid.UUID("11111111-1111-1111-1111-111111111111")


class ParseMessageIdTests(unittest.TestCase):

    def test_resp_prefix(self):
        self.assertEqual(parse_message_id(f"resp_{MID}"), MID)

    def test_chatcmpl_prefix(self):
        self.assertEqual(parse_message_id(f"chatcmpl-{MID}"), MID)

    def test_msg_prefix(self):
        """Platform listings отдают msg_<uuid>; парсер должен принимать
        и его — клиент может сохранить id именно из listing'а и потом
        обратиться в Responses API GET."""
        self.assertEqual(parse_message_id(f"msg_{MID}"), MID)

    def test_bare_uuid(self):
        self.assertEqual(parse_message_id(str(MID)), MID)

    def test_uppercase_uuid(self):
        """UUID регистронезависимый."""
        self.assertEqual(parse_message_id(str(MID).upper()), MID)
        self.assertEqual(parse_message_id(f"resp_{str(MID).upper()}"), MID)

    def test_prefix_case_insensitive(self):
        """Префикс тоже — принимаем Resp_, RESP_, ChatCmpl-, Msg_."""
        self.assertEqual(parse_message_id(f"Resp_{MID}"), MID)
        self.assertEqual(parse_message_id(f"RESP_{MID}"), MID)
        self.assertEqual(parse_message_id(f"CHATCMPL-{MID}"), MID)
        self.assertEqual(parse_message_id(f"MSG_{MID}"), MID)

    def test_unknown_prefix_rejected(self):
        for raw in (f"foo_{MID}", f"resp-{MID}",
                    f"chatcmpl_{MID}", f"msg-{MID}"):
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidMessageId):
                    parse_message_id(raw)

    def test_not_a_uuid_rejected(self):
        for raw in ("resp_", "chatcmpl-", "msg_", "resp_abc",
                    "resp_12345", "not-a-uuid", "resp_11111111"):
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidMessageId):
                    parse_message_id(raw)

    def test_empty_and_none_rejected(self):
        with self.assertRaises(InvalidMessageId):
            parse_message_id("")
        with self.assertRaises(InvalidMessageId):
            parse_message_id(None)  # type: ignore[arg-type]

    def test_trailing_garbage_rejected(self):
        """`resp_<uuid>_extra` — не наш формат. Точный match, не substring."""
        with self.assertRaises(InvalidMessageId):
            parse_message_id(f"resp_{MID}extra")
        with self.assertRaises(InvalidMessageId):
            parse_message_id(f"msg_{MID}/foo")


class FormatMessageIdTests(unittest.TestCase):

    def test_format_message_id_gives_resp(self):
        """Responses API формат: `resp_<uuid>`."""
        self.assertEqual(format_message_id(MID), f"resp_{MID}")

    def test_format_platform_message_id_gives_msg(self):
        """Platform listings формат: `msg_<uuid>`. Тот же UUID, другой
        префикс — семантика endpoint'а. Регрессия: если случайно
        поменяют на `resp_`, оба форматера станут делать одно, и клиент
        не различит контексты."""
        self.assertEqual(format_platform_message_id(MID), f"msg_{MID}")

    def test_round_trip_from_any_prefix_to_resp(self):
        """Четыре формы на входе → `resp_<uuid>` на выходе через
        Responses-форматер."""
        for raw in (f"resp_{MID}", f"chatcmpl-{MID}",
                    f"msg_{MID}", str(MID)):
            self.assertEqual(
                format_message_id(parse_message_id(raw)),
                f"resp_{MID}")

    def test_round_trip_from_any_prefix_to_msg(self):
        """То же через Platform-форматер → `msg_<uuid>`."""
        for raw in (f"resp_{MID}", f"chatcmpl-{MID}",
                    f"msg_{MID}", str(MID)):
            self.assertEqual(
                format_platform_message_id(parse_message_id(raw)),
                f"msg_{MID}")


if __name__ == "__main__":
    unittest.main()
