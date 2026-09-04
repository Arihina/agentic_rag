from __future__ import annotations

"""parse_message_id / format_message_id.

Три валидные формы на входе, всегда `resp_<uuid>` на выходе. Регистр
префикса и UUID неважен.
"""

import unittest
import uuid

from app.api.ids import InvalidMessageId, format_message_id, parse_message_id


MID = uuid.UUID("11111111-1111-1111-1111-111111111111")


class ParseMessageIdTests(unittest.TestCase):

    def test_resp_prefix(self):
        self.assertEqual(parse_message_id(f"resp_{MID}"), MID)

    def test_chatcmpl_prefix(self):
        self.assertEqual(parse_message_id(f"chatcmpl-{MID}"), MID)

    def test_bare_uuid(self):
        self.assertEqual(parse_message_id(str(MID)), MID)

    def test_uppercase_uuid(self):
        """UUID регистронезависимый."""
        self.assertEqual(parse_message_id(str(MID).upper()), MID)
        self.assertEqual(parse_message_id(f"resp_{str(MID).upper()}"), MID)

    def test_prefix_case_insensitive(self):
        """Префикс тоже — принимаем Resp_, RESP_, ChatCmpl-."""
        self.assertEqual(parse_message_id(f"Resp_{MID}"), MID)
        self.assertEqual(parse_message_id(f"RESP_{MID}"), MID)
        self.assertEqual(parse_message_id(f"CHATCMPL-{MID}"), MID)

    def test_unknown_prefix_rejected(self):
        for raw in (f"foo_{MID}", f"msg_{MID}", f"resp-{MID}",
                    f"chatcmpl_{MID}"):
            with self.subTest(raw=raw):
                with self.assertRaises(InvalidMessageId):
                    parse_message_id(raw)

    def test_not_a_uuid_rejected(self):
        for raw in ("resp_", "chatcmpl-", "resp_abc",
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
            parse_message_id(f"resp_{MID}/foo")


class FormatMessageIdTests(unittest.TestCase):

    def test_always_resp_prefix(self):
        """Инвариант: наружу — всегда resp_. Регресс: если кто-то захочет
        сделать format_message_id 'умным' (например, возвращать тот
        префикс, с которым парсили), это тестом заваливается."""
        self.assertEqual(format_message_id(MID), f"resp_{MID}")

    def test_round_trip_normalizes(self):
        """Три формы на входе → одна на выходе."""
        for raw in (f"resp_{MID}", f"chatcmpl-{MID}", str(MID)):
            self.assertEqual(
                format_message_id(parse_message_id(raw)),
                f"resp_{MID}")


if __name__ == "__main__":
    unittest.main()
