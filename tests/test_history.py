from __future__ import annotations

"""Тесты sliding window истории.

Два уровня:
- **Юниты `_apply_sliding_window`**: pure-python, с fake char-tokenizer.
  Проверяем логику окна изолированно, без БД и без загрузки HF-модели.
- **Интеграция `build_history_for_rewriter`**: через in-memory SQLite,
  проверяем скоупинг, фильтр status='ok' и совместимость с репозиторием.

HF-токенайзер `Qwen/Qwen3-8B` в CI не тянем — он весит десятки МБ и требует
интернета. Проверяется вручную через LIVE-режим (задел на 2.9).
"""

import unittest
import uuid

from app.db import repository as repo
from app.db.session import init_db, make_engine, make_session_maker
from app.services.history import (
    HistoryOverflowError,
    _apply_sliding_window,
    _MESSAGE_OVERHEAD_TOKENS,
    _strip_leading_assistant,
    build_history_for_rewriter,
    count_tokens,
)


USER = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER = uuid.UUID("22222222-2222-2222-2222-222222222222")
RAG = uuid.UUID("33333333-3333-3333-3333-333333333333")


class _CharTokenizer:
    """Простой fake: 1 char = 1 token. Достаточно, чтобы проверить логику
    окна без загрузки HF-модели. `encode` возвращает список — count_tokens
    смотрит на len."""

    def encode(self, text: str) -> list[str]:
        return list(text)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# Юниты: логика окна
# ---------------------------------------------------------------------------

class SlidingWindowLogicTests(unittest.TestCase):

    def setUp(self):
        self.tok = _CharTokenizer()

    def test_empty_history_returns_empty(self):
        self.assertEqual(
            _apply_sliding_window([], self.tok, 100, "truncate"), [])

    def test_within_limit_returns_all(self):
        history = [_msg("user", "abc"), _msg("assistant", "de")]
        # 3+5 + 2+5 = 15 токенов; лимит 100 — влезает.
        self.assertEqual(
            _apply_sliding_window(history, self.tok, 100, "truncate"),
            history)

    def test_overflow_strict_raises(self):
        history = [_msg("user", "a" * 100)]  # ~105 токенов
        with self.assertRaises(HistoryOverflowError):
            _apply_sliding_window(history, self.tok, 50, "strict")

    def test_truncate_drops_oldest_until_target(self):
        """При overflow дропаем oldest пока не спустимся до 65% от лимита.
        Дропаем ПАЧКОЙ, не по одному — окно "прыгает" редко."""
        # 4 сообщения по 45 токенов (40 char + 5 overhead) = 180 всего.
        history = [_msg("user" if i % 2 == 0 else "assistant",
                        f"content-{i}" + "x" * 30)
                   for i in range(4)]
        # 4 × ~45 = 180. Лимит 150, target = 97. Дропнется 1-2 старых
        # сообщения, но пара новых обязана остаться.
        result = _apply_sliding_window(history, self.tok, 150, "truncate")
        self.assertLess(len(result), len(history))
        self.assertGreater(len(result), 0, "не должны опустошать")
        # Оставшееся суммарно <= target (97).
        remaining_tokens = sum(
            len(m["content"]) + _MESSAGE_OVERHEAD_TOKENS for m in result)
        self.assertLessEqual(remaining_tokens, 97)

    def test_truncate_keeps_newest(self):
        """При дропе выкидываются старые, не новые. Иначе rewriter
        потеряет самый релевантный контекст (последний обмен) — тот, к
        которому пользователь только что обратился."""
        history = [
            _msg("user", "old-question"),
            _msg("assistant", "old-answer"),
            _msg("user", "middle-question"),
            _msg("assistant", "middle-answer"),
            _msg("user", "new-question"),
            _msg("assistant", "new-answer"),
        ]
        # Лимит с запасом на последнюю пару (17+15=32 + overhead 10 = 42),
        # но не на всю историю (сумма ≈ 102). target=65*0.65=42, ровно на
        # пару последних.
        result = _apply_sliding_window(history, self.tok, 65, "truncate")
        self.assertTrue(any("new" in m["content"] for m in result),
                        "самое свежее сообщение должно остаться")
        self.assertFalse(any("old" in m["content"] for m in result),
                         "самые старые сообщения должны быть дропнуты")

    def test_leading_assistant_stripped(self):
        """Если после truncate осталось начиная с assistant — он выкидывается.
        Иначе rewriter трактует его как system-подсказку."""
        history = [
            _msg("user", "стартовый вопрос " + "x" * 50),
            _msg("assistant", "answer-1"),  # должен остаться в паре
            _msg("user", "next"),
        ]
        # Лимит впритык, чтобы user-1 не влез, но assistant-1 остался.
        result = _apply_sliding_window(history, self.tok, 30, "truncate")
        if result:
            self.assertNotEqual(result[0]["role"], "assistant",
                                "orphan assistant в начале сломает rewriter")

    def test_strip_leading_assistant_helper(self):
        self.assertEqual(_strip_leading_assistant([]), [])
        # Один assistant → пусто.
        self.assertEqual(
            _strip_leading_assistant([_msg("assistant", "a")]), [])
        # Начинается с user → без изменений.
        h = [_msg("user", "q"), _msg("assistant", "a")]
        self.assertEqual(_strip_leading_assistant(list(h)), h)
        # Начинается с двух assistant подряд → оба уходят.
        h = [_msg("assistant", "a1"), _msg("assistant", "a2"),
             _msg("user", "q")]
        self.assertEqual(
            _strip_leading_assistant(h),
            [_msg("user", "q")])

    def test_target_ratio_reasonable(self):
        """Регрессия: если случайно поменяют _DROP_TARGET_RATIO на что-то
        близкое к 1 (например, 0.95), окно будет сдвигаться каждый ход —
        поломается вся идея сокращать частоту KV-cache перепрефилла."""
        from app.services.history import _DROP_TARGET_RATIO
        self.assertLess(_DROP_TARGET_RATIO, 0.8,
                        "target > 80% лимита = дроп по одному сообщению — "
                        "плохо для KV-cache")
        self.assertGreater(_DROP_TARGET_RATIO, 0.5,
                           "слишком агрессивный дроп теряет контекст")

    def test_never_leaves_history_empty_when_overflow(self):
        """Один-единственный огромный message — оставляем как есть, а не
        возвращаем []. Не дропаем последнее, даже если оно превышает
        лимит; иначе не с чего будет строить контекст."""
        history = [_msg("user", "x" * 1000)]
        result = _apply_sliding_window(history, self.tok, 50, "truncate")
        self.assertEqual(len(result), 1)


class CountTokensTests(unittest.TestCase):

    def test_count_via_encode_len(self):
        tok = _CharTokenizer()
        self.assertEqual(count_tokens("hello", tok), 5)
        self.assertEqual(count_tokens("", tok), 0)


class TokenizerCacheTests(unittest.TestCase):

    def test_get_tokenizer_is_cached(self):
        """lru_cache гарантирует, что одна и та же строка возвращает тот
        же инстанс — иначе каждый запрос грузил бы токенайзер заново."""
        from app.services import history

        # Подменяем реальный import ленивый — считаем вызовы. Простая
        # проверка: при повторном вызове с тем же repo — тот же объект.
        real_from_pretrained = None
        try:
            from tokenizers import Tokenizer
            real_from_pretrained = Tokenizer.from_pretrained
        except ImportError:
            self.skipTest("tokenizers не установлен")

        calls = {"n": 0}

        def fake_from_pretrained(repo):
            calls["n"] += 1
            return _CharTokenizer()

        try:
            Tokenizer.from_pretrained = fake_from_pretrained
            history.get_tokenizer.cache_clear()
            a = history.get_tokenizer("fake/repo")
            b = history.get_tokenizer("fake/repo")
            self.assertIs(a, b)
            self.assertEqual(calls["n"], 1,
                             "второе обращение к get_tokenizer должно "
                             "быть кешировано, не грузить заново")
        finally:
            Tokenizer.from_pretrained = real_from_pretrained
            history.get_tokenizer.cache_clear()


# ---------------------------------------------------------------------------
# Интеграция с БД
# ---------------------------------------------------------------------------

class BuildHistoryFromDbTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.engine = make_engine("sqlite+aiosqlite:///:memory:")
        async with self.engine.begin() as conn:
            from sqlalchemy import text
            await conn.execute(text("PRAGMA foreign_keys=ON"))
        await init_db(self.engine)
        self.session_maker = make_session_maker(self.engine)
        self.tok = _CharTokenizer()

    async def asyncTearDown(self):
        await self.engine.dispose()

    async def _mk_conv(self, user=USER) -> uuid.UUID:
        async with self.session_maker() as s:
            conv = await repo.create_conversation(s, user_id=user, rag_id=RAG)
            await s.commit()
            return conv.id

    async def test_empty_conversation_returns_empty(self):
        conv_id = await self._mk_conv()
        async with self.session_maker() as s:
            history = await build_history_for_rewriter(
                s, conv_id, USER, tokenizer=self.tok, token_limit=1000)
        self.assertEqual(history, [])

    async def test_full_conversation_within_limit(self):
        conv_id = await self._mk_conv()
        async with self.session_maker() as s:
            await repo.add_user_message(s, conv_id, "первый вопрос")
            msg = await repo.add_pending_assistant_message(s, conv_id)
            await s.commit()
            await repo.mark_message_ok(s, msg.id, "первый ответ")
            await s.commit()

        async with self.session_maker() as s:
            history = await build_history_for_rewriter(
                s, conv_id, USER, tokenizer=self.tok, token_limit=1000)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user",
                                      "content": "первый вопрос"})
        self.assertEqual(history[1], {"role": "assistant",
                                      "content": "первый ответ"})

    async def test_failed_messages_excluded(self):
        """Инвариант: failed не попадают в rewriter. Иначе пользователь
        видит, что «ход упал», а rewriter учитывает контекст, которого
        UI никогда не показал."""
        conv_id = await self._mk_conv()
        async with self.session_maker() as s:
            await repo.add_user_message(s, conv_id, "q1")
            m1 = await repo.add_pending_assistant_message(s, conv_id)
            await s.commit()
            await repo.mark_message_ok(s, m1.id, "a1")
            await s.commit()

            await repo.add_user_message(s, conv_id, "q2")
            m2 = await repo.add_pending_assistant_message(s, conv_id)
            await s.commit()
            await repo.mark_message_failed(s, m2.id, "ollama таймаут")
            await s.commit()

        async with self.session_maker() as s:
            history = await build_history_for_rewriter(
                s, conv_id, USER, tokenizer=self.tok, token_limit=1000)

        contents = [m["content"] for m in history]
        # q1, a1, q2 — есть. q2 остаётся, потому что status=ok (это же
        # исходный вопрос пользователя, он не failed). Failed — только
        # m2 (assistant, к нему привязана ошибка).
        self.assertIn("q1", contents)
        self.assertIn("a1", contents)
        self.assertIn("q2", contents)
        # Проверяем, что failed-content не всплывает — mark_message_failed
        # его чистит, но даже если бы оставил, статус !='ok' его отсеет.
        for msg in history:
            if msg["role"] == "assistant":
                self.assertNotEqual(msg["content"], "",
                                    "должны отфильтроваться пустые failed")

    async def test_scoping_by_user(self):
        """Чужой чат — NotFoundOrForbidden от repository.get_conversation."""
        conv_id = await self._mk_conv(user=OTHER)
        from app.db.repository import NotFoundOrForbidden
        async with self.session_maker() as s:
            with self.assertRaises(NotFoundOrForbidden):
                await build_history_for_rewriter(
                    s, conv_id, USER, tokenizer=self.tok, token_limit=1000)

    async def test_overflow_truncate_used_by_default(self):
        """Дефолт truncate — при overflow не бросает, а обрезает."""
        conv_id = await self._mk_conv()
        async with self.session_maker() as s:
            for i in range(6):
                await repo.add_user_message(
                    s, conv_id, f"q{i} " + "x" * 30)
                m = await repo.add_pending_assistant_message(s, conv_id)
                await s.commit()
                await repo.mark_message_ok(s, m.id, f"a{i} " + "y" * 30)
                await s.commit()

        async with self.session_maker() as s:
            history = await build_history_for_rewriter(
                s, conv_id, USER, tokenizer=self.tok,
                token_limit=150, overflow="truncate")

        self.assertLess(len(history), 12,
                        "12 сообщений по ~35 токенов не должны все влезть "
                        "в лимит 150")
        self.assertGreater(len(history), 0,
                           "хоть что-то должно остаться")


if __name__ == "__main__":
    unittest.main()
