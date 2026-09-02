from __future__ import annotations

"""Тесты run_agent — сборка агентского цикла на fakes.

Проверяем поведение стоп-условий и обработку истории. Один канонический
happy path + отдельные тесты на каждую из трёх причин остановки.
"""

import unittest

from app.core.agent import run_agent
from app.core.answer import GeneratedAnswer
from app.core.evaluation import EvalResult
from tests.core_fakes import FakeEmbed, FakeLLM, FakeOpenSearch, hit


class RunAgentTests(unittest.IsolatedAsyncioTestCase):

    async def test_happy_path_first_iteration_sufficient(self):
        """Eval сразу говорит sufficient — второй итерации не должно быть."""
        llm = FakeLLM(
            evaluator=lambda p, c: EvalResult(sufficient=True, reasoning="ok"),
            answerer=lambda p, c: GeneratedAnswer(answer="готово",
                                                  grounded=True))
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a"), hit("b")]])

        trace = await run_agent(
            os_client, llm, embed, "как настроить X", top_k=5)

        self.assertEqual(trace.stopped_reason, "sufficient")
        self.assertEqual(len(trace.iterations), 1)
        self.assertEqual(len(trace.final_chunks), 2)
        self.assertEqual(trace.answer.answer, "готово")

    async def test_max_iterations_when_never_sufficient(self):
        """Eval всегда говорит insufficient — доходим до max_iterations."""
        llm = FakeLLM(
            evaluator=lambda p, c: EvalResult(
                sufficient=False, next_queries=["nq"]),
            answerer=lambda p, c: GeneratedAnswer(answer="частично",
                                                  grounded=False))
        embed = FakeEmbed()
        # На итерации 1 три запроса (rewritten + 2 варианта) × BM25+kNN = 6
        # search-вызовов; на итерациях 2 и 3 — по одному запросу из
        # next_queries × 2 = 2 вызова. Между итерациями возвращаем разные
        # документы, чтобы overlap не достиг порога diminishing_returns.
        it1 = [hit(f"iter1_{i}") for i in range(3)]
        it2 = [hit(f"iter2_{i}") for i in range(3)]
        it3 = [hit(f"iter3_{i}") for i in range(3)]
        os_client = FakeOpenSearch(
            responses=[it1] * 6 + [it2] * 2 + [it3] * 2)

        trace = await run_agent(
            os_client, llm, embed, "как настроить X",
            top_k=5, max_iterations=3)

        self.assertEqual(trace.stopped_reason, "max_iterations")
        self.assertEqual(len(trace.iterations), 3)

    async def test_diminishing_returns_stops_early(self):
        """Если новая итерация приносит те же документы (overlap >= 0.8),
        останавливаемся, не тратя оставшиеся итерации."""
        llm = FakeLLM(
            evaluator=lambda p, c: EvalResult(
                sufficient=False, next_queries=["nq"]),
            answerer=lambda p, c: GeneratedAnswer(answer="ok", grounded=True))
        embed = FakeEmbed()
        # На всех итерациях возвращаем одни и те же три документа —
        # overlap на второй итерации будет 1.0.
        common = [hit("x"), hit("y"), hit("z")]
        os_client = FakeOpenSearch(responses=[common])

        trace = await run_agent(
            os_client, llm, embed, "как настроить X",
            top_k=5, max_iterations=3)

        self.assertEqual(trace.stopped_reason, "diminishing_returns")
        self.assertEqual(len(trace.iterations), 2)

    async def test_empty_history_skips_rewriter(self):
        """При пустой истории rewriter не должен вызываться — экономия."""
        llm = FakeLLM(
            evaluator=lambda p, c: EvalResult(sufficient=True),
            answerer=lambda p, c: GeneratedAnswer(answer="ok", grounded=True))
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])

        trace = await run_agent(
            os_client, llm, embed, "первый вопрос", history=None)

        # В call_log fake-LLM должно быть: QueryVariants, EvalResult,
        # GeneratedAnswer — но НЕ RewrittenQuery.
        called_models = [name for name, _ in llm.call_log]
        self.assertNotIn("RewrittenQuery", called_models)
        # А rewritten в trace — сам исходный запрос, без правок.
        self.assertEqual(trace.rewritten_query, "первый вопрос")

    async def test_history_triggers_rewriter(self):
        llm = FakeLLM(
            rewriter=lambda p: "переписанный запрос",
            evaluator=lambda p, c: EvalResult(sufficient=True),
            answerer=lambda p, c: GeneratedAnswer(answer="ok", grounded=True))
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])

        trace = await run_agent(
            os_client, llm, embed, "а он тоже так делает?",
            history=[{"role": "user", "content": "расскажи про X"},
                     {"role": "assistant", "content": "X делает то-то"}])

        self.assertEqual(trace.rewritten_query, "переписанный запрос")

    async def test_no_chunks_returns_ungrounded_answer(self):
        """OpenSearch не нашёл ничего за все итерации — answer помечен
        grounded=false, а не крэш."""
        llm = FakeLLM(
            evaluator=lambda p, c: EvalResult(
                sufficient=False, next_queries=["nq"]))
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[]])

        trace = await run_agent(
            os_client, llm, embed, "невозможный запрос",
            top_k=5, max_iterations=2)

        self.assertEqual(trace.final_chunks, [])
        self.assertFalse(trace.answer.grounded)


if __name__ == "__main__":
    unittest.main()
