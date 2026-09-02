from __future__ import annotations

"""Тесты multi_query_hybrid_search.

Ядро проверяемого поведения:
- батч-эмбеддинг: /embed зовётся один раз со всем списком queries,
  а НЕ по разу на каждый вариант;
- OpenSearch зовётся 2N раз (BM25+kNN на каждый вариант);
- primary_weight=2.0 применяется к rewritten (queries[0]),
  остальные варианты идут с весом 1.0.
"""

import asyncio
import unittest

from app.search.hybrid import multi_query_hybrid_search
from tests.core_fakes import FakeEmbed, FakeOpenSearch, hit


class MultiQueryHybridSearchTests(unittest.IsolatedAsyncioTestCase):

    async def test_embed_called_once_with_all_queries(self):
        """Инвариант batch-embed: один вызов /embed на всю multi-query.
        Если реализация случайно вернётся к «embed в цикле», sparse-вектора
        разъедутся против того, что писала ingestion."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            os_client, embed, ["main", "v1", "v2"], final_top_k=5)
        self.assertEqual(len(embed.call_log), 1)
        self.assertEqual(embed.call_log[0], ["main", "v1", "v2"])

    async def test_opensearch_called_twice_per_query(self):
        """BM25 + kNN на каждую формулировку. 3 запроса → 6 вызовов search."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            os_client, embed, ["q1", "q2", "q3"], final_top_k=5)
        self.assertEqual(len(os_client.call_log), 6)

    async def test_primary_weight_boosts_rewritten_ranking(self):
        """queries[0] (rewritten) — вес 2.0, остальные — 1.0.
        Документ, найденный только rewritten-веткой, должен обойти документ,
        найденный только вариантами, при прочих равных."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[
            [hit("rewritten_only")],  # BM25 для queries[0]
            [hit("rewritten_only")],  # kNN для queries[0]
            [hit("variant_only")],    # BM25 для queries[1]
            [hit("variant_only")],    # kNN для queries[1]
        ])
        result = await multi_query_hybrid_search(
            os_client, embed, ["main", "v1"], final_top_k=5)
        self.assertEqual(result[0]["_id"], "rewritten_only")

    async def test_empty_queries_returns_empty(self):
        embed = FakeEmbed()
        os_client = FakeOpenSearch()
        result = await multi_query_hybrid_search(
            os_client, embed, [], final_top_k=5)
        self.assertEqual(result, [])
        self.assertEqual(len(embed.call_log), 0,
                         "пустой список не должен звать embed")

    async def test_source_excludes_dense_and_sparse(self):
        """Запросы обязаны исключать content_vector и content_sparse из
        _source, иначе на каждый hit прилетает 1024-мерный dense — сотни
        КБ payload'а на топ-50 без всякой пользы."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            os_client, embed, ["q"], final_top_k=5)

        for call in os_client.call_log:
            src = call["body"].get("_source")
            self.assertIsInstance(src, dict,
                                  "_source должен быть {'excludes': [...]}, "
                                  "а не whitelist — иначе новые поля в "
                                  "kb-v2 будут молча теряться")
            self.assertIn("content_vector", src.get("excludes", []))
            self.assertIn("content_sparse", src.get("excludes", []))

    async def test_index_defaults_to_kb_v2(self):
        """Индекс по умолчанию берётся из settings.index_name (kb-v2)."""
        from app.config import settings
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            os_client, embed, ["q"], final_top_k=5)
        for call in os_client.call_log:
            self.assertEqual(call["index"], settings.index_name)


if __name__ == "__main__":
    unittest.main()
