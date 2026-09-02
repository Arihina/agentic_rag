from __future__ import annotations

"""Тесты multi_query_hybrid_search.

Ядро проверяемого поведения:
- rag_id — обязательный первый positional-only аргумент; вызов kwarg'ом
  или без него не проходит компиляцию;
- батч-эмбеддинг: /embed зовётся один раз со всем списком queries;
- OpenSearch зовётся 3N раз (BM25 + kNN + sparse на каждый вариант);
- primary_weight=2.0 применяется к rewritten (queries[0]) ко ВСЕМ трём
  веткам, варианты идут с весом 1.0.
"""

import unittest

from app.search.hybrid import multi_query_hybrid_search
from app.search.scoring import score_from_cosine
from tests.core_fakes import FakeEmbed, FakeOpenSearch, hit

RAG = "11111111-1111-1111-1111-111111111111"


class MultiQueryHybridSearchTests(unittest.IsolatedAsyncioTestCase):

    async def test_embed_called_once_with_all_queries(self):
        """Инвариант batch-embed: один вызов /embed на всю multi-query.
        Если реализация случайно вернётся к «embed в цикле», sparse-вектора
        разъедутся против того, что писала ingestion."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            RAG, os_client, embed, ["main", "v1", "v2"], final_top_k=5)
        self.assertEqual(len(embed.call_log), 1)
        self.assertEqual(embed.call_log[0], ["main", "v1", "v2"])

    async def test_opensearch_called_three_times_per_query(self):
        """BM25 + kNN + sparse на каждую формулировку. 3 запроса → 9 вызовов."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            RAG, os_client, embed, ["q1", "q2", "q3"], final_top_k=5)
        self.assertEqual(len(os_client.call_log), 9)

    async def test_primary_weight_boosts_rewritten_ranking(self):
        """queries[0] (rewritten) — вес 2.0, остальные — 1.0.
        Документ, найденный только rewritten-ветками, должен обойти
        документ, найденный только вариантами, при прочих равных."""
        embed = FakeEmbed()
        # Порядок вызовов: rewritten × [bm25, knn, sparse], v1 × [bm25, knn, sparse].
        os_client = FakeOpenSearch(responses=[
            [hit("rewritten_only")],  # bm25 для queries[0]
            [hit("rewritten_only")],  # knn для queries[0]
            [hit("rewritten_only")],  # sparse для queries[0]
            [hit("variant_only")],    # bm25 для queries[1]
            [hit("variant_only")],    # knn для queries[1]
            [hit("variant_only")],    # sparse для queries[1]
        ])
        result = await multi_query_hybrid_search(
            RAG, os_client, embed, ["main", "v1"], final_top_k=5)
        self.assertEqual(result[0]["_id"], "rewritten_only")

    async def test_empty_queries_returns_empty(self):
        embed = FakeEmbed()
        os_client = FakeOpenSearch()
        result = await multi_query_hybrid_search(
            RAG, os_client, embed, [], final_top_k=5)
        self.assertEqual(result, [])
        self.assertEqual(len(embed.call_log), 0,
                         "пустой список не должен звать embed")

    async def test_rag_id_in_every_search_call(self):
        """Инвариант мультитенантности: фильтр по rag_id обязан быть во
        всех 3N search-запросах, каждой ветки. Если хоть один запрос
        уходит без фильтра — утечка чужих документов."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            RAG, os_client, embed, ["q1", "q2"], final_top_k=5)

        for call in os_client.call_log:
            body = call["body"]
            # Ветки строят по-разному; общий инвариант — где-то в теле
            # должно быть {"term": {"rag_id": RAG}}. Пройдём по всем
            # уровням bool.filter / knn.filter.
            has_filter = _find_rag_filter(body, RAG)
            self.assertTrue(has_filter,
                            f"rag_id-фильтр не найден в теле запроса: {body}")

    async def test_source_excludes_dense_and_sparse(self):
        """Запросы обязаны исключать content_vector и content_sparse из
        _source, иначе на каждый hit прилетает 1024-мерный dense — сотни
        КБ payload'а на топ-50 без всякой пользы."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            RAG, os_client, embed, ["q"], final_top_k=5)

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
            RAG, os_client, embed, ["q"], final_top_k=5)
        for call in os_client.call_log:
            self.assertEqual(call["index"], settings.index_name)

    async def test_score_threshold_becomes_knn_min_score(self):
        """score_threshold задан в косинусах, конвертится в OpenSearch
        `_score` через score_from_cosine и применяется ТОЛЬКО к kNN-ветке.
        BM25 и sparse не должны получить min_score вовсе."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            RAG, os_client, embed, ["q"],
            score_threshold=0.5, final_top_k=5)

        expected = score_from_cosine(0.5)
        knn_calls = [c for c in os_client.call_log
                     if "knn" in str(c["body"].get("query", {}))]
        non_knn_calls = [c for c in os_client.call_log
                         if "knn" not in str(c["body"].get("query", {}))]

        self.assertEqual(len(knn_calls), 1,
                         "ожидался ровно 1 kNN-вызов на 1 запрос")
        self.assertAlmostEqual(
            knn_calls[0]["body"].get("min_score"), expected, places=6)
        for call in non_knn_calls:
            self.assertNotIn("min_score", call["body"],
                             "BM25 и sparse не имеют осмысленной шкалы "
                             "для порога — min_score не должен ставиться")

    async def test_no_score_threshold_no_min_score(self):
        """Если threshold не задан — min_score не выставляется даже в kNN."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await multi_query_hybrid_search(
            RAG, os_client, embed, ["q"], final_top_k=5)
        for call in os_client.call_log:
            self.assertNotIn("min_score", call["body"])

    async def test_rag_id_must_be_positional(self):
        """rag_id positional-only: вызов kwarg'ом должен падать TypeError.
        Это защита от случая, когда клиентский код случайно потерял
        аргумент и Python подставил бы дефолт — здесь дефолта нет вовсе."""
        embed = FakeEmbed()
        os_client = FakeOpenSearch()
        with self.assertRaises(TypeError):
            await multi_query_hybrid_search(  # type: ignore
                rag_id=RAG, os_client=os_client, embed=embed,
                queries=["q"], final_top_k=5)


def _find_rag_filter(node: dict | list, rag_id: str) -> bool:
    """Проверить, есть ли где-то в теле запроса {"term": {"rag_id": rag_id}}."""
    if isinstance(node, dict):
        if node.get("term", {}).get("rag_id") == rag_id:
            return True
        return any(_find_rag_filter(v, rag_id) for v in node.values())
    if isinstance(node, list):
        return any(_find_rag_filter(item, rag_id) for item in node)
    return False


if __name__ == "__main__":
    unittest.main()
