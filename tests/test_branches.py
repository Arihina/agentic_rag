from __future__ import annotations

"""Три ветки поиска — тесты на инварианты каждой.

Ключевое: rag_id — positional-only, попытка передать kwarg-ом падает
TypeError. Это не косметика: если бы `rag_id` был обычным keyword'ом,
клиентский код мог бы случайно забыть аргумент и получить дефолт,
пропускающий фильтр, — а с positional-only попытка так вызвать даже не
скомпилируется.
"""

import unittest

from app.search.branches import bm25_branch, knn_branch, sparse_branch
from tests.core_fakes import FakeOpenSearch, hit

RAG = "11111111-1111-1111-1111-111111111111"


class BranchInvariantsTests(unittest.IsolatedAsyncioTestCase):
    """Общий инвариант всех трёх ветвей: rag_id positional-only."""

    async def test_bm25_rejects_rag_id_as_kwarg(self):
        os_client = FakeOpenSearch()
        with self.assertRaises(TypeError):
            await bm25_branch(  # type: ignore
                rag_id=RAG, query="q", top_k=5,
                client=os_client, index="kb-v2")

    async def test_knn_rejects_rag_id_as_kwarg(self):
        os_client = FakeOpenSearch()
        with self.assertRaises(TypeError):
            await knn_branch(  # type: ignore
                rag_id=RAG, query_vector=[0.1], top_k=5, min_score=None,
                client=os_client, index="kb-v2")

    async def test_sparse_rejects_rag_id_as_kwarg(self):
        os_client = FakeOpenSearch()
        with self.assertRaises(TypeError):
            await sparse_branch(  # type: ignore
                rag_id=RAG, sparse_weights={"1": 1.0}, top_k=5,
                client=os_client, index="kb-v2")


class BM25BranchTests(unittest.IsolatedAsyncioTestCase):

    async def test_body_has_rag_filter_and_match(self):
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await bm25_branch(RAG, "как настроить X", 5,
                          client=os_client, index="kb-v2")
        body = os_client.call_log[0]["body"]

        # Форма: bool { filter: [term rag_id], must: [match content] }.
        bool_q = body["query"]["bool"]
        self.assertEqual(bool_q["filter"], [{"term": {"rag_id": RAG}}])
        self.assertEqual(bool_q["must"], [
                         {"match": {"content": "как настроить X"}}])
        self.assertEqual(body["size"], 5)


class KnnBranchTests(unittest.IsolatedAsyncioTestCase):

    async def test_body_has_prefilter_in_knn(self):
        """kNN + pre-filter: filter кладётся ВНУТРЬ knn, не снаружи через
        bool. Без pre-filter топ-k из всех наборов может целиком оказаться
        из чужих, и после post-filter останется 0."""
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await knn_branch(RAG, [0.1, 0.2], 5, None,
                         client=os_client, index="kb-v2")
        body = os_client.call_log[0]["body"]

        knn = body["query"]["knn"]["content_vector"]
        self.assertEqual(knn["vector"], [0.1, 0.2])
        self.assertEqual(knn["k"], 5)
        self.assertEqual(knn["filter"], {"term": {"rag_id": RAG}})
        # min_score не задан — не должен появиться.
        self.assertNotIn("min_score", body)

    async def test_min_score_applied(self):
        """min_score применяется на уровне тела запроса (не внутри knn) —
        так надёжнее по версиям OpenSearch."""
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        await knn_branch(RAG, [0.1], 5, 0.75,
                         client=os_client, index="kb-v2")
        body = os_client.call_log[0]["body"]

        self.assertAlmostEqual(body["min_score"], 0.75)


class SparseBranchTests(unittest.IsolatedAsyncioTestCase):

    async def test_body_has_rank_feature_per_token(self):
        """Sparse-запрос: bool.should из N rank_feature-подзапросов, по
        одному на ненулевой токен. `minimum_should_match=1` обязателен —
        иначе bool с только filter+should пропускает всё."""
        os_client = FakeOpenSearch(responses=[[hit("a")]])
        weights = {"12345": 0.87, "67890": 0.42, "11111": 0.15}
        await sparse_branch(RAG, weights, 5,
                            client=os_client, index="kb-v2")
        body = os_client.call_log[0]["body"]

        bool_q = body["query"]["bool"]
        self.assertEqual(bool_q["filter"], [{"term": {"rag_id": RAG}}])
        self.assertEqual(bool_q["minimum_should_match"], 1)

        # По одному rank_feature на токен, с корректными field/boost.
        should = bool_q["should"]
        self.assertEqual(len(should), 3)
        fields = {c["rank_feature"]["field"] for c in should}
        self.assertEqual(fields, {
            "content_sparse.12345",
            "content_sparse.67890",
            "content_sparse.11111",
        })
        # Веса переданы как boost'ы (порядок не гарантирован, проверяем set).
        boosts = {round(c["rank_feature"]["boost"], 4) for c in should}
        self.assertEqual(boosts, {0.87, 0.42, 0.15})

    async def test_empty_sparse_weights_returns_empty(self):
        """Если /embed вернул пустой sparse — ветку вообще не зовём.
        Пустой bool.should с minimum_should_match=1 вернул бы 0 хитов,
        зря нагрузив OpenSearch."""
        os_client = FakeOpenSearch()
        result = await sparse_branch(RAG, {}, 5,
                                     client=os_client, index="kb-v2")
        self.assertEqual(result, [])
        self.assertEqual(len(os_client.call_log), 0,
                         "пустой sparse не должен даже вызывать OpenSearch")


if __name__ == "__main__":
    unittest.main()
