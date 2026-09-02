from __future__ import annotations

"""Мульти-запросный гибридный поиск: BM25 + kNN, RRF-фьюжн по 2N веткам.

primary_weight (2.0) для rewritten-ветки — эмпирика: варианты multi-query
специально «уплывают» синонимами ради recall, доверять им наравне с
rewritten нельзя, иначе фьюжн тянет выдачу в сторону наименее точного
варианта.
"""

from collections import defaultdict

from opensearchpy import AsyncOpenSearch

from app.clients.embed import EmbedClient
from app.config import settings

_SOURCE_FIELDS = ["chunk_id", "doc_id", "content", "breadcrumb",
                  "chunk_type", "source_file", "anchor", "source_format"]


async def _bm25_search(
    client: AsyncOpenSearch, index: str, query_text: str, top_k: int,
) -> list[dict]:
    body = {
        "size": top_k,
        "query": {"match": {"content": query_text}},
        "_source": _SOURCE_FIELDS,
    }
    resp = await client.search(index=index, body=body)
    return resp["hits"]["hits"]


async def _knn_search(
    client: AsyncOpenSearch, index: str,
    query_vector: list[float], top_k: int,
) -> list[dict]:
    body = {
        "size": top_k,
        "query": {"knn": {"content_vector": {
            "vector": query_vector, "k": top_k}}},
        "_source": _SOURCE_FIELDS,
    }
    resp = await client.search(index=index, body=body)
    return resp["hits"]["hits"]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[dict]:
    """Плоский RRF по произвольному числу отсортированных списков хитов.
    Один проход, без парного слияния — так и алгоритм проще, и веса ветвей
    применяются напрямую, а не размазываются по глубине дерева слияний."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, float] = defaultdict(float)
    doc_lookup: dict[str, dict] = {}

    for list_idx, hits in enumerate(ranked_lists):
        weight = weights[list_idx]
        for rank, hit in enumerate(hits, start=1):
            doc_id = hit["_id"]
            scores[doc_id] += weight * (1.0 / (k + rank))
            doc_lookup.setdefault(doc_id, hit)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [{**doc_lookup[doc_id], "_rrf_score": score}
            for doc_id, score in ranked]


async def multi_query_hybrid_search(
    os_client: AsyncOpenSearch,
    embed: EmbedClient,
    queries: list[str],
    *,
    index: str | None = None,
    bm25_top_k: int | None = None,
    knn_top_k: int | None = None,
    final_top_k: int = 10,
    primary_weight: float | None = None,
) -> list[dict]:
    """queries[0] — rewritten (основной запрос, повышенный вес);
    queries[1:] — варианты multi-query."""
    if not queries:
        return []

    index = index or settings.index_name
    bm25_top_k = bm25_top_k or settings.bm25_top_k
    knn_top_k = knn_top_k or settings.knn_top_k
    primary_weight = (primary_weight if primary_weight is not None
                      else settings.primary_weight)

    items = await embed.embed(queries, pool="query")
    query_vectors = [item.dense for item in items]

    ranked_lists: list[list[dict]] = []
    weights: list[float] = []
    for i, query_text in enumerate(queries):
        bm25_hits = await _bm25_search(
            os_client, index, query_text, bm25_top_k)
        knn_hits = await _knn_search(
            os_client, index, query_vectors[i], knn_top_k)
        weight = primary_weight if i == 0 else 1.0
        ranked_lists.extend([bm25_hits, knn_hits])
        weights.extend([weight, weight])

    fused = reciprocal_rank_fusion(ranked_lists, weights=weights)
    return fused[:final_top_k]
