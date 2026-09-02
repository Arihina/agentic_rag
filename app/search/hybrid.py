from __future__ import annotations

"""Мульти-запросный гибридный поиск: BM25 + kNN + sparse, RRF-фьюжн по 3N
веткам.

primary_weight (2.0) для rewritten-ветки — эмпирика: варианты multi-query
специально «уплывают» синонимами ради recall, доверять им наравне с
rewritten нельзя, иначе фьюжн тянет выдачу в сторону наименее точного
варианта. На каждый запрос идёт три ветки — все три получают одинаковый
вес: primary если i==0, 1.0 иначе.

score_threshold задаётся пользователем в косинусах (rag_set.score_threshold),
конвертится в OpenSearch `_score` через score_from_cosine и применяется ТОЛЬКО
к kNN-ветке. BM25 и sparse не имеют осмысленной шкалы для порога, отсеиваются
только глубиной top_k каждой ветки.
"""

from collections import defaultdict

from opensearchpy import AsyncOpenSearch

from app.clients.embed import EmbedClient
from app.config import settings


SEARCH_SOURCE_EXCLUDES = ["content_vector", "content_sparse"]


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
    rag_id: str,
    os_client: AsyncOpenSearch,
    embed: EmbedClient,
    queries: list[str],
    /,
    *,
    score_threshold: float | None = None,
    index: str | None = None,
    bm25_top_k: int | None = None,
    knn_top_k: int | None = None,
    sparse_top_k: int | None = None,
    final_top_k: int = 10,
    primary_weight: float | None = None,
) -> list[dict]:
    """queries[0] — rewritten (основной запрос, повышенный вес);
    queries[1:] — варианты multi-query.

    rag_id — positional-only, чтобы синтаксически исключить вызов без
    tenant-фильтра. score_threshold — в косинусах, конвертится в
    OpenSearch `min_score` только для kNN-ветки (BM25 и sparse отсекаются
    только глубиной top_k).
    """
    if not queries:
        return []

    from app.search.branches import bm25_branch, knn_branch, sparse_branch
    from app.search.scoring import score_from_cosine

    index = index or settings.index_name
    bm25_top_k = bm25_top_k or settings.bm25_top_k
    knn_top_k = knn_top_k or settings.knn_top_k
    sparse_top_k = sparse_top_k or settings.sparse_top_k
    primary_weight = (primary_weight if primary_weight is not None
                      else settings.primary_weight)
    knn_min_score = (score_from_cosine(score_threshold)
                     if score_threshold is not None else None)

    items = await embed.embed(queries, pool="query")

    ranked_lists: list[list[dict]] = []
    weights: list[float] = []
    for i, query_text in enumerate(queries):
        bm25_hits = await bm25_branch(
            rag_id, query_text, bm25_top_k,
            client=os_client, index=index)
        knn_hits = await knn_branch(
            rag_id, items[i].dense, knn_top_k, knn_min_score,
            client=os_client, index=index)
        sparse_hits = await sparse_branch(
            rag_id, items[i].sparse, sparse_top_k,
            client=os_client, index=index)
        weight = primary_weight if i == 0 else 1.0
        ranked_lists.extend([bm25_hits, knn_hits, sparse_hits])
        weights.extend([weight, weight, weight])

    fused = reciprocal_rank_fusion(ranked_lists, weights=weights)
    return fused[:final_top_k]
