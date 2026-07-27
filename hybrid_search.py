from collections import defaultdict

from opensearchpy import OpenSearch

from config import settings
from embeddings import embed_query

_SOURCE_FIELDS = ["chunk_id", "doc_id", "content", "breadcrumb",
                  "chunk_type", "source_file", "anchor", "source_format"]


def _bm25_search(client: OpenSearch, index_name: str, query_text: str, top_k: int) -> list[dict]:
    body = {
        "size": top_k,
        "query": {"match": {"content": query_text}},
        "_source": _SOURCE_FIELDS,
    }
    resp = client.search(index=index_name, body=body)
    return resp["hits"]["hits"]


def _knn_search(client: OpenSearch, index_name: str, query_vector: list[float], top_k: int) -> list[dict]:
    body = {
        "size": top_k,
        "query": {"knn": {"content_vector": {"vector": query_vector, "k": top_k}}},
        "_source": _SOURCE_FIELDS,
    }
    resp = client.search(index=index_name, body=body)
    return resp["hits"]["hits"]


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[dict]:
    """
    Плоский RRF по произвольному числу уже отсортированных списков хитов OpenSearch.
    Один проход, без промежуточного фьюжна по парам (см. обсуждение архитектуры).
    """
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
    return [{**doc_lookup[doc_id], "_rrf_score": score} for doc_id, score in ranked]


def hybrid_search(
    client: OpenSearch,
    query_text: str,
    index_name: str = settings.index_name,
    bm25_top_k: int = 50,
    knn_top_k: int = 50,
    final_top_k: int = 10,
) -> list[dict]:
    """Гибридный поиск по одному запросу: BM25 + kNN, зафьюженные через RRF."""
    query_vector = embed_query(query_text)
    bm25_hits = _bm25_search(client, index_name, query_text, bm25_top_k)
    knn_hits = _knn_search(client, index_name, query_vector, knn_top_k)
    fused = reciprocal_rank_fusion([bm25_hits, knn_hits])
    return fused[:final_top_k]


def multi_query_hybrid_search(
    client: OpenSearch,
    queries: list[str],
    index_name: str = settings.index_name,
    bm25_top_k: int = 50,
    knn_top_k: int = 50,
    final_top_k: int = 10,
    primary_weight: float = 2.0,
) -> list[dict]:
    """
    Плоский RRF по 2N спискам (BM25+kNN на каждый вариант запроса).
    queries[0] считается основным (переписанным) запросом и получает больший вес,
    остальные варианты от multi-query — вес 1.0.
    """
    ranked_lists: list[list[dict]] = []
    weights: list[float] = []

    for i, query_text in enumerate(queries):
        query_vector = embed_query(query_text)
        bm25_hits = _bm25_search(client, index_name, query_text, bm25_top_k)
        knn_hits = _knn_search(client, index_name, query_vector, knn_top_k)
        weight = primary_weight if i == 0 else 1.0
        ranked_lists.extend([bm25_hits, knn_hits])
        weights.extend([weight, weight])

    fused = reciprocal_rank_fusion(ranked_lists, weights=weights)
    return fused[:final_top_k]
