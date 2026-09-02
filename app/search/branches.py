from __future__ import annotations

"""Три ветви поиска по kb-v2: BM25, kNN, sparse (rank_features).

Инвариант мультитенантности: `rag_id` — обязательный ПЕРВЫЙ позиционный
аргумент каждой ветки, `positional-only` через `/`. Это делает синтаксически
невозможным вызвать ветку без него или подменить kwarg'ом в клиентском
коде. Индекс общий на всю платформу; забытый фильтр — не деградация
выдачи, а утечка чужих документов.

Все три ветки строят один и тот же `bool.filter = [{term rag_id}]` — фильтр
не участвует в скоринге (нам не нужно, чтобы факт «набор совпал» тянул
документ вверх), но обязателен и всегда первый.

Поля _source везде через `SEARCH_SOURCE_EXCLUDES` — на выходе всё, кроме
тяжёлых `content_vector` и `content_sparse`.
"""

from opensearchpy import AsyncOpenSearch

from app.search.hybrid import SEARCH_SOURCE_EXCLUDES


def _rag_filter(rag_id: str) -> dict:
    return {"term": {"rag_id": rag_id}}


async def bm25_branch(
    rag_id: str, query: str, top_k: int, /,
    *, client: AsyncOpenSearch, index: str,
) -> list[dict]:
    """BM25 по полю content, отфильтрованному по rag_id."""
    body = {
        "size": top_k,
        "query": {"bool": {
            "filter": [_rag_filter(rag_id)],
            "must": [{"match": {"content": query}}],
        }},
        "_source": {"excludes": SEARCH_SOURCE_EXCLUDES},
    }
    resp = await client.search(index=index, body=body)
    return resp["hits"]["hits"]


async def knn_branch(
    rag_id: str, query_vector: list[float], top_k: int,
    min_score: float | None, /,
    *, client: AsyncOpenSearch, index: str,
) -> list[dict]:
    """kNN + pre-filter по rag_id + опциональный min_score.

    filter внутри knn — pre-filter, работает на уровне HNSW-графа
    (OpenSearch 2.4+, faiss engine). Это правильнее, чем post-filter на
    top_k результатах: без pre-filter топ-k из ВСЕХ наборов может целиком
    оказаться из чужих, и после отсева останется 0-1 своих.

    min_score применяется поверх результата knn — на уровне тела запроса,
    не внутри knn (min_score внутри knn — версия-зависимо и хуже
    документировано)."""
    body: dict = {
        "size": top_k,
        "query": {"knn": {"content_vector": {
            "vector": query_vector,
            "k": top_k,
            "filter": _rag_filter(rag_id),
        }}},
        "_source": {"excludes": SEARCH_SOURCE_EXCLUDES},
    }
    if min_score is not None:
        body["min_score"] = min_score
    resp = await client.search(index=index, body=body)
    return resp["hits"]["hits"]


async def sparse_branch(
    rag_id: str, sparse_weights: dict[str, float], top_k: int, /,
    *, client: AsyncOpenSearch, index: str,
) -> list[dict]:
    """Sparse-ветка по `content_sparse` (rank_features).

    rank_feature-запрос работает по одному конкретному полю за раз, а
    content_sparse — это плоский объект `{"<token_id>": <weight>}`. Значит
    для запроса из N ненулевых токенов собираем bool.should из N
    rank_feature-подзапросов, каждый с boost'ом равным весу токена.
    minimum_should_match=1 — иначе bool с только filter + should вернёт
    ноль (без явного match хоть одному should документы отфильтровались бы
    и по релевантности не оказались).
    """
    if not sparse_weights:
        return []

    should = [
        {"rank_feature": {
            "field": f"content_sparse.{token_id}",
            "boost": float(weight),
        }}
        for token_id, weight in sparse_weights.items()
    ]
    body = {
        "size": top_k,
        "query": {"bool": {
            "filter": [_rag_filter(rag_id)],
            "should": should,
            "minimum_should_match": 1,
        }},
        "_source": {"excludes": SEARCH_SOURCE_EXCLUDES},
    }
    resp = await client.search(index=index, body=body)
    return resp["hits"]["hits"]
