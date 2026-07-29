from opensearchpy import OpenSearch

from hybrid_search import multi_query_hybrid_search
from multi_query import generate_query_variants
from rewriter import rewrite_query


def retrieve(
    client: OpenSearch,
    current_message: str,
    history: list[dict[str, str]] | None = None,
    top_k: int = 10,
) -> tuple[str, list[str], list[dict]]:
    rewritten = rewrite_query(history or [], current_message)
    variants = generate_query_variants(rewritten)
    all_queries = [rewritten, *variants]

    results = multi_query_hybrid_search(client, all_queries, final_top_k=top_k)

    return rewritten, variants, results
