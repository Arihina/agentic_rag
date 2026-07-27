"""
Поиск по уже наполненной коллекции OpenSearch (после прогона
create_collection_opensearch.py на реальных документах предприятия).
Ничего не создаёт и не переиндексирует — только читает существующий индекс.

Запуск:
    python query_search.py                          # интерактивный режим (REPL)
    python query_search.py "как обслуживать щит"     # разовый запрос и выход
    python query_search.py --top_k 10 --index my_index

В интерактивном режиме:
    - обычная строка              -> hybrid_search (BM25 + kNN + RRF)
    - строка с разделителем "|"   -> multi_query_hybrid_search
      (пример: щит обслуживание | периодичность ТО щита | плановый осмотр щита)
    - "exit" / "quit" / Ctrl+D    -> выход
"""
import argparse

from opensearchpy import OpenSearch

from config import settings
from hybrid_search import hybrid_search, multi_query_hybrid_search
from opensearch_client import get_client


def print_index_stats(client: OpenSearch, index_name: str) -> None:
    if not client.indices.exists(index=index_name):
        print(
            f"Индекс '{index_name}' не найден. Сначала прогони create_collection_opensearch.py.")
        raise SystemExit(1)
    count = client.count(index=index_name)["count"]
    print(f"Индекс '{index_name}': документов в коллекции — {count}\n")


def print_results(results: list[dict]) -> None:
    if not results:
        print("  Ничего не найдено.\n")
        return
    for rank, hit in enumerate(results, start=1):
        src = hit["_source"]
        print(
            f"{rank}. [{hit['_rrf_score']:.4f}] {src.get('doc_id', '?')} / {src.get('source_file', '?')}")
        print(
            f"   {src.get('breadcrumb', '')} ({src.get('chunk_type', '?')}, {src.get('source_format', '?')})")
        content = src.get("content", "")
        print(f"   {content[:300]}{'...' if len(content) > 300 else ''}")
        print()


def run_query(client: OpenSearch, query_text: str, top_k: int, index_name: str) -> None:
    if "|" in query_text:
        variants = [q.strip() for q in query_text.split("|") if q.strip()]
        print(f"Multi-query поиск, вариантов: {len(variants)}")
        results = multi_query_hybrid_search(
            client, variants, index_name=index_name, final_top_k=top_k)
    else:
        results = hybrid_search(
            client, query_text, index_name=index_name, final_top_k=top_k)
    print_results(results)


def interactive_loop(client: OpenSearch, top_k: int, index_name: str) -> None:
    print("Интерактивный режим. 'exit'/'quit' для выхода, '|' между вариантами запроса — multi-query.\n")
    while True:
        try:
            query_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query_text:
            continue
        if query_text.lower() in {"exit", "quit"}:
            break
        run_query(client, query_text, top_k, index_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Поиск по существующей коллекции OpenSearch")
    parser.add_argument("query", nargs="?", default=None,
                        help="Разовый запрос; без него — интерактивный режим")
    parser.add_argument("--index", default=settings.index_name)
    parser.add_argument("--top_k", type=int, default=15)
    args = parser.parse_args()

    client = get_client()
    print_index_stats(client, args.index)

    if args.query:
        run_query(client, args.query, args.top_k, args.index)
    else:
        interactive_loop(client, args.top_k, args.index)


if __name__ == "__main__":
    main()
