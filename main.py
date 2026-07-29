"""
Ручное тестирование полного агентного цикла:
rewrite -> multi-query -> hybrid search -> eval/reflection -> (повтор или ответ).

Запуск:
    python main.py                              # интерактивный многоходовой режим
    python main.py "как обслуживать РЩ-3"       # разовый запрос без истории
    python main.py --top_k 3 --max_iterations 5

В интерактивном режиме история копится автоматически между сообщениями.
'reset' — очистить историю, 'exit'/'quit' — выйти.
"""
import argparse

from opensearchpy import OpenSearch

from agent import AgentTrace, run_agent
from opensearch_client import get_client


def print_trace(trace: AgentTrace) -> None:
    print(f"\nПереписанный запрос: {trace.rewritten_query}")

    for log in trace.iterations:
        print(f"\n--- Итерация {log.iteration} ---")
        print("Запросы:")
        for q in log.queries:
            print(f"  - {q}")
        print(
            f"Новых чанков найдено: {log.new_chunks_found} | пересечение с пулом: {log.overlap_with_pool:.2f}")
        print(
            f"Eval: sufficient={log.eval_result.sufficient} | {log.eval_result.reasoning}")
        if log.eval_result.missing_aspects:
            print(
                f"  Не хватает: {', '.join(log.eval_result.missing_aspects)}")
        if log.eval_result.next_queries:
            print(
                f"  Следующие запросы: {', '.join(log.eval_result.next_queries)}")

    print(
        f"\nОстановка: {trace.stopped_reason} (итераций: {len(trace.iterations)})")

    print(f"\nИтоговые фрагменты (top {len(trace.final_chunks)}):")
    if not trace.final_chunks:
        print("  Ничего не найдено.")
    for rank, hit in enumerate(trace.final_chunks, start=1):
        src = hit["_source"]
        print(
            f"[{rank}] [{hit['_rrf_score']:.4f}] {src.get('doc_id', '?')} / {src.get('source_file', '?')}")
        print(
            f"    {src.get('breadcrumb', '')} ({src.get('chunk_type', '?')}, {src.get('source_format', '?')})")
        content = src.get("content", "")
        print(f"    {content[:200]}{'...' if len(content) > 200 else ''}")

    print(f"\n{'-' * 60}")
    print(f"Ответ (grounded={trace.answer.grounded}):")
    print(trace.answer.answer)
    print(f"{'-' * 60}")


def run_once(
    client: OpenSearch, query_text: str, history: list[dict[str, str]], top_k: int, max_iterations: int | None,
) -> AgentTrace:
    trace = run_agent(client, query_text, history=history,
                      top_k=top_k, max_iterations=max_iterations)
    print_trace(trace)
    return trace


def interactive_loop(client: OpenSearch, top_k: int, max_iterations: int | None) -> None:
    print("Интерактивный многоходовой режим. 'reset' — очистить историю, 'exit'/'quit' — выйти.\n")
    history: list[dict[str, str]] = []
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
        if query_text.lower() == "reset":
            history = []
            print("История очищена.\n")
            continue

        trace = run_once(client, query_text, history, top_k, max_iterations)
        history.append({"role": "user", "content": query_text})
        history.append({"role": "assistant", "content": trace.answer.answer})
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Тестирование полного агентного цикла agentic RAG")
    parser.add_argument("query", nargs="?", default=None,
                        help="Разовый запрос; без него — интерактивный режим")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="По умолчанию — из settings.max_iterations")
    args = parser.parse_args()

    client = get_client()

    if args.query:
        run_once(client, args.query, history=[], top_k=args.top_k,
                 max_iterations=args.max_iterations)
    else:
        interactive_loop(client, args.top_k, args.max_iterations)


if __name__ == "__main__":
    main()
