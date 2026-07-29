"""
Ручное тестирование полного конвейера: rewrite -> multi-query -> hybrid search -> генерация ответа.
Поддерживает многоходовой диалог, чтобы проверить, как rewriter разворачивает
местоимения и ссылки на предыдущий контекст ('а по нему?', 'а как часто это?').

Запуск:
    python main.py                              # интерактивный многоходовой режим
    python main.py "как обслуживать РЩ-3"       # разовый запрос без истории
    python main.py --top_k 3

В интерактивном режиме история копится автоматически между сообщениями
(реальный сгенерированный ответ идёт в историю как реплика ассистента).
'reset' — очистить историю, 'exit'/'quit' — выйти.
"""
import argparse

from opensearchpy import OpenSearch

from answer import GeneratedAnswer, generate_answer
from opensearch_client import get_client
from retrieve import retrieve


def print_pipeline_result(original: str, rewritten: str, variants: list[str], results: list[dict]) -> None:
    print(f"\nИсходный запрос:      {original}")
    print(f"Переписанный запрос:  {rewritten}")
    print("Варианты multi-query:")
    for v in variants:
        print(f"  - {v}")

    print(f"\nНайденные фрагменты (top {len(results)}):")
    if not results:
        print("  Ничего не найдено.")
        return
    for rank, hit in enumerate(results, start=1):
        src = hit["_source"]
        print(
            f"[{rank}] [{hit['_rrf_score']:.4f}] {src.get('doc_id', '?')} / {src.get('source_file', '?')}")
        print(
            f"    {src.get('breadcrumb', '')} ({src.get('chunk_type', '?')}, {src.get('source_format', '?')})")
        content = src.get("content", "")
        print(f"    {content[:200]}{'...' if len(content) > 200 else ''}")


def print_answer(answer: GeneratedAnswer) -> None:
    print(f"\n{'-' * 60}")
    print(f"Ответ (grounded={answer.grounded}):")
    print(answer.answer)
    print(f"{'-' * 60}")


def run_once(
    client: OpenSearch, query_text: str, history: list[dict[str, str]], top_k: int,
) -> tuple[list[dict], GeneratedAnswer]:
    rewritten, variants, results = retrieve(
        client, query_text, history=history, top_k=top_k)
    print_pipeline_result(query_text, rewritten, variants, results)

    answer = generate_answer(query_text, results, history=history)
    print_answer(answer)
    return results, answer


def interactive_loop(client: OpenSearch, top_k: int) -> None:
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

        _, answer = run_once(client, query_text, history, top_k)
        history.append({"role": "user", "content": query_text})
        history.append({"role": "assistant", "content": answer.answer})
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Тестирование rewrite -> multi-query -> hybrid search -> ответ")
    parser.add_argument("query", nargs="?", default=None,
                        help="Разовый запрос; без него — интерактивный режим")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    client = get_client()

    if args.query:
        run_once(client, args.query, history=[], top_k=args.top_k)
    else:
        interactive_loop(client, args.top_k)


if __name__ == "__main__":
    main()
