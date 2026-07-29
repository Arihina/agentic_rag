def format_context(chunks: list[dict]) -> str:
    """
    Формирует пронумерованный контекст из хитов OpenSearch для промптов.
    Нумерация [1], [2]... используется и в eval, и в generate_answer — по одной
    и той же нумерации LLM ссылается на источники, что позволяет сверять цитаты
    в ответе с конкретным чанком в выводе.
    """
    parts = []
    for i, hit in enumerate(chunks, start=1):
        src = hit["_source"]
        header = f"[{i}] Источник: {src.get('source_file', '?')} | {src.get('breadcrumb', '')}"
        parts.append(f"{header}\n{src.get('content', '')}")

    return "\n\n".join(parts)
