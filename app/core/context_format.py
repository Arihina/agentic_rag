from __future__ import annotations

"""Формат контекста для LLM.

Нумерация [1], [2]… идёт по одному и тому же массиву chunks и в eval, и в
answer — так LLM ссылается на источники, а мы потом по номерам собираем
message_sources из тех же хитов. Разъезд нумерации сломает грouding.
"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(chunks, start=1):
        src = hit["_source"]
        header = (f"[{i}] Источник: {src.get('source_file', '?')} | "
                  f"{src.get('breadcrumb', '')}")
        parts.append(f"{header}\n{src.get('content', '')}")

    return "\n\n".join(parts)
