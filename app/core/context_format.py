from __future__ import annotations

"""Формат контекста для LLM (индекс kb-v2).

Формат одного фрагмента: `[i] {heading path} · с. {pages}\n{content}`.

Нумерация [1], [2]… используется и в eval, и в answer — по одной и той
же нумерации LLM ссылается на источники в тексте ответа, а мы потом по
этим номерам собираем message_sources из тех же самых hits. Разъезд
нумерации сломает grounding: LLM скажет "[2]", а UI подставит другой
чанк.
"""


def _format_location(source: dict) -> str:
    parts: list[str] = []

    headings = source.get("headings") or []
    if headings:
        crumbs = ([headings] if isinstance(headings, str) else headings)
        joined = " > ".join(str(h) for h in crumbs if h)
        if joined:
            parts.append(joined)

    pages = source.get("pages") or []
    if pages:
        try:
            page_ints = sorted(int(p) for p in pages)
            first, last = page_ints[0], page_ints[-1]
            parts.append(f"с. {first}" if first == last
                         else f"с. {first}-{last}")
        except (TypeError, ValueError):
            pass

    return " · ".join(parts)


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(chunks, start=1):
        src = hit["_source"]
        location = _format_location(src)
        header = f"[{i}]" + (f" {location}" if location else "")
        parts.append(f"{header}\n{src.get('content', '')}")
    return "\n\n".join(parts)
