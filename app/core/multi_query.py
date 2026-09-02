from __future__ import annotations

"""Генерация N альтернативных формулировок поискового запроса.

Задача — расширить recall: rewritten-запрос точен, но синонимический
запас модели bge-m3 всё равно ограничен, и лексически близкие, но иначе
сформулированные варианты вытаскивают релевантные документы, которые
первая формулировка пропустила. В RRF-фьюжне варианты потом идут с
меньшим весом, чем rewritten.
"""

from pydantic import BaseModel, Field

from app.clients.llm import LLMClient
from app.config import settings


class QueryVariants(BaseModel):
    variants: list[str] = Field(
        description="Альтернативные формулировки поискового запроса")


async def generate_query_variants(
    llm: LLMClient,
    query: str,
    n: int | None = None,
) -> list[str]:
    n = n or settings.multi_query_variants_count

    result = await llm.generate_structured(
        model=settings.llm_model_multi_query,
        system=settings.multi_query_system_prompt_template.format(n=n),
        prompt=f"Исходный запрос: {query}",
        response_model=QueryVariants,
        temperature=settings.llm_temperature_multi_query,
    )
    return result.variants[:n]
