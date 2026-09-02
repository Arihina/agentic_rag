from __future__ import annotations

"""Eval + reflection одним LLM-вызовом."""

from pydantic import BaseModel, Field

from app.clients.llm import LLMClient
from app.config import settings
from app.core.context_format import format_context


class EvalResult(BaseModel):
    sufficient: bool = Field(
        description="True, если найденных фрагментов достаточно для полного "
                    "и точного ответа")
    reasoning: str = Field(
        default="", description="Краткое обоснование решения")
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="Каких конкретно сведений не хватает в контексте; "
                    "пусто, если sufficient=true")
    next_queries: list[str] = Field(
        default_factory=list,
        description="Новые поисковые запросы для следующей итерации; "
                    "пусто, если sufficient=true")


async def evaluate(
    llm: LLMClient, query: str, chunks: list[dict],
) -> EvalResult:
    if not chunks:
        return EvalResult(
            sufficient=False,
            reasoning="По запросу не найдено ни одного фрагмента.",
            missing_aspects=["вся информация по теме запроса"],
            next_queries=[query],
        )

    context = format_context(chunks)
    user_prompt = (
        f"Вопрос пользователя: {query}\n\n"
        f"Найденные фрагменты документации:\n{context}\n\n"
        f"Оцени, достаточно ли этих фрагментов, чтобы полно и точно "
        f"ответить на вопрос."
    )

    return await llm.generate_structured(
        model=settings.llm_model_eval,
        system=settings.eval_system_prompt,
        prompt=user_prompt,
        response_model=EvalResult,
        temperature=settings.llm_temperature_eval,
    )
