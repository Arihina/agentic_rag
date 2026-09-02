from __future__ import annotations

"""Финальный ответ пользователю по собранному пулу фрагментов."""

from pydantic import BaseModel, Field

from app.clients.llm import LLMClient
from app.config import settings
from app.core.context_format import format_context


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        description="Ответ пользователю на основе найденных фрагментов "
                    "документации")
    grounded: bool = Field(
        description="True, если в найденных фрагментах было достаточно "
                    "информации для ответа; False, если информации "
                    "недостаточно и это отражено в тексте ответа")


async def generate_answer(
    llm: LLMClient,
    user_query: str,
    chunks: list[dict],
    history: list[dict[str, str]] | None = None,
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> GeneratedAnswer:
    if not chunks:
        return GeneratedAnswer(
            answer="По вашему запросу не найдено релевантной информации "
                   "в базе знаний.",
            grounded=False,
        )

    context = format_context(chunks)
    history_text = ""
    if history:
        history_text = "История диалога:\n" + "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in history
        ) + "\n\n"

    user_prompt = (
        f"{history_text}"
        f"Найденные фрагменты документации:\n{context}\n\n"
        f"Вопрос пользователя: {user_query}\n\n"
        f"Сформулируй ответ, используя только приведённые фрагменты, со "
        f"ссылками на номера источников вида [1], [2] там, где это уместно."
    )

    return await llm.generate_structured(
        model=settings.llm_model_answer,
        system=system_prompt or settings.answer_system_prompt,
        prompt=user_prompt,
        response_model=GeneratedAnswer,
        temperature=(temperature if temperature is not None
                     else settings.llm_temperature_answer),
    )
