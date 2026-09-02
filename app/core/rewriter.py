from __future__ import annotations

"""Переписать сообщение пользователя в самодостаточный поисковый запрос
с учётом истории диалога.
"""

from pydantic import BaseModel, Field

from app.clients.llm import LLMClient
from app.config import settings


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(
        description="Самодостаточная переформулировка запроса пользователя")


async def rewrite_query(
    llm: LLMClient,
    history: list[dict[str, str]],
    current_message: str,
) -> str:
    if not history:
        return current_message

    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in history)
    user_prompt = (
        f"История диалога:\n{history_text}\n\n"
        f"Последнее сообщение пользователя: {current_message}\n\n"
        f"Сформулируй самодостаточный поисковый запрос."
    )

    result = await llm.generate_structured(
        model=settings.llm_model_rewriter,
        system=settings.rewriter_system_prompt,
        prompt=user_prompt,
        response_model=RewrittenQuery,
        temperature=settings.llm_temperature_rewriter,
    )
    return result.rewritten_query
