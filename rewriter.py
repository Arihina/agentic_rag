from pydantic import BaseModel, Field

from config import settings
from llm_client import generate_structured


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(
        description="Самодостаточная переформулировка запроса пользователя")


def rewrite_query(history: list[dict[str, str]], current_message: str) -> str:
    """
    history: список {"role": "user"|"assistant", "content": str} в хронологическом порядке.
    При пустой истории переписывать нечего — возвращает current_message без изменений
    (лишний LLM-вызов на первом сообщении диалога не нужен).
    """
    if not history:
        return current_message

    history_text = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in history)
    user_prompt = (
        f"История диалога:\n{history_text}\n\n"
        f"Последнее сообщение пользователя: {current_message}\n\n"
        f"Сформулируй самодостаточный поисковый запрос."
    )

    result = generate_structured(
        system_prompt=settings.rewriter_system_prompt,
        user_prompt=user_prompt,
        response_model=RewrittenQuery,
        model=settings.llm_model,
        temperature=settings.rewriter_temperature,
    )
    
    return result.rewritten_query
