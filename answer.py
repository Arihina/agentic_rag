from pydantic import BaseModel, Field

from config import settings
from llm_client import generate_structured


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        description="Ответ пользователю на основе найденных фрагментов документации")
    grounded: bool = Field(
        description="True, если в найденных фрагментах было достаточно информации для ответа; "
                    "False, если информации недостаточно и это отражено в тексте ответа"
    )


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(chunks, start=1):
        src = hit["_source"]
        header = f"[{i}] Источник: {src.get('source_file', '?')} | {src.get('breadcrumb', '')}"
        parts.append(f"{header}\n{src.get('content', '')}")
    return "\n\n".join(parts)


def generate_answer(
    user_query: str,
    chunks: list[dict],
    history: list[dict[str, str]] | None = None,
) -> GeneratedAnswer:
    if not chunks:
        return GeneratedAnswer(
            answer="По вашему запросу не найдено релевантной информации в базе знаний.",
            grounded=False,
        )

    context = _format_context(chunks)
    history_text = ""
    if history:
        history_text = "История диалога:\n" + "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in history
        ) + "\n\n"

    user_prompt = (
        f"{history_text}"
        f"Найденные фрагменты документации:\n{context}\n\n"
        f"Вопрос пользователя: {user_query}\n\n"
        f"Сформулируй ответ, используя только приведённые фрагменты, со ссылками на номера "
        f"источников вида [1], [2] там, где это уместно."
    )

    return generate_structured(
        system_prompt=settings.answer_system_prompt,
        user_prompt=user_prompt,
        response_model=GeneratedAnswer,
        model=settings.llm_model,
        temperature=settings.answer_temperature,
    )
