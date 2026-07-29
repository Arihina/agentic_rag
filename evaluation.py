from pydantic import BaseModel, Field

from config import settings
from context_format import format_context
from llm_client import generate_structured


class EvalResult(BaseModel):
    sufficient: bool = Field(
        description="True, если найденных фрагментов достаточно для полного и точного ответа")
    reasoning: str = Field(
        default="", description="Краткое обоснование решения")
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="Каких конкретно сведений не хватает в контексте; пусто, если sufficient=true",
    )
    next_queries: list[str] = Field(
        default_factory=list,
        description="Новые поисковые запросы для следующей итерации; пусто, если sufficient=true",
    )


def evaluate(query: str, chunks: list[dict]) -> EvalResult:
    """
    Eval + reflection одним вызовом: не только 'хватает ли данных', но сразу и
    'чего не хватает' + 'что искать дальше' — чтобы решение не расходилось между
    двумя отдельными LLM-вызовами (см. обсуждение архитектуры).
    """
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
        f"Оцени, достаточно ли этих фрагментов, чтобы полно и точно ответить на вопрос."
    )

    return generate_structured(
        system_prompt=settings.eval_system_prompt,
        user_prompt=user_prompt,
        response_model=EvalResult,
        model=settings.llm_model,
        temperature=settings.eval_temperature,
    )
