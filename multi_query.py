from pydantic import BaseModel, Field

from config import settings
from llm_client import generate_structured


class QueryVariants(BaseModel):
    variants: list[str] = Field(
        description="Альтернативные формулировки поискового запроса")


def generate_query_variants(query: str, n: int | None = None) -> list[str]:
    n = n or settings.multi_query_variants_count

    result = generate_structured(
        system_prompt=settings.multi_query_system_prompt_template.format(n=n),
        user_prompt=f"Исходный запрос: {query}",
        response_model=QueryVariants,
        model=settings.llm_model,
        temperature=settings.multi_query_temperature,
    )
    
    return result.variants[:n]
