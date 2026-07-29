import json

from ollama import Client, RequestError, ResponseError
from pydantic import BaseModel, ValidationError

from config import settings


class LLMError(Exception):
    pass


_client = Client(host=settings.ollama_base_url,
                 timeout=settings.llm_request_timeout)


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return text
    return text[start:end + 1]


def generate_structured(
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    model: str | None = None,
    temperature: float = 0.0,
) -> BaseModel:
    model_name = model or settings.llm_model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = _client.chat(
                model=model_name,
                messages=messages,
                format=response_model.model_json_schema(),
                options={"temperature": temperature},
            )
            
            content = response.message.content
            return response_model.model_validate_json(_extract_json(content))
        except (RequestError, ResponseError, ValidationError, json.JSONDecodeError, AttributeError) as exc:
            last_error = exc
            messages.append({
                "role": "user",
                "content": "Ответ должен быть строго валидным JSON по заданной схеме, без пояснений и markdown.",
            })

    raise LLMError(
        f"Не удалось получить структурированный ответ от '{model_name}' "
        f"после {settings.llm_max_retries + 1} попыток: {last_error}"
    )
