from __future__ import annotations

"""Асинхронный клиент к Ollama с retry-loop и обязательным num_ctx.

Разложение по ролям (rewriter/multi_query/eval/answer) — не здесь: клиент
не знает, какую роль он обслуживает, только модель + опции. Роли живут в
core/, они вызывают generate/generate_structured с нужными параметрами.

Инвариант: num_ctx передаётся в options ЯВНО, не через модельный modelfile.
Без этого ollama молча урезает промпт до дефолтных 2048 и первым выкидывает
system-инструкции — молчаливо и без warnings.
"""

import asyncio
import logging
from typing import Any, Type, TypeVar

from ollama import AsyncClient, ResponseError
from pydantic import BaseModel, ValidationError

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Все попытки исчерпаны или структура ответа не восстанавливается."""


class LLMClient:
    """Тонкая обёртка над ollama.AsyncClient.

    Не пул: сам AsyncClient держит httpx.AsyncClient с keepalive внутри,
    один экземпляр на процесс достаточен для агентского цикла с 3-6
    последовательными вызовами.
    """

    def __init__(self):
        self._client = AsyncClient(
            host=settings.ollama_url,
            timeout=settings.ollama_timeout,
        )

    async def close(self) -> None:
        # AsyncClient — обёртка над httpx.AsyncClient, чей внутренний
        # ресурс закрывается вот так.
        inner = getattr(self._client, "_client", None)
        if inner is not None and hasattr(inner, "aclose"):
            await inner.aclose()

    async def ping(self) -> bool:
        try:
            await self._client.list()
            return True
        except Exception:
            logger.warning("ollama недоступен", exc_info=True)
            return False

    async def generate(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        temperature: float,
        num_ctx: int | None = None,
        format: str | dict | None = None,
    ) -> str:
        """Одиночный вызов /api/chat с двумя сообщениями (system + user).

        format="json" — режим JSON-принудительной генерации; используется в
        generate_structured. Для свободного ответа — None.
        """
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": num_ctx or settings.llm_num_ctx,
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(1, settings.llm_retry_attempts + 1):
            try:
                response = await self._client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    format=format,
                    stream=False,
                )
                return response["message"]["content"]
            except ResponseError as e:
                last_error = e
                logger.warning(
                    "ollama ResponseError на попытке %d/%d: %s",
                    attempt, settings.llm_retry_attempts, e)
            except Exception as e:
                last_error = e
                logger.warning(
                    "ollama сбой на попытке %d/%d: %s",
                    attempt, settings.llm_retry_attempts, e)

            if attempt < settings.llm_retry_attempts:
                await asyncio.sleep(settings.llm_retry_backoff * attempt)

        raise LLMError(f"ollama не отвечает после "
                       f"{settings.llm_retry_attempts} попыток: {last_error}")

    async def generate_structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        response_model: Type[T],
        temperature: float,
        num_ctx: int | None = None,
    ) -> T:
        """format="json" + валидация в pydantic; при невалидном JSON — повтор
        внутри того же retry-бюджета."""
        schema_hint = response_model.model_json_schema()
        raw = await self.generate(
            model=model,
            system=system + f"\n\nОтвет строго в JSON по схеме: {schema_hint}",
            prompt=prompt,
            temperature=temperature,
            num_ctx=num_ctx,
            format="json",
        )
        try:
            return response_model.model_validate_json(raw)
        except ValidationError as e:
            raise LLMError(
                f"ollama вернула невалидный JSON для {response_model.__name__}: "
                f"{e}\nСырой ответ: {raw[:500]}")
