from __future__ import annotations

"""Асинхронные фейки для STUB-тестов ядра.

Все три фейка совместимы по типу с реальными клиентами по подмножеству
методов, что использует ядро — этого хватает: питон-дак-тайпинг + async.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel

from app.clients.embed import EmbedItem
from app.core.answer import GeneratedAnswer
from app.core.evaluation import EvalResult
from app.core.multi_query import QueryVariants
from app.core.rewriter import RewrittenQuery


@dataclass
class FakeLLM:
    """Отдаёт заранее заготовленные ответы по типу response_model.

    Тест регистрирует калбеки на каждый тип; если калбек не задан,
    возвращает разумный дефолт. Все методы структурно совместимы
    с LLMClient (по подмножеству, что использует ядро)."""
    call_log: list[tuple[str, str]] = field(default_factory=list)
    rewriter: Callable[[str], str] | None = None
    variants: Callable[[str], list[str]] | None = None
    evaluator: Callable[[str, list[dict]], EvalResult] | None = None
    answerer: Callable[[str, list[dict]], GeneratedAnswer] | None = None

    async def close(self) -> None:
        pass

    async def ping(self) -> bool:
        return True

    async def generate(self, *args, **kwargs) -> str:
        raise NotImplementedError(
            "ядро в 2.2 использует только generate_structured")

    async def generate_structured(
        self, *, model: str, system: str, prompt: str,
        response_model: type, temperature: float, num_ctx: int | None = None,
    ):
        self.call_log.append((response_model.__name__, prompt[:80]))
        if response_model is RewrittenQuery:
            text = self.rewriter(prompt) if self.rewriter else "rewritten"
            return RewrittenQuery(rewritten_query=text)
        if response_model is QueryVariants:
            vs = self.variants(prompt) if self.variants else ["v1", "v2"]
            return QueryVariants(variants=vs)
        if response_model is EvalResult:
            if self.evaluator:
                return self.evaluator(prompt, [])
            return EvalResult(sufficient=True, reasoning="ok")
        if response_model is GeneratedAnswer:
            if self.answerer:
                return self.answerer(prompt, [])
            return GeneratedAnswer(answer="ответ", grounded=True)
        raise NotImplementedError(response_model.__name__)


@dataclass
class FakeEmbed:
    """Возвращает детерминированные вектора: dense — единичный, sparse —
    один токен «1» с весом 1.0. Пригодно для тестов, где нам важна не
    геометрия, а факт вызова."""
    call_log: list[list[str]] = field(default_factory=list)

    async def close(self) -> None:
        pass

    async def ping(self) -> bool:
        return True

    async def embed(
        self, texts: list[str],
        pool: Literal["query", "ingest"] = "query",
    ) -> list[EmbedItem]:
        self.call_log.append(list(texts))
        return [EmbedItem(dense=[1.0, 0.0, 0.0], sparse={"1": 1.0})
                for _ in texts]


@dataclass
class FakeOpenSearch:
    """Отдаёт preset-ответы на search в порядке вызова. Если preset
    исчерпан — циклически повторяет последний (обычно пустой)."""
    responses: list[list[dict]] = field(default_factory=list)
    call_log: list[dict] = field(default_factory=list)

    async def close(self) -> None:
        pass

    async def ping(self) -> bool:
        return True

    async def search(self, *, index: str, body: dict) -> dict[str, Any]:
        self.call_log.append({"index": index, "body": body})
        if not self.responses:
            hits = []
        elif len(self.responses) == 1:
            hits = self.responses[0]
        else:
            hits = self.responses.pop(0)
        return {"hits": {"hits": hits}}


def hit(
    chunk_id: str,
    content: str = "текст",
    *,
    rag_id: str = "00000000-0000-0000-0000-000000000000",
    document_id: str = "d1",
    chunk_index: int = 0,
    headings: list[str] | None = None,
    pages: list[int] | None = None,
    **extra,
) -> dict:
    """Хелпер: собрать hit в формате kb-v2. `chunk_id` в реальности —
    blake2b-хеш, но нам в тестах важна только его уникальность и
    сопоставление hit["_id"]; вычислять хеш не нужно."""
    src = {
        "rag_id": rag_id,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "content": content,
        "headings": headings if headings is not None else [],
        "pages": pages if pages is not None else [],
        "content_hash": "0" * 32,
    }
    src.update(extra)
    return {"_id": chunk_id, "_score": 1.0, "_source": src}
