from __future__ import annotations

"""Диагностический прогон одного запроса против живого стека.

Использование:
    python -m app.debug.query --rag-id <uuid> --query "как настроить X"
    python -m app.debug.query --rag-id <uuid> --query "..." \
        --top-k 5 --score-threshold 0.3 --max-iterations 2

Инструмент **не для production**: сам поднимает клиенты (без lifespan
FastAPI), гонит одну run_agent-итерацию, печатает trace как pretty JSON,
закрывает клиенты и выходит. Помогает верифицировать связку с реальными
OpenSearch/Ollama/ingestion без запуска http-слоя и без Postgres.

Живому rag_ingestion должен быть доступен на INGEST_INTERNAL_URL с
поднятым /embed; OpenSearch с загруженным kb-v2; ollama с моделями из
LLM_MODEL_*. Ничего из этого CLI не проверяет — на несоответствие
получите обычные исключения клиентов.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import asdict, is_dataclass

from app.clients.embed import EmbedClient
from app.clients.ingest import IngestClient
from app.clients.llm import LLMClient
from app.clients.opensearch import make_opensearch_client
from app.core.agent import run_agent


def _to_jsonable(obj):
    """Простой fallback-сериализатор для dataclass/set в trace."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"{type(obj).__name__} не сериализуется")


async def _run(args: argparse.Namespace) -> int:
    os_client = make_opensearch_client()
    llm = LLMClient()
    embed = EmbedClient()
    ingest = IngestClient()

    try:
        trace = await run_agent(
            args.rag_id, os_client, llm, embed, args.query,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            max_iterations=args.max_iterations,
            index=args.index,
        )
    finally:
        await asyncio.gather(
            os_client.close(), llm.close(),
            embed.close(), ingest.close(),
            return_exceptions=True,
        )

    print(json.dumps(trace, default=_to_jsonable,
                     ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.debug.query",
        description="Прогон run_agent против живого стека (dev-диагностика).",
    )
    parser.add_argument("--rag-id", required=True,
                        help="UUID набора; обязателен, отсутствие фильтра "
                             "по нему — синтаксически невозможно даже в CLI")
    parser.add_argument("--query", required=True,
                        help="Вопрос пользователя")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Финальный размер пула для answer (default: 10)")
    parser.add_argument("--score-threshold", type=float, default=None,
                        help="Порог косинуса для kNN-ветки (0..1). "
                             "Не задан — без порога")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Верхняя граница итераций цикла; "
                             "None → берётся из settings")
    parser.add_argument("--index", default=None,
                        help="Имя индекса; None → settings.index_name")
    args = parser.parse_args()

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
