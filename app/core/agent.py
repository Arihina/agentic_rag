from __future__ import annotations

"""Агентский цикл: rewriter → multi_query → search → evaluate → (?повтор).

Цикл останавливается по одной из причин (`stopped_reason`):
  - "sufficient" — eval сказал, что данных достаточно;
  - "diminishing_returns" — новые запросы приносят те же документы;
  - "max_iterations" — исчерпан бюджет итераций.
"""

from dataclasses import dataclass, field

from opensearchpy import AsyncOpenSearch

from app.clients.embed import EmbedClient
from app.clients.llm import LLMClient
from app.config import settings
from app.core.answer import GeneratedAnswer, generate_answer
from app.core.evaluation import EvalResult, evaluate
from app.core.multi_query import generate_query_variants
from app.core.rewriter import rewrite_query
from app.search.hybrid import multi_query_hybrid_search


@dataclass
class IterationLog:
    iteration: int
    queries: list[str]
    new_chunks_found: int
    overlap_with_pool: float
    eval_result: EvalResult


@dataclass
class AgentTrace:
    rewritten_query: str
    iterations: list[IterationLog] = field(default_factory=list)
    final_chunks: list[dict] = field(default_factory=list)
    answer: GeneratedAnswer | None = None
    stopped_reason: str = ""


def _overlap_ratio(new_ids: set[str], existing_ids: set[str]) -> float:
    if not new_ids:
        return 1.0
    return len(new_ids & existing_ids) / len(new_ids)


async def run_agent(
    os_client: AsyncOpenSearch,
    llm: LLMClient,
    embed: EmbedClient,
    user_query: str,
    history: list[dict[str, str]] | None = None,
    *,
    top_k: int = 10,
    max_iterations: int | None = None,
    index: str | None = None,
    answer_system_prompt: str | None = None,
    answer_temperature: float | None = None,
) -> AgentTrace:
    max_iterations = max_iterations or settings.max_iterations
    history = history or []

    rewritten = await rewrite_query(llm, history, user_query)
    trace = AgentTrace(rewritten_query=rewritten)

    pool: dict[str, dict] = {}
    variants = await generate_query_variants(llm, rewritten)
    queries = [rewritten, *variants]

    for iteration in range(1, max_iterations + 1):
        new_results = await multi_query_hybrid_search(
            os_client, embed, queries, index=index, final_top_k=top_k)
        new_ids = {hit["_id"] for hit in new_results}
        existing_ids = set(pool.keys())
        overlap = (_overlap_ratio(new_ids, existing_ids)
                   if iteration > 1 else 0.0)

        for hit in new_results:
            pool.setdefault(hit["_id"], hit)

        eval_result = await evaluate(llm, user_query, list(pool.values()))
        trace.iterations.append(IterationLog(
            iteration=iteration,
            queries=queries,
            new_chunks_found=len(new_ids - existing_ids),
            overlap_with_pool=overlap,
            eval_result=eval_result,
        ))

        if eval_result.sufficient:
            trace.stopped_reason = "sufficient"
            break
        if (iteration > 1
                and overlap >= settings.early_stop_overlap_ratio):
            trace.stopped_reason = "diminishing_returns"
            break
        if iteration == max_iterations:
            trace.stopped_reason = "max_iterations"
            break

        queries = (eval_result.next_queries if eval_result.next_queries
                   else [user_query])

    trace.final_chunks = list(pool.values())
    trace.answer = await generate_answer(
        llm, user_query, trace.final_chunks, history,
        system_prompt=answer_system_prompt,
        temperature=answer_temperature)

    return trace
