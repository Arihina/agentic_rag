from dataclasses import dataclass, field

from opensearchpy import OpenSearch

from answer import GeneratedAnswer, generate_answer
from config import settings
from evaluation import EvalResult, evaluate
from hybrid_search import multi_query_hybrid_search
from multi_query import generate_query_variants
from rewriter import rewrite_query


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
    stopped_reason: str = ""  # "sufficient" | "diminishing_returns" | "max_iterations"


def _overlap_ratio(new_ids: set[str], existing_ids: set[str]) -> float:
    if not new_ids:
        return 1.0
    return len(new_ids & existing_ids) / len(new_ids)


def run_agent(
    client: OpenSearch,
    user_query: str,
    history: list[dict[str, str]] | None = None,
    top_k: int = 10,
    max_iterations: int | None = None,
) -> AgentTrace:
    max_iterations = max_iterations or settings.max_iterations
    history = history or []

    rewritten = rewrite_query(history, user_query)
    trace = AgentTrace(rewritten_query=rewritten)

    pool: dict[str, dict] = {}
    queries = [rewritten, *generate_query_variants(rewritten)]

    for iteration in range(1, max_iterations + 1):
        new_results = multi_query_hybrid_search(
            client, queries, final_top_k=top_k)
        new_ids = {hit["_id"] for hit in new_results}
        existing_ids = set(pool.keys())
        overlap = _overlap_ratio(
            new_ids, existing_ids) if iteration > 1 else 0.0

        for hit in new_results:
            pool.setdefault(hit["_id"], hit)

        eval_result = evaluate(user_query, list(pool.values()))
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
        if iteration > 1 and overlap >= settings.early_stop_overlap_ratio:
            trace.stopped_reason = "diminishing_returns"
            break
        if iteration == max_iterations:
            trace.stopped_reason = "max_iterations"
            break

        queries = eval_result.next_queries if eval_result.next_queries else [
            user_query]

    trace.final_chunks = list(pool.values())
    trace.answer = generate_answer(user_query, trace.final_chunks, history)

    return trace
