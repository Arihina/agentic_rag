from __future__ import annotations

"""Конфигурация сервиса. Префикс AGENTIC_RAG_, файл agentic_rag.env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_RAG_",
        env_file="agentic_rag.env",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8020
    reload: bool = False
    log_level: str = "info"
    timeout_keep_alive: int = 300

    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str | None = None
    opensearch_password: str | None = None
    index_name: str = "kb-v2"
    opensearch_timeout: float = 30.0

    ollama_url: str = "http://localhost:11434"
    ollama_timeout: float = 300.0

    llm_model_rewriter: str = "qwen3:8b"
    llm_model_multi_query: str = "qwen3:8b"
    llm_model_eval: str = "qwen3:8b"
    llm_model_answer: str = "qwen3:8b"

    llm_num_ctx: int = 8192

    llm_temperature_rewriter: float = 0.0
    llm_temperature_eval: float = 0.0
    llm_temperature_multi_query: float = 0.7
    llm_temperature_answer: float = 0.3

    llm_retry_attempts: int = 3
    llm_retry_backoff: float = 1.0

    ingest_internal_url: str = "http://127.0.0.1:8012"
    ingest_timeout: float = 30.0

    bm25_top_k: int = 50
    knn_top_k: int = 50
    sparse_top_k: int = 50

    primary_weight: float = 2.0

    multi_query_variants_count: int = 3

    max_iterations: int = 3
    early_stop_overlap_ratio: float = 0.8


    tokenizer_repo: str = "Qwen/Qwen3-8B"
    history_token_limit: int = 6000
    history_overflow: str = "truncate"

    sse_heartbeat_interval: float = 15.0

    database_url: str = (
        "postgresql+asyncpg://rag:rag@localhost:5437/agentic_rag?ssl=disable")
    db_pool_size: int = 10
    db_max_overflow: int = 20


    answer_system_prompt: str = (
        "Ты — ассистент технической поддержки, отвечающий сотрудникам "
        "предприятия на основе внутренней документации. Отвечай только на "
        "основе предоставленных фрагментов документации, ничего не "
        "придумывай и не используй знания вне контекста. Если фрагменты не "
        "содержат ответа на вопрос — прямо скажи об этом в поле answer и "
        "установи grounded=false, не пытайся угадать или додумать ответ. "
        "Ссылайся на источники в квадратных скобках вида [1], [2], "
        "соответствующих номеру фрагмента в предоставленном контексте. "
        "Отвечай на языке вопроса пользователя. Будь лаконичен и по делу.")

    rewriter_system_prompt: str = (
        "Ты — модуль переформулировки запросов в RAG-системе технической "
        "документации предприятия. На основе истории диалога и последнего "
        "сообщения пользователя сформулируй самодостаточный поисковый "
        "запрос: разверни местоимения и ссылки на предыдущий контекст "
        "('он', 'это', 'у него', 'а по нему', 'а для этого случая') в явные "
        "сущности из истории. Сохрани исходный смысл и язык запроса. Не "
        "отвечай на вопрос пользователя и не добавляй ничего от себя — "
        "верни только переформулированный поисковый запрос.")

    eval_system_prompt: str = (
        "Ты — модуль оценки достаточности контекста в RAG-системе "
        "технической документации предприятия. По вопросу пользователя и "
        "найденным фрагментам документации определи, хватает ли информации "
        "в этих фрагментах, чтобы дать полный и точный ответ. Если хватает "
        "— sufficient=true, остальные поля можно оставить пустыми. Если "
        "информации не хватает или она релевантна лишь частично — "
        "sufficient=false, укажи в missing_aspects конкретно, каких сведений "
        "недостаёт (по существу вопроса, не общими словами), и предложи в "
        "next_queries 2-3 новых поисковых запроса, сформулированных иначе, "
        "чем уже использованные, которые могут найти недостающее. Не "
        "оценивай стиль, орфографию или формат фрагментов — только "
        "фактическую достаточность для ответа на вопрос.")

    multi_query_system_prompt_template: str = (
        "Ты — модуль генерации вариантов поискового запроса в RAG-системе "
        "технической документации предприятия. По исходному запросу "
        "сгенерируй ровно {n} альтернативных формулировок, которые ищут ту "
        "же информацию, но другими словами: используй синонимы, техническую "
        "и разговорную лексику, разный уровень детализации, переформулировку "
        "вопроса в утверждение и наоборот. Не повторяй исходную формулировку "
        "дословно. Пиши на том же языке, что и исходный запрос.")


settings = Settings()
