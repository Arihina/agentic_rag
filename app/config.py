from __future__ import annotations

"""Конфигурация сервиса. Префикс AGENTIC_RAG_, файл agentic_rag.env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_RAG_",
        env_file="agentic_rag.env",
        extra="ignore",
    )

    # --- сеть -------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8020
    reload: bool = False
    log_level: str = "info"
    timeout_keep_alive: int = 300

    # --- OpenSearch (только чтение) ---------------------------------------
    opensearch_url: str = "http://localhost:9200"
    opensearch_user: str | None = None
    opensearch_password: str | None = None
    index_name: str = "kb-v2"
    opensearch_timeout: float = 30.0

    # --- LLM (Ollama) -----------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    ollama_timeout: float = 300.0
    # Три роли одной ЛЛМ. По арх-документу это одна модель, но семантически
    # они разные, а промпты и температура настраиваются отдельно, поэтому
    # держим три поля — на случай если понадобится развести (например, лёгкая
    # модель для rewriter, серьёзная для answer).
    llm_model_rewriter: str = "qwen3:8b"
    llm_model_multi_query: str = "qwen3:8b"
    llm_model_eval: str = "qwen3:8b"
    llm_model_answer: str = "qwen3:8b"

    # ollama context window. Явно передаётся в каждый вызов — без этого
    # ollama молча урезает промпт до 2048 и первым выкидывает system.
    llm_num_ctx: int = 8192

    # Температуры rewriter/eval/multi_query — из .env; для answer температура
    # приходит из конфига набора (rag_set.temperature), эти поля здесь не
    # заводим, чтобы случайно не перекрыть пользовательскую настройку.
    llm_temperature_rewriter: float = 0.0
    llm_temperature_eval: float = 0.0
    llm_temperature_multi_query: float = 0.7

    llm_retry_attempts: int = 3
    llm_retry_backoff: float = 1.0

    # --- ingestion client -------------------------------------------------
    # /embed и /v1/internal/* — на внутреннем порту 8012 (не 8011).
    ingest_internal_url: str = "http://127.0.0.1:8012"
    ingest_timeout: float = 30.0

    # --- поиск (значения от сервиса, не от пользователя) ------------------
    # final_top_k — из конфига набора (rag_set.top_k). Здесь только глубина
    # каждой отдельной ветки до фьюжна.
    bm25_top_k: int = 50
    knn_top_k: int = 50
    sparse_top_k: int = 50

    # Множитель для rewritten-ветки в RRF-фьюжне. Rewrite ближе к тому, что
    # реально спросил пользователь; варианты multi-query «уплывают» ради
    # recall — доверять им наравне с rewritten нельзя.
    primary_weight: float = 2.0

    # --- multi-query ------------------------------------------------------
    multi_query_variants_count: int = 3

    # --- агентский цикл ---------------------------------------------------
    max_iterations: int = 3

    # --- история чата -----------------------------------------------------
    # Скользящее окно по токенам HF-токенайзером модели.
    tokenizer_repo: str = "Qwen/Qwen3-8B"
    history_token_limit: int = 6000
    # truncate локально / strict в проде: сколько сообщений тихо обрезать,
    # прежде чем сдаться и вернуть 400.
    history_overflow: str = "truncate"

    # --- SSE --------------------------------------------------------------
    sse_heartbeat_interval: float = 15.0

    # --- Postgres ---------------------------------------------------------
    # Появится в 2.3; в 2.1 не подключается, но поле заведено, чтобы
    # .env.example был исчерпывающим.
    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/agentic_rag"
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # --- системные промпты (заглушки, переопределяются через .env) --------
    # Реальные промпты живут отдельными многострочными записями в .env,
    # чтобы редактировать их без пересборки образа.
    answer_system_prompt: str = (
        "Ты — ассистент по документации. Отвечай строго по предоставленным "
        "источникам, ссылайся на них в виде [1], [2]. Если сведений нет — "
        "так и скажи, ничего не додумывай.")
    rewriter_system_prompt: str = (
        "Ты переформулируешь вопрос пользователя в самодостаточный поисковый "
        "запрос, учитывая контекст диалога.")
    eval_system_prompt: str = (
        "Оцени, достаточно ли найденных фрагментов для ответа на вопрос. "
        "Верни JSON: {sufficient: bool, missing_topics: [str]}.")
    multi_query_system_prompt_template: str = (
        "Сформулируй {n} альтернативных поисковых запросов, покрывающих ту "
        "же тему разными формулировками. Верни JSON: {{variants: [str]}}.")


settings = Settings()
