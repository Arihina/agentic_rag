from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore")

    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_use_ssl: bool = False

    index_name: str = Field(default="knowledge_base", alias="OPENSEARCH_INDEX")

    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384

    llm_model: str = "gemma2:2b"
    ollama_base_url: str = "http://localhost:11434"

    llm_request_timeout: float = 360.0
    llm_max_retries: int = 2

    rewriter_temperature: float = 0.0

    multi_query_temperature: float = 0.7
    multi_query_variants_count: int = 3

    answer_temperature: float = 0.3

    rewriter_system_prompt: str = (
        "Ты — модуль переформулировки запросов в RAG-системе технической документации предприятия. "
        "На основе истории диалога и последнего сообщения пользователя сформулируй самодостаточный "
        "поисковый запрос: разверни местоимения и ссылки на предыдущий контекст ('он', 'это', 'у него', "
        "'а по нему', 'а для этого случая') в явные сущности из истории. Сохрани исходный смысл и язык "
        "запроса. Не отвечай на вопрос пользователя и не добавляй ничего от себя — верни только "
        "переформулированный поисковый запрос."
    )

    multi_query_system_prompt_template: str = (
        "Ты — модуль генерации вариантов поискового запроса в RAG-системе технической документации "
        "предприятия. По исходному запросу сгенерируй ровно {n} альтернативных формулировок, которые "
        "ищут ту же информацию, но другими словами: используй синонимы, техническую и разговорную "
        "лексику, разный уровень детализации, переформулировку вопроса в утверждение и наоборот. "
        "Не повторяй исходную формулировку дословно. Пиши на том же языке, что и исходный запрос."
    )

    answer_system_prompt: str = (
        "Ты — ассистент технической поддержки, отвечающий сотрудникам предприятия на основе "
        "внутренней документации. Отвечай только на основе предоставленных фрагментов документации, "
        "ничего не придумывай и не используй знания вне контекста. Если фрагменты не содержат ответа "
        "на вопрос — прямо скажи об этом в поле answer и установи grounded=false, не пытайся угадать "
        "или додумать ответ. Ссылайся на источники в квадратных скобках вида [1], [2], соответствующих "
        "номеру фрагмента в предоставленном контексте. Отвечай на языке вопроса пользователя. "
        "Будь лаконичен и по делу."
    )


settings = Settings()
