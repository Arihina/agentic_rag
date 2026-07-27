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

    # ollama_model: str = "gemma2:2b"
    # ollama_host: str = "localhost"
    # ollama_port: int = 11434


settings = Settings()
