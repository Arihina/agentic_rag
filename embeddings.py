from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_query(text: str) -> list[float]:
    """e5-модели ожидают префикс 'query: ' для поисковых запросов."""
    vector = get_embedder().encode(f"query: {text}", normalize_embeddings=True)
    return vector.tolist()


def embed_passage(text: str) -> list[float]:
    """e5-модели ожидают префикс 'passage: ' для индексируемых документов."""
    vector = get_embedder().encode(
        f"passage: {text}", normalize_embeddings=True)
    return vector.tolist()
