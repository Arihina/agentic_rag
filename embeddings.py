from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import settings


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_query(text: str) -> list[float]:
    vector = get_embedder().encode(settings.query_prefix +
                                   text, normalize_embeddings=True)
    return vector.tolist()


def embed_passage(text: str) -> list[float]:
    vector = get_embedder().encode(settings.passage_prefix +
                                   text, normalize_embeddings=True)
    return vector.tolist()
