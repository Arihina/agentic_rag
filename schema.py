from opensearchpy import OpenSearch

from config import settings

INDEX_MAPPING = {
    "settings": {
        "index": {"knn": True},
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "source_file": {"type": "keyword"},
            "chunk_type": {"type": "keyword"},
            "breadcrumb": {"type": "text"},
            "anchor": {"type": "keyword"},
            "source_format": {"type": "keyword"},
            "content": {
                "type": "text",
                "analyzer": "russian",
            },
            "content_vector": {
                "type": "knn_vector",
                "dimension": settings.embedding_dim,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                    "parameters": {"ef_construction": 128, "m": 24},
                },
            },
        }
    },
}


def create_index(client: OpenSearch, index_name: str = settings.index_name, recreate: bool = False) -> None:
    if client.indices.exists(index=index_name):
        if not recreate:
            print(f"Индекс '{index_name}' уже существует, пропускаю создание.")
            return
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, body=INDEX_MAPPING)
    print(f"Индекс '{index_name}' создан.")
