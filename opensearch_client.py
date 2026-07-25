from opensearchpy import OpenSearch

from config import settings


def get_client() -> OpenSearch:
    """Клиент для локального стенда с отключённым security-плагином"""
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host,
                "port": settings.opensearch_port}],
        http_compress=True,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
    )
