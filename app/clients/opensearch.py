from __future__ import annotations

"""AsyncOpenSearch — только чтение kb-v2.

Индекс наполняется ingestion, наш сервис его никогда не создаёт и не
изменяет; проверка ping при старте — единственная запись, которую он себе
позволяет (и то через http).
"""

from opensearchpy import AsyncOpenSearch

from app.config import settings


def make_opensearch_client() -> AsyncOpenSearch:
    auth = (
        (settings.opensearch_user, settings.opensearch_password)
        if settings.opensearch_user
        else None
    )
    return AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=auth,
        verify_certs=bool(auth),
        timeout=settings.opensearch_timeout,
    )
