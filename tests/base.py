from __future__ import annotations

"""Общая инфраструктура тестов.

Два режима: STUB (по умолчанию, без внешних сервисов) и LIVE (подключение
к живым OpenSearch/Ollama/ingestion — по флагу AGENTIC_RAG_TESTS_LIVE=1).
Пока в 2.1 набор тестов маленький, но каркас закладываем сразу — иначе
дописывать будет больнее.
"""

import os
import unittest
import uuid

LIVE = os.environ.get("AGENTIC_RAG_TESTS_LIVE") == "1"
USER_ID = "11111111-1111-1111-1111-111111111111"
OTHER_USER_ID = "22222222-2222-2222-2222-222222222222"
AUTH = {"X-User-Id": USER_ID}
OTHER_AUTH = {"X-User-Id": OTHER_USER_ID}


def live_only(reason: str = "требует живого стека"):
    return unittest.skipUnless(LIVE, reason)


def new_uuid() -> str:
    return str(uuid.uuid4())
