from __future__ import annotations

"""Тесты /health и /health/ready.

Основной инвариант: до окончания lifespan-подготовки /health/ready — 503,
а /health — 200 со `status=starting`. После — /health возвращает состояние
зависимостей (ok/degraded), а /health/ready — 200.

Все тесты STUB: реальные ollama/opensearch/ingestion не поднимаем.
"""

import unittest

from fastapi.testclient import TestClient

from app.state import state
from tests import base
from tests.support import (
    FakeEmbed, FakeIngest, FakeLLM, FakeOpenSearch, FakeSessionMaker,
)


class HealthEndpointTests(unittest.TestCase):
    """Ставим fake-клиенты в state ДО того, как TestClient откроет lifespan.
    Настоящий lifespan уже проставит поверх этих fake свои реальные клиенты —
    но fake-ы, в отличие от реальных, безопасно закрываются в shutdown
    (у них есть async close без ресурсов), а перед этим на время активного
    теста мы их снова кладём на место через _override."""

    def setUp(self):
        from app import main  # импорт откладываем, чтобы .env читался с env
        self.main = main
        self._originals = {}

    def _override(self, **fakes):
        """Подмена state после того, как TestClient запустил lifespan
        (реальные клиенты уже поставлены — заменяем их fake-ами)."""
        for name, obj in fakes.items():
            self._originals[name] = getattr(state, name, None)
            setattr(state, name, obj)

    def tearDown(self):
        for name, obj in self._originals.items():
            if obj is not None:
                setattr(state, name, obj)

    def test_health_reports_ok_when_all_up(self):
        with TestClient(self.main.app) as client:
            self._override(os_client=FakeOpenSearch(True),
                           llm=FakeLLM(True),
                           embed=FakeEmbed(True),
                           session_maker=FakeSessionMaker(True))
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "ok")
            self.assertTrue(body["ready"])
            self.assertTrue(body["opensearch"])
            self.assertTrue(body["ollama"])
            self.assertTrue(body["ingestion"])
            self.assertTrue(body["database"])

    def test_health_reports_degraded_when_one_down(self):
        with TestClient(self.main.app) as client:
            self._override(os_client=FakeOpenSearch(True),
                           llm=FakeLLM(False),  # ollama лежит
                           embed=FakeEmbed(True),
                           session_maker=FakeSessionMaker(True))
            response = client.get("/health")
            self.assertEqual(response.status_code, 200,
                             "health возвращает 200 даже при degraded — иначе "
                             "kubernetes зарестартит по цепочке")
            body = response.json()
            self.assertEqual(body["status"], "degraded")
            self.assertFalse(body["ollama"])

    def test_health_reports_degraded_when_db_down(self):
        """Регрессия: упавший Postgres тоже должен давать degraded,
        а не невнятный 500 при попытке SELECT 1."""
        with TestClient(self.main.app) as client:
            self._override(os_client=FakeOpenSearch(True),
                           llm=FakeLLM(True),
                           embed=FakeEmbed(True),
                           session_maker=FakeSessionMaker(False))
            response = client.get("/health")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "degraded")
            self.assertFalse(body["database"])

    def test_ready_endpoint_returns_200_after_lifespan(self):
        with TestClient(self.main.app) as client:
            response = client.get("/health/ready")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["ready"])

    def test_ready_endpoint_returns_503_before_lifespan(self):
        """До запуска TestClient (значит без lifespan) /health/ready — 503.
        Тест эмулирует это, сбрасывая state.ready на новом Event."""
        import asyncio
        original = state.ready
        state.ready = asyncio.Event()
        try:
            # Создаём TestClient без входа в контекст: lifespan НЕ запускается.
            client = TestClient(self.main.app)
            response = client.get("/health/ready")
            self.assertEqual(response.status_code, 503)
            self.assertFalse(response.json()["ready"])
        finally:
            state.ready = original


class SettingsSanityTests(unittest.TestCase):
    """Настройки читаются без .env-файла: все обязательные поля имеют
    дефолты. Регрессия — чтобы добавление нового поля без дефолта не
    сломало запуск в CI."""

    def test_settings_load_with_defaults_only(self):
        from app.config import Settings
        s = Settings()  # без агрумегтов = только дефолты и env
        self.assertGreater(s.port, 0)
        self.assertIn("qwen", s.llm_model_answer.lower())
        self.assertGreaterEqual(s.max_iterations, 1)


if __name__ == "__main__":
    unittest.main()
