from __future__ import annotations

"""Тесты with_heartbeat.

Ключевые сценарии:
- passthrough: если source достаточно быстрый, heartbeat вообще не идёт;
- silence: если source долго молчит между item'ами, между ними появляется
  HEARTBEAT в правильном месте;
- StopAsyncIteration: source закончился — wrapper тоже завершается;
- cancel: клиент отвалился — source закрывается через aclose (иначе
  run_turn в 2.4.d не запишет client_disconnected в БД).
"""

import asyncio
import unittest

from app.sse.events import HEARTBEAT
from app.sse.stream import with_heartbeat


async def _fast_source(items: list[bytes], delay: float = 0.0):
    """Async-генератор: отдаёт items с задержкой delay между каждым."""
    for item in items:
        if delay:
            await asyncio.sleep(delay)
        yield item


class TrackingSource:
    """Генератор с флагом закрытия — чтобы проверить, вызвал ли wrapper
    aclose() при cancel'е."""

    def __init__(self, items: list[bytes], delay: float = 0.0):
        self.items = list(items)
        self.delay = delay
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if not self.items:
            raise StopAsyncIteration
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.items.pop(0)

    async def aclose(self):
        self.closed = True


class HeartbeatPassthroughTests(unittest.IsolatedAsyncioTestCase):

    async def test_fast_source_no_heartbeat(self):
        """Если source отдаёт item'ы быстрее interval, heartbeat не нужен."""
        source = _fast_source([b"a", b"b", b"c"], delay=0.0)
        result = [chunk async for chunk in with_heartbeat(source, 1.0)]
        self.assertEqual(result, [b"a", b"b", b"c"])

    async def test_empty_source_terminates(self):
        """Пустой source → wrapper тоже сразу завершается."""
        source = _fast_source([], delay=0.0)
        result = [chunk async for chunk in with_heartbeat(source, 1.0)]
        self.assertEqual(result, [])


class HeartbeatTimingTests(unittest.IsolatedAsyncioTestCase):

    async def test_silence_inserts_heartbeat(self):
        """Source молчит 0.15 сек между item'ами, interval=0.05 сек —
        значит между item'ами должен появиться HEARTBEAT (хотя бы один).

        Точное число heartbeat'ов не проверяем: планировщик asyncio плавает,
        и жёсткая проверка `== 2 ping'а` может флейкать под нагрузкой."""
        source = _fast_source([b"first", b"second"], delay=0.15)
        result = [chunk async for chunk in with_heartbeat(source, 0.05)]

        self.assertIn(b"first", result)
        self.assertIn(b"second", result)
        self.assertIn(HEARTBEAT, result,
                      "минимум один heartbeat должен быть между item'ами")
        # Порядок: сначала любое количество HEARTBEAT, потом first,
        # снова любое количество HEARTBEAT, потом second.
        first_idx = result.index(b"first")
        second_idx = result.index(b"second")
        self.assertLess(first_idx, second_idx)
        # Между first и second — только heartbeat (или ничего).
        between = result[first_idx + 1:second_idx]
        self.assertTrue(all(item == HEARTBEAT for item in between))

    async def test_heartbeat_never_replaces_item(self):
        """HEARTBEAT — ДОБАВЛЯЕТСЯ, а не заменяет реальные события.
        Все item'ы source обязаны дойти до потребителя, независимо от
        того, сколько heartbeat'ов между ними."""
        items = [b"x", b"y", b"z"]
        source = _fast_source(items, delay=0.1)
        result = [chunk async for chunk in with_heartbeat(source, 0.03)]
        real_items = [c for c in result if c != HEARTBEAT]
        self.assertEqual(real_items, items)


class HeartbeatCancelSafetyTests(unittest.IsolatedAsyncioTestCase):

    async def test_cancel_closes_source(self):
        """Клиент отвалился → wrapper обязан вызвать source.aclose(), чтобы
        run_turn (в 2.4.d) увидел GeneratorExit и записал
        client_disconnected. Без этого в БД остаются pending-сообщения."""
        # Задержка гарантирует, что при cancel'е task ещё не готов.
        source = TrackingSource([b"never_delivered"], delay=10.0)

        async def _consume():
            async for _ in with_heartbeat(source, 0.05):
                pass

        task = asyncio.create_task(_consume())
        # Ждём, чтобы wrapper успел войти в цикл и создать task на __anext__.
        await asyncio.sleep(0.1)
        task.cancel()
        # Дожидаемся, чтобы finally отработал.
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(source.closed,
                        "wrapper обязан закрыть source при cancel — иначе "
                        "run_turn не увидит client_disconnected")

    async def test_normal_exit_also_closes_source(self):
        """Регрессия: даже при обычном завершении (source опустошился)
        aclose должен быть вызван. Иначе source-специфичные ресурсы
        (курсоры БД, httpx-стримы) остаются висеть."""
        source = TrackingSource([b"a", b"b"], delay=0.0)
        _ = [chunk async for chunk in with_heartbeat(source, 1.0)]

        # У TrackingSource StopAsyncIteration бросается вручную — а до
        # aclose() дойдём в finally нашего wrapper'а.
        # Асинхронный gc может не сработать сразу; но у нас в finally
        # aclose вызывается синхронно после исчерпания.
        self.assertTrue(source.closed)


if __name__ == "__main__":
    unittest.main()
