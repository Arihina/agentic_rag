from __future__ import annotations

"""Обёртка async-генератора событий с heartbeat.

Ключевые моменты реализации:

- **asyncio.wait с timeout, а не wait_for(shield(...))**. wait_for на
  timeout ОТМЕНЯЕТ задачу — shield нужен, чтобы этого не произошло. Но
  через `wait` тот же эффект достигается проще: task живёт между
  итерациями цикла, мы просто периодически заглядываем, готов ли он.

- **Task создаётся ОДИН раз на item.** __anext__ — это корутина; чтобы её
  можно было await'ить в нескольких итерациях (жди 15 сек, ping, жди
  ещё 15), нужен Task. Пересоздаём только когда предыдущий готов и
  результат обработан.

- **Cancel-safety в finally.** Если клиент отвалился, StreamingResponse
  бросает CancelledError по всему стеку. Наш `finally` гарантирует, что:
    1. висящий task на __anext__ отменён (иначе event loop будет ждать
       его завершения при закрытии);
    2. source (генератор из run_turn) закрыт через aclose() — это
       поднимет GeneratorExit внутри run_turn, где try/finally запишет
       'client_disconnected' в БД (см. 2.4.d).
"""

import asyncio
import logging
from typing import AsyncIterator

from app.sse.events import HEARTBEAT

logger = logging.getLogger(__name__)


async def with_heartbeat(
    source: AsyncIterator[bytes],
    interval: float,
) -> AsyncIterator[bytes]:
    """Транслирует байты из source; если source молчит дольше interval,
    вставляет HEARTBEAT.

    source — уже готовые SSE-frames (bytes), не Event-объекты.
    """

    async_iter = source.__aiter__()
    task: asyncio.Task[bytes] | None = None
    try:
        while True:
            if task is None:
                task = asyncio.create_task(async_iter.__anext__())

            done, _pending = await asyncio.wait({task}, timeout=interval)

            if task in done:
                try:
                    item = task.result()
                except StopAsyncIteration:
                    return
                except Exception:
                    logger.exception("исходный SSE-генератор упал")
                    raise
                finally:
                    task = None
                yield item
            else:
                yield HEARTBEAT
    finally:
        if task is not None and not task.done():
            task.cancel()
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:
                logger.debug("aclose(source) исключение", exc_info=True)
