from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")
_LOGGER = logging.getLogger("immune")
_PING_TIMEOUT_S = 0.5


class SensorLoop:
    def __init__(self, name: str = "immune-sensor") -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def submit(self, coroutine: Coroutine[Any, Any, T]) -> concurrent.futures.Future[T]:
        return asyncio.run_coroutine_threadsafe(coroutine, self._ensure())

    async def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        loop = self._ensure()
        if asyncio.get_running_loop() is loop:
            return await coroutine
        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(coroutine, loop))

    def run_sync(self, coroutine: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
        return self.submit(coroutine).result(timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def responsive(self, timeout: float = _PING_TIMEOUT_S) -> bool:
        loop = self._loop
        if loop is None or not self.running:
            return False
        answered = threading.Event()
        try:
            loop.call_soon_threadsafe(answered.set)
        except RuntimeError:
            return False
        return answered.wait(timeout)

    def recover(self) -> bool:
        if self.responsive():
            return False
        with self._lock:
            abandoned = self._loop
            self._loop, self._thread = None, None
            self._generation += 1
        if abandoned is not None:
            abandoned.call_soon_threadsafe(abandoned.stop)
        _LOGGER.error("immune: sensor loop stopped responding; started a new one")
        return True

    def close(self, finalizer: Coroutine[Any, Any, Any] | None = None) -> None:
        with self._lock:
            loop, thread = self._loop, self._thread
            self._loop, self._thread = None, None
        if loop is None or thread is None:
            if finalizer is not None:
                finalizer.close()
            return
        if finalizer is not None:
            try:
                asyncio.run_coroutine_threadsafe(finalizer, loop).result(timeout=5)
            except (concurrent.futures.TimeoutError, RuntimeError):
                _LOGGER.warning("immune: sensor did not close within 5s")
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        if not thread.is_alive():
            loop.close()

    def _ensure(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._thread is None or not self._thread.is_alive():
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=self._serve, args=(loop,), name=self._name, daemon=True)
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    @staticmethod
    def _serve(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()
