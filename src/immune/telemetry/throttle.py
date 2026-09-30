from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable


class ThrottledLog:
    def __init__(self, logger: logging.Logger, interval_s: float = 10.0, clock: Callable[[], float] = time.monotonic):
        self._logger = logger
        self._interval_s = interval_s
        self._clock = clock
        self._last: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}
        self._lock = threading.Lock()

    def warning(self, key: str, message: str, *args: object) -> None:
        now = self._clock()
        with self._lock:
            if now - self._last.get(key, float("-inf")) < self._interval_s:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                return
            self._last[key] = now
            suppressed = self._suppressed.pop(key, 0)
        suffix = f" ({suppressed} similar messages suppressed)" if suppressed else ""
        self._logger.warning(message + suffix, *args)
