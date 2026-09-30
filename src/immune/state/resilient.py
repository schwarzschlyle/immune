from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TypeVar

from immune.state.backend import Change, Keep, StateBackend

_LOGGER = logging.getLogger("immune")
T = TypeVar("T")


class ResilientBackend(StateBackend):
    shared = True

    def __init__(self, primary: StateBackend, fallback: StateBackend, cooldown_s: float = 30.0) -> None:
        self.name = primary.name
        self._primary = primary
        self._fallback = fallback
        self._cooldown_s = cooldown_s
        self._failed_at: float | None = None
        self._lock = threading.Lock()

    @property
    def degraded(self) -> bool:
        return self._failed_at is not None

    def get(self, key: str) -> bytes | None:
        return self._call(lambda backend: backend.get(key))

    def put(self, key: str, value: bytes, ttl_s: float | None = None) -> None:
        self._call(lambda backend: backend.put(key, value, ttl_s))

    def put_if_absent(self, key: str, value: bytes, ttl_s: float | None = None) -> bool:
        return self._call(lambda backend: backend.put_if_absent(key, value, ttl_s))

    def update(self, key: str, change: Change, ttl_s: float | None = None) -> bytes:
        return self._call(lambda backend: backend.update(key, change, ttl_s))

    def delete(self, key: str) -> None:
        self._call(lambda backend: backend.delete(key))

    def increment(self, key: str, amounts: Mapping[str, int]) -> None:
        self._call(lambda backend: backend.increment(key, amounts))

    def set_fields(self, key: str, values: Mapping[str, int], if_absent: bool = False) -> None:
        self._call(lambda backend: backend.set_fields(key, values, if_absent))

    def fields(self, key: str) -> dict[str, int]:
        return self._call(lambda backend: backend.fields(key))

    def touch(self, key: str, members: Iterable[str], at: float) -> None:
        listed = list(members)
        self._call(lambda backend: backend.touch(key, listed, at))

    def seen(self, key: str, members: Sequence[str], since: float) -> int:
        return self._call(lambda backend: backend.seen(key, members, since))

    def forget_before(self, key: str, before: float) -> None:
        self._call(lambda backend: backend.forget_before(key, before))

    def append(self, key: str, record: bytes) -> None:
        self._call(lambda backend: backend.append(key, record))

    def records(self, key: str) -> list[bytes]:
        return self._call(lambda backend: backend.records(key))

    def take_records(self, key: str, limit: int) -> list[bytes]:
        return self._call(lambda backend: backend.take_records(key, limit))

    def trim_records(self, key: str, keep_last: int) -> int:
        return self._call(lambda backend: backend.trim_records(key, keep_last))

    def filter_records(self, key: str, keep: Keep) -> int:
        return self._call(lambda backend: backend.filter_records(key, keep))

    def try_lock(self, key: str, ttl_s: float) -> bool:
        return self._call(lambda backend: backend.try_lock(key, ttl_s))

    def unlock(self, key: str) -> None:
        self._call(lambda backend: backend.unlock(key))

    def flush(self) -> None:
        self._fallback.flush()
        self._call(lambda backend: backend.flush())

    def close(self) -> None:
        for backend in (self._primary, self._fallback):
            try:
                backend.close()
            except Exception:
                _LOGGER.exception("immune: failed to close the %s state backend", backend.name)

    def _call(self, operation: Callable[[StateBackend], T]) -> T:
        if self._use_primary():
            try:
                return operation(self._primary)
            except Exception as error:
                self._fail(error)
        return operation(self._fallback)

    def _use_primary(self) -> bool:
        with self._lock:
            if self._failed_at is None:
                return True
            if time.monotonic() - self._failed_at >= self._cooldown_s:
                self._failed_at = None
                _LOGGER.warning("immune: retrying the %s state backend", self._primary.name)
                return True
            return False

    def _fail(self, error: Exception) -> None:
        with self._lock:
            first = self._failed_at is None
            self._failed_at = time.monotonic()
        if first:
            _LOGGER.error(
                "immune: %s state backend failed (%s); using in-memory state for %.0fs",
                self._primary.name,
                error,
                self._cooldown_s,
            )
