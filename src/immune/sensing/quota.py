from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar

from immune.config.spec import QuestionSpec
from immune.errors import SensorUnavailable
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading

_DEFAULT_KEY = ""


class RequestPriority(IntEnum):
    OBSERVED = 0
    ENFORCED = 1
    FLOOR = 2


class SensorShed(SensorUnavailable):
    pass


_PRIORITY: contextvars.ContextVar[RequestPriority] = contextvars.ContextVar(
    "immune_request_priority", default=RequestPriority.FLOOR
)
_API_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar("immune_api_key", default=None)


class SensingContext:
    @staticmethod
    @contextlib.contextmanager
    def priority(priority: RequestPriority) -> Iterator[None]:
        token = _PRIORITY.set(priority)
        try:
            yield
        finally:
            _PRIORITY.reset(token)

    @staticmethod
    def current_priority() -> RequestPriority:
        return _PRIORITY.get()

    @staticmethod
    def api_key() -> str | None:
        return _API_KEY.get()


class TokenBucket:
    def __init__(self, per_minute: float, burst_s: float = 10.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._rate = per_minute / 60.0
        self._capacity = max(1.0, self._rate * burst_s)
        self._tokens = self._capacity
        self._clock = clock
        self._updated = clock()

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens

    def take(self) -> bool:
        self._refill()
        if self._tokens < 1.0:
            return False
        self._tokens -= 1.0
        return True

    def seconds_until_available(self) -> float:
        self._refill()
        return 0.0 if self._tokens >= 1.0 else (1.0 - self._tokens) / self._rate

    def _refill(self) -> None:
        now = self._clock()
        self._tokens = min(self._capacity, self._tokens + (now - self._updated) * self._rate)
        self._updated = now


@dataclass(frozen=True, slots=True)
class QuotaPolicy:
    observed_below: float = 0.7
    enforced_below: float = 0.9
    floor_wait_s: float = 0.15


class QuotaGovernor:
    def __init__(
        self,
        keys: Sequence[str | None],
        per_minute: float,
        policy: QuotaPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._buckets = {key or _DEFAULT_KEY: TokenBucket(per_minute, clock=clock) for key in (keys or [None])}
        self._policy = policy or QuotaPolicy()
        self._lock = threading.Lock()
        self.admitted: Counter[str] = Counter()
        self.shed: Counter[str] = Counter()

    @property
    def utilization(self) -> float:
        with self._lock:
            capacity = sum(bucket.capacity for bucket in self._buckets.values())
            available = sum(bucket.available for bucket in self._buckets.values())
        return 1.0 - available / capacity

    async def admit(self, priority: RequestPriority) -> str | None:
        if not self._allowed(priority):
            self.shed[priority.name.lower()] += 1
            return None
        key = self._take()
        if key is None and priority is RequestPriority.FLOOR:
            await asyncio.sleep(min(self._policy.floor_wait_s, self._shortest_wait()))
            key = self._take()
        if key is None:
            self.shed[priority.name.lower()] += 1
            return None
        self.admitted[priority.name.lower()] += 1
        return key

    def _allowed(self, priority: RequestPriority) -> bool:
        utilization = self.utilization
        if priority is RequestPriority.OBSERVED:
            return utilization < self._policy.observed_below
        if priority is RequestPriority.ENFORCED:
            return utilization < self._policy.enforced_below
        return True

    def _take(self) -> str | None:
        with self._lock:
            key, bucket = max(self._buckets.items(), key=lambda item: item[1].available)
            return key if bucket.take() else None

    def _shortest_wait(self) -> float:
        with self._lock:
            return min(bucket.seconds_until_available() for bucket in self._buckets.values())


class QuotedSensor(Sensor):
    name: ClassVar[str] = "quoted"

    def __init__(self, inner: Sensor, governor: QuotaGovernor) -> None:
        self._inner = inner
        self._governor = governor

    @property
    def governor(self) -> QuotaGovernor:
        return self._governor

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        priority = SensingContext.current_priority()
        key = await self._governor.admit(priority)
        if key is None:
            raise SensorShed(f"sensor quota exhausted; {priority.name.lower()} request shed")
        token = _API_KEY.set(key or None)
        try:
            return await self._inner.read(state, questions)
        finally:
            _API_KEY.reset(token)

    async def close(self) -> None:
        await self._inner.close()

    def reset(self) -> None:
        self._inner.reset()
