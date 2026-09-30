from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from immune.config.spec import QuestionSpec
from immune.errors import SensorError, SensorUnavailable
from immune.sensing.quota import SensorShed
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading
from immune.state.backend import StateBackend
from immune.state.codec import ReadingCodec


@dataclass(frozen=True, slots=True)
class BreakerPolicy:
    error_rate: float = 0.2
    window_s: float = 60.0
    cooldown_s: float = 30.0
    minimum_calls: int = 5


class CircuitBreaker:
    def __init__(self, policy: BreakerPolicy, clock: Callable[[], float] = time.monotonic) -> None:
        self._policy = policy
        self._clock = clock
        self._outcomes: deque[tuple[float, bool]] = deque()
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if self._clock() - self._opened_at >= self._policy.cooldown_s:
                self._opened_at = None
                self._outcomes.clear()
                return False
            return True

    def record(self, success: bool) -> None:
        with self._lock:
            now = self._clock()
            self._outcomes.append((now, success))
            while self._outcomes and now - self._outcomes[0][0] > self._policy.window_s:
                self._outcomes.popleft()
            failures = sum(1 for _, outcome in self._outcomes if not outcome)
            if (
                len(self._outcomes) >= self._policy.minimum_calls
                and failures / len(self._outcomes) > self._policy.error_rate
            ):
                self._opened_at = now


class ReadingCache:
    def __init__(self, capacity: int = 4096) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[str, SensorReading] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> str:
        described = [
            (question.key, question.kind, question.text, question.options, question.levels) for question in questions
        ]
        payload = json.dumps({"state": state, "questions": described}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> SensorReading | None:
        with self._lock:
            reading = self._entries.get(key)
            if reading is not None:
                self._entries.move_to_end(key)
            return reading

    def put(self, key: str, reading: SensorReading) -> None:
        with self._lock:
            self._entries[key] = reading
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)


class SharedReadingCache(ReadingCache):
    def __init__(self, backend: StateBackend, ttl_s: float = 3_600.0) -> None:
        super().__init__(capacity=1024)
        self._backend = backend
        self._ttl_s = ttl_s

    def get(self, key: str) -> SensorReading | None:
        local = super().get(key)
        if local is not None:
            return local
        raw = self._backend.get(f"reading:{key}")
        if raw is None:
            return None
        reading = ReadingCodec.decode(json.loads(raw))
        super().put(key, reading)
        return reading

    def put(self, key: str, reading: SensorReading) -> None:
        super().put(key, reading)
        self._backend.put(f"reading:{key}", json.dumps(ReadingCodec.encode(reading)).encode("utf-8"), self._ttl_s)


class GuardedSensor(Sensor):
    name: ClassVar[str] = "guarded"

    def __init__(
        self, inner: Sensor, timeout_s: float, breaker: CircuitBreaker | None = None, cache: ReadingCache | None = None
    ) -> None:
        self.inner = inner
        self._timeout_s = timeout_s
        self._breaker = breaker or CircuitBreaker(BreakerPolicy())
        self._cache = cache or ReadingCache()

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        if not questions:
            return SensorReading.empty(self.inner.name)
        key = ReadingCache.key(state, questions)
        cached = self._cache.get(key)
        if cached is not None:
            return SensorReading(signals=cached.signals, source=cached.source, model=cached.model, calls=0)
        if self._breaker.is_open:
            raise SensorUnavailable("sensor circuit breaker is open")
        try:
            reading = await asyncio.wait_for(self.inner.read(state, questions), timeout=self._timeout_s)
        except SensorShed:
            raise
        except TimeoutError as error:
            self._breaker.record(success=False)
            raise SensorUnavailable(f"sensor exceeded {self._timeout_s:.2f}s") from error
        except SensorError:
            self._breaker.record(success=False)
            raise
        except Exception as error:
            self._breaker.record(success=False)
            raise SensorUnavailable(f"sensor failed: {error}") from error
        self._breaker.record(success=True)
        self._cache.put(key, reading)
        return reading

    async def close(self) -> None:
        await self.inner.close()

    def reset(self) -> None:
        self.inner.reset()
