from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from immune.telemetry.throttle import ThrottledLog
from immune.types import Verdict

if TYPE_CHECKING:
    from immune.config.spec import QuestionSpec
    from immune.sensing.signals import SensorReading

_LOGGER = logging.getLogger("immune")
_THROTTLED = ThrottledLog(_LOGGER, interval_s=60.0)
VerdictCallback = Callable[[Verdict], object]


@dataclass(frozen=True, slots=True)
class CallStart:
    trace_id: str
    provider: str
    model: str
    started_ns: int = field(default_factory=time.time_ns)


@dataclass(frozen=True, slots=True)
class ToolSummary:
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class CallSummary:
    trace_id: str
    site: str
    session: str | None
    provider: str
    model: str
    user: str = ""
    data: tuple[str, ...] = ()
    reply: str = ""
    tools: tuple[ToolSummary, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SensorExchange:
    trace_id: str
    name: str
    state: Mapping[str, Any]
    questions: tuple[QuestionSpec, ...]
    reading: SensorReading | None
    error: str | None = None


class CallObserver:
    detailed: bool = False

    def started(self, call: CallStart) -> None:
        return None

    def sensed(self, exchange: SensorExchange) -> None:
        return None

    def summarized(self, summary: CallSummary) -> None:
        return None

    def finished(self, call: CallStart, verdict: Verdict) -> None:
        return None

    def close(self) -> None:
        return None


class VerdictCallbacks(CallObserver):
    def __init__(self, capacity: int = 10_000) -> None:
        self._callbacks: list[VerdictCallback] = []
        self._queue: queue.Queue[Verdict | None] = queue.Queue(maxsize=capacity)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.dropped = 0

    def add(self, callback: VerdictCallback) -> Callable[[], None]:
        with self._lock:
            self._callbacks.append(callback)
            if self._thread is None:
                self._thread = threading.Thread(target=self._drain, name="immune-callbacks", daemon=True)
                self._thread.start()

        def remove() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return remove

    def finished(self, call: CallStart, verdict: Verdict) -> None:
        if not self._callbacks:
            return
        try:
            self._queue.put_nowait(verdict)
        except queue.Full:
            self.dropped += 1
            _THROTTLED.warning("callbacks-full", "immune: verdict callback queue is full; dropping verdicts")

    def flush(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

    def close(self) -> None:
        self.flush()
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2)
            self._thread = None

    def _drain(self) -> None:
        while True:
            verdict = self._queue.get()
            if verdict is None:
                return
            with self._lock:
                callbacks = list(self._callbacks)
            for callback in callbacks:
                try:
                    callback(verdict)
                except Exception:
                    _LOGGER.exception("immune: a verdict callback failed")


class Observers(CallObserver):
    def __init__(self, observers: list[CallObserver] | None = None, capacity: int = 50_000) -> None:
        self._observers = list(observers or [])
        self._calls: OrderedDict[str, CallStart] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def add(self, observer: CallObserver) -> None:
        self._observers.append(observer)

    def wants_details(self) -> bool:
        return any(observer.detailed for observer in self._observers)

    def sensed(self, exchange: SensorExchange) -> None:
        for observer in self._observers:
            if observer.detailed:
                self._safely(observer.sensed, exchange)

    def summarized(self, summary: CallSummary) -> None:
        for observer in self._observers:
            if observer.detailed:
                self._safely(observer.summarized, summary)

    def started(self, call: CallStart) -> None:
        with self._lock:
            self._calls[call.trace_id] = call
            while len(self._calls) > self._capacity:
                self._calls.popitem(last=False)
        for observer in self._observers:
            self._safely(observer.started, call)

    def finish(self, verdict: Verdict) -> None:
        with self._lock:
            call = self._calls.pop(verdict.trace_id, None)
        if call is None:
            return
        for observer in self._observers:
            self._safely(observer.finished, call, verdict)

    def close(self) -> None:
        for observer in self._observers:
            self._safely(observer.close)

    @staticmethod
    def _safely(method: Callable[..., None], *args: object) -> None:
        try:
            method(*args)
        except Exception:
            _LOGGER.exception("immune: an observer failed")
