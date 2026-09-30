from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from immune.heads.promotion import WilsonBound
from immune.telemetry.observers import CallObserver, CallStart
from immune.types import Verdict

_LOGGER = logging.getLogger("immune")
_BUCKET_S = 300.0
_PRUNE_EVERY = 1_000


@dataclass(frozen=True, slots=True)
class SpikePolicy:
    window_s: float = 900.0
    baseline_s: float = 7 * 86_400.0
    ratio: float = 5.0
    min_fired: int = 5
    min_baseline_calls: int = 200
    absolute_rate: float = 0.05
    cooldown_s: float = 3_600.0


@dataclass(frozen=True, slots=True)
class Alert:
    site: str
    threat: str
    fired: int
    screened: int
    rate: float
    baseline_rate: float | None
    at: float

    @property
    def message(self) -> str:
        baseline = f"{self.baseline_rate:.2%}" if self.baseline_rate is not None else "no baseline yet"
        return (
            f"{self.threat} fired on {self.fired} of {self.screened} recent calls at {self.site} "
            f"({self.rate:.1%}; baseline {baseline})"
        )


AlertCallback = Callable[[Alert], object]


class SpikeDetector(CallObserver):
    def __init__(self, policy: SpikePolicy | None = None, clock: Callable[[], float] = time.time) -> None:
        self._policy = policy or SpikePolicy()
        self._clock = clock
        self._screened: defaultdict[str, defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._fired: defaultdict[tuple[str, str], defaultdict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._alerted: dict[tuple[str, str], float] = {}
        self._callbacks: list[AlertCallback] = []
        self._lock = threading.Lock()
        self._observed = 0
        self.alerts: list[Alert] = []

    def on_alert(self, callback: AlertCallback) -> None:
        self._callbacks.append(callback)

    def finished(self, call: CallStart, verdict: Verdict) -> None:
        now = self._clock()
        bucket = int(now // _BUCKET_S)
        raised: list[Alert] = []
        with self._lock:
            self._screened[verdict.site][bucket] += 1
            for threat in verdict.threats():
                key = (verdict.site, threat)
                self._fired[key][bucket] += 1
                alert = self._check(key, now)
                if alert is not None:
                    raised.append(alert)
            self._observed += 1
            if self._observed % _PRUNE_EVERY == 0:
                self._prune(bucket)
        for alert in raised:
            self._raise(alert)

    def _check(self, key: tuple[str, str], now: float) -> Alert | None:
        policy = self._policy
        if now - self._alerted.get(key, -math.inf) < policy.cooldown_s:
            return None
        site = key[0]
        current = int((now - policy.window_s) // _BUCKET_S)
        oldest = int((now - policy.baseline_s) // _BUCKET_S)
        fired = sum(count for bucket, count in self._fired[key].items() if bucket > current)
        screened = sum(count for bucket, count in self._screened[site].items() if bucket > current)
        if fired < policy.min_fired or screened == 0:
            return None
        base_fired = sum(count for bucket, count in self._fired[key].items() if oldest < bucket <= current)
        base_screened = sum(count for bucket, count in self._screened[site].items() if oldest < bucket <= current)
        lower = self._lower(fired, screened)
        if base_screened >= policy.min_baseline_calls:
            baseline: float | None = base_fired / base_screened
            spiking = lower > policy.ratio * WilsonBound.upper(base_fired, base_screened)
        else:
            baseline = None
            spiking = lower > policy.absolute_rate
        if not spiking:
            return None
        self._alerted[key] = now
        return Alert(site, key[1], fired, screened, fired / screened, baseline, now)

    def _raise(self, alert: Alert) -> None:
        self.alerts.append(alert)
        _LOGGER.warning("immune: alert: %s", alert.message)
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception:
                _LOGGER.exception("immune: an alert callback failed")

    def _prune(self, bucket: int) -> None:
        oldest = bucket - int(self._policy.baseline_s // _BUCKET_S) - 1
        for counts in (*self._screened.values(), *self._fired.values()):
            for stale in [item for item in counts if item < oldest]:
                del counts[stale]

    @staticmethod
    def _lower(successes: int, trials: int, z: float = 1.96) -> float:
        proportion = successes / trials
        denominator = 1 + z * z / trials
        centre = proportion + z * z / (2 * trials)
        margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        return max(0.0, (centre - margin) / denominator)
