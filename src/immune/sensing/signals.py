from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

SignalKind = Literal["noul", "choice", "score"]
DEFANGED_SUFFIX = "__defanged"


@dataclass(frozen=True, slots=True)
class Signal:
    key: str
    kind: SignalKind
    probability: float
    distribution: Mapping[str, float] = field(default_factory=dict)
    value: str | float | None = None
    confidence: float | None = None

    def probability_of(self, options: Iterable[str]) -> float:
        selected = tuple(options)
        if not selected or self.kind != "choice":
            return self.probability
        return min(1.0, sum(self.distribution.get(option, 0.0) for option in selected))


@dataclass(frozen=True, slots=True)
class SensorReading:
    signals: Mapping[str, Signal]
    source: str
    model: str | None = None
    input_tokens: int = 0
    latency_ms: float = 0.0
    calls: int = 1

    @classmethod
    def empty(cls, source: str = "none") -> SensorReading:
        return cls(signals={}, source=source, calls=0)

    @property
    def is_empty(self) -> bool:
        return not self.signals

    def get(self, key: str) -> Signal | None:
        return self.signals.get(key)

    def merged(self, *others: SensorReading) -> SensorReading:
        readings = (self, *others)
        signals: dict[str, Signal] = {}
        for reading in readings:
            signals.update(reading.signals)
        sources = {reading.source for reading in readings if reading.calls}
        return SensorReading(
            signals=signals,
            source=sources.pop() if len(sources) == 1 else ("+".join(sorted(sources)) or self.source),
            model=next((reading.model for reading in readings if reading.model), None),
            input_tokens=sum(reading.input_tokens for reading in readings),
            latency_ms=max((reading.latency_ms for reading in readings), default=0.0),
            calls=sum(reading.calls for reading in readings),
        )

    def scoped(self, prefix: str) -> SensorReading:
        marker = f"{prefix}__"
        signals = {key[len(marker) :]: signal for key, signal in self.signals.items() if key.startswith(marker)}
        return SensorReading(
            signals=signals,
            source=self.source,
            model=self.model,
            input_tokens=self.input_tokens,
            latency_ms=self.latency_ms,
            calls=self.calls,
        )
