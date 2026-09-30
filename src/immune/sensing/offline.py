from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

from immune.config.spec import QuestionSpec
from immune.errors import SensorUnavailable
from immune.sensing.guard import ReadingCache
from immune.sensing.sensor import Sensor
from immune.sensing.signals import DEFANGED_SUFFIX, SensorReading, Signal

ScriptedValue: TypeAlias = "float | str | Mapping[str, float]"
BENIGN_AFFIRMATIVE = frozenset({"task_fidelity", "serves_request", "commitment_supported"})
# Candidate questions (``cand_3__secret_real``) are answered as a Jev that agrees with the reflex that nominated them,
# so offline runs keep the floor. Script a value (``secret_real=0.05``) to model Jev declining. These two keys are
# asked the other way round: "yes" means the user asked for it.
CANDIDATE_BENIGN = frozenset({"destination_requested", "irreversible_requested"})
_CANDIDATE = re.compile(r"^cand_\d+__")
Rule: TypeAlias = "Callable[[Mapping[str, Any], QuestionSpec], ScriptedValue | None]"


class MockSensor(Sensor):
    name: ClassVar[str] = "mock"

    def __init__(
        self,
        signals: Mapping[str, ScriptedValue] | None = None,
        rules: Sequence[Rule] = (),
        default_noul: float = 0.02,
        fail: bool = False,
    ) -> None:
        self._signals = dict(signals or {})
        self._rules = tuple(rules)
        self._default_noul = default_noul
        self._fail = fail
        self.calls: list[tuple[Mapping[str, Any], tuple[str, ...]]] = []

    def script(self, **signals: ScriptedValue) -> MockSensor:
        self._signals.update(signals)
        return self

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        self.calls.append((state, tuple(question.key for question in questions)))
        if self._fail:
            raise SensorUnavailable("mock sensor configured to fail")
        signals = {question.key: self._signal(state, question) for question in questions}
        return SensorReading(signals=signals, source=self.name, model="mock", input_tokens=0)

    def _signal(self, state: Mapping[str, Any], question: QuestionSpec) -> Signal:
        value = self._lookup(state, question)
        if question.kind == "choice":
            return self._choice(question, value)
        probability = float(value) if isinstance(value, (int, float)) else self._default(question.key)
        return Signal(
            key=question.key,
            kind=question.kind,
            probability=probability,
            value=probability if question.kind == "score" else None,
        )

    def _default(self, key: str) -> float:
        generic = self._candidates(key)[-1]
        if _CANDIDATE.match(key):
            return self._default_noul if generic in CANDIDATE_BENIGN else 1 - self._default_noul
        return 1 - self._default_noul if generic in BENIGN_AFFIRMATIVE else self._default_noul

    def _lookup(self, state: Mapping[str, Any], question: QuestionSpec) -> ScriptedValue | None:
        for rule in self._rules:
            value = rule(state, question)
            if value is not None:
                return value
        for candidate in self._candidates(question.key):
            if candidate in self._signals:
                return self._signals[candidate]
        return None

    @staticmethod
    def _candidates(key: str) -> tuple[str, ...]:
        base = key.removesuffix(DEFANGED_SUFFIX)
        generic = base.split("__", 1)[1] if "__" in base else base
        return (key, base, generic)

    @staticmethod
    def _choice(question: QuestionSpec, value: ScriptedValue | None) -> Signal:
        options = list(question.options)
        if isinstance(value, Mapping):
            distribution = {option: float(value.get(option, 0.0)) for option in options}
        else:
            chosen = (
                value if isinstance(value, str) and value in options else ("none" if "none" in options else options[0])
            )
            spread = 0.1 / max(len(options) - 1, 1)
            distribution = {option: (0.9 if option == chosen else spread) for option in options}
        top = max(distribution, key=lambda option: distribution[option])
        return Signal(
            key=question.key,
            kind="choice",
            probability=distribution[top],
            distribution=distribution,
            value=top,
            confidence=distribution[top],
        )


class RecordingSensor(Sensor):
    name: ClassVar[str] = "recording"

    def __init__(self, inner: Sensor, path: Path) -> None:
        self._inner = inner
        self._store = _ReadingStore(path)

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        reading = await self._inner.read(state, questions)
        self._store.save(ReadingCache.key(state, questions), reading)
        return reading


class ReplaySensor(Sensor):
    name: ClassVar[str] = "replay"

    def __init__(self, path: Path) -> None:
        self._store = _ReadingStore(path)

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        reading = self._store.load(ReadingCache.key(state, questions))
        if reading is None:
            raise SensorUnavailable("no recorded reading for this request")
        return reading


class _ReadingStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self._entries[record["key"]] = record

    def save(self, key: str, reading: SensorReading) -> None:
        record = {
            "key": key,
            "source": reading.source,
            "model": reading.model,
            "signals": {
                name: {
                    "kind": signal.kind,
                    "probability": signal.probability,
                    "distribution": dict(signal.distribution),
                    "value": signal.value,
                    "confidence": signal.confidence,
                }
                for name, signal in reading.signals.items()
            },
        }
        with self._lock:
            self._entries[key] = record
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")

    def load(self, key: str) -> SensorReading | None:
        record = self._entries.get(key)
        if record is None:
            return None
        signals = {name: Signal(key=name, **fields) for name, fields in record["signals"].items()}
        return SensorReading(signals=signals, source="replay", model=record.get("model"))
