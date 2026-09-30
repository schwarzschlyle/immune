from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from immune.core.assess import Assessment, Assessor
from immune.evidence.examples import Example
from immune.heads.model import HeadRegistry
from immune.reflexes.findings import Finding
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading
from immune.testing.fake_provider import FakeReply, FakeToolCall
from immune.testing.harness import ImmuneHarness
from immune.types import Stage

if TYPE_CHECKING:
    from immune.core.candidates import Candidate

SensorFactory = Callable[[Example], Sensor]


@dataclass(frozen=True, slots=True)
class FeatureRow:
    example: str
    source: str
    split: str
    language: str
    threat: str
    vector: tuple[float, ...]
    label: bool


@dataclass(frozen=True, slots=True)
class SignalRow:
    example: str
    split: str
    key: str
    probability: float
    labels: dict[str, bool]


@dataclass(frozen=True, slots=True)
class _Judgement:
    stage: Stage
    reading: SensorReading
    findings: frozenset[str]
    organs: frozenset[str]
    threats: tuple[str, ...] | None = None


class _RecordingAssessor(Assessor):
    def __init__(self, inner: Assessor, sink: list[_Judgement]) -> None:
        super().__init__(inner.spec, inner.heads)
        self._sink = sink

    def judged(
        self,
        stage: Stage,
        reading: SensorReading,
        findings: Iterable[Finding],
        organs: frozenset[str],
        subject: str | None = None,
        site: str | None = None,
    ) -> list[Assessment]:
        listed = list(findings)
        if not reading.is_empty:
            self._sink.append(_Judgement(stage, reading, frozenset(item.threat for item in listed), organs))
        return super().judged(stage, reading, listed, organs, subject, site)

    def candidates(
        self,
        candidates: Iterable[Candidate],
        reading: SensorReading | None,
        unavailable: bool,
        act_on_outage: bool,
        site: str | None = None,
    ) -> list[Assessment]:
        listed = list(candidates)
        for candidate in listed:
            scoped = reading.scoped(candidate.id) if reading is not None else SensorReading.empty()
            if not scoped.is_empty:
                self._sink.append(_Judgement(candidate.stage, scoped, frozenset(), frozenset(), (candidate.threat,)))
        return super().candidates(listed, reading, unavailable, act_on_outage, site)


class FeatureCollector:
    def __init__(self, sensors: SensorFactory) -> None:
        self._sensors = sensors

    def collect(self, examples: Iterable[Example]) -> Iterator[FeatureRow | SignalRow]:
        for example in examples:
            yield from self._example(example)

    def _example(self, example: Example) -> Iterator[FeatureRow | SignalRow]:
        judgements: list[_Judgement] = []
        with tempfile.TemporaryDirectory() as directory:
            harness = ImmuneHarness(Path(directory), sensor=self._sensors(example), script=self._reply(example))
            runtime = harness.runtime
            recording = _RecordingAssessor(runtime.parts.assessor, judgements)
            runtime.parts = replace(runtime.parts, assessor=recording)
            runtime.pipeline.reconfigure(runtime.parts)
            try:
                harness.openai().chat.completions.create(model="evidence", **self._request(example))
            finally:
                harness.close()
            heads = runtime.parts.assessor.heads
        yield from self._rows(example, judgements, heads, recording)

    def _rows(
        self, example: Example, judgements: Sequence[_Judgement], heads: HeadRegistry, assessor: Assessor
    ) -> Iterator[FeatureRow | SignalRow]:
        for judgement in judgements:
            threats = judgement.threats or assessor.judged_threats(judgement.stage, judgement.organs)
            for threat in threats:
                head = heads.get(threat)
                label = example.labels.get(threat)
                vector = head.vector(judgement.reading, set(judgement.findings)) if head else None
                if label is not None and vector is not None:
                    yield FeatureRow(
                        example.id, example.source, example.split, example.language, threat, tuple(vector), label
                    )
            for key, signal in judgement.reading.signals.items():
                yield SignalRow(example.id, example.split, key, signal.probability, dict(example.labels))

    @staticmethod
    def _reply(example: Example) -> FakeReply:
        calls = [FakeToolCall(call.name, call.arguments) for call in example.calls]
        return FakeReply(text=example.reply, tool_calls=calls)

    @staticmethod
    def _request(example: Example) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": example.operator},
            {"role": "user", "content": example.user},
        ]
        tools = list(dict.fromkeys([*example.tools, *(["read_document"] if example.data else [])]))
        for index, text in enumerate(example.data):
            call_id = f"doc_{index}"
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": "read_document", "arguments": "{}"}}
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        request: dict[str, Any] = {"messages": messages}
        if tools:
            request["tools"] = [
                {"type": "function", "function": {"name": name, "description": name.replace("_", " ")}}
                for name in tools
            ]
        return request


class FeatureFile:
    @staticmethod
    def write(path: Path, rows: Iterable[FeatureRow | SignalRow]) -> int:
        count = 0
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                kind = "feature" if isinstance(row, FeatureRow) else "signal"
                handle.write(json.dumps({"kind": kind, **asdict(row)}) + "\n")
                count += 1
        return count

    @staticmethod
    def read(path: Path) -> tuple[list[FeatureRow], list[SignalRow]]:
        features: list[FeatureRow] = []
        signals: list[SignalRow] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.pop("kind")
            if kind == "feature":
                record["vector"] = tuple(record["vector"])
                features.append(FeatureRow(**record))
            else:
                signals.append(SignalRow(**record))
        return features, signals
