from __future__ import annotations

import json
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from immune.config.spec import QuestionSpec
from immune.sensing.jev import JevSensor
from immune.sensing.offline import RecordingSensor
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading
from immune.testing import ImmuneHarness, JevWireStub
from tests.live.conversations import CONVERSATIONS, LiveConversation

CASSETTES = Path(__file__).parent / "cassettes"
REPORT = Path(os.environ.get("IMMUNE_LIVE_REPORT", Path(__file__).parent / "report.json"))

STUBBED = os.environ.get("IMMUNE_LIVE_TRANSPORT") == "stub"

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not (os.environ.get("TYPESAFE_API_KEY") or STUBBED), reason="needs TYPESAFE_API_KEY"),
]


class AuditingSensor(Sensor):
    name: ClassVar[str] = "auditing"

    def __init__(self, inner: Sensor) -> None:
        self._inner = inner
        self.exchanges: list[tuple[Mapping[str, Any], Sequence[QuestionSpec], SensorReading]] = []

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        reading = await self._inner.read(state, questions)
        self.exchanges.append((state, questions, reading))
        return reading

    async def close(self) -> None:
        await self._inner.close()


def estimated_tokens(state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> int:
    described = [(question.text, question.options, question.levels) for question in questions]
    return len(json.dumps({"state": state, "questions": described})) // 4


@pytest.fixture(scope="module")
def exchanges(tmp_path_factory: pytest.TempPathFactory) -> dict[str, AuditingSensor]:
    audited: dict[str, AuditingSensor] = {}
    for conversation in CONVERSATIONS:
        audited[conversation.name] = run(conversation, tmp_path_factory.mktemp(conversation.name))
    return audited


def run(conversation: LiveConversation, state_dir: Path) -> AuditingSensor:
    model = os.environ.get("IMMUNE_JEV_MODEL", "jev-1.13.0")
    if STUBBED:
        jev = JevSensor(model, timeout_s=5.0, api_key="ts-stub", transport=JevWireStub(model=model).transport())
        cassette = state_dir / "cassette.jsonl"
    else:
        jev = JevSensor(model, timeout_s=5.0)
        cassette = CASSETTES / f"{conversation.name}.jsonl"
    sensor = AuditingSensor(RecordingSensor(jev, cassette))
    with ImmuneHarness(
        state_dir, sensor=sensor, script=conversation.reply, config={"sensor": {"timeout_ms": 5000}}
    ) as harness:
        conversation.call(harness)
    return sensor


@pytest.mark.parametrize("conversation", CONVERSATIONS, ids=[item.name for item in CONVERSATIONS])
def test_every_question_is_answered_with_its_kind(
    conversation: LiveConversation, exchanges: dict[str, AuditingSensor]
) -> None:
    audited = exchanges[conversation.name]
    assert audited.exchanges
    for _, questions, reading in audited.exchanges:
        assert reading.source == "jev"
        for question in questions:
            signal = reading.signals[question.key]
            assert signal.kind == question.kind
            assert 0.0 <= signal.probability <= 1.0


def test_latency_and_token_estimates(exchanges: dict[str, AuditingSensor]) -> None:
    rows = [
        {
            "conversation": name,
            "questions": len(questions),
            "latency_ms": round(reading.latency_ms, 1),
            "billed_tokens": reading.input_tokens,
            "estimated_tokens": estimated_tokens(state, questions),
        }
        for name, audited in exchanges.items()
        for state, questions, reading in audited.exchanges
    ]
    latencies = [row["latency_ms"] for row in rows]
    ratios = [row["billed_tokens"] / max(1, row["estimated_tokens"]) for row in rows]
    REPORT.write_text(
        json.dumps(
            {
                "calls": rows,
                "latency_p50_ms": statistics.median(latencies),
                "latency_max_ms": max(latencies),
                "token_ratio_median": statistics.median(ratios),
            },
            indent=2,
        )
    )
    assert max(latencies) < 3000
    assert 0.5 <= statistics.median(ratios) <= 2.0
