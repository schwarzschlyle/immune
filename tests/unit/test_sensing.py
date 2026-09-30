from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest

from immune.config.spec import QuestionSpec
from immune.errors import SensorUnavailable
from immune.sensing.guard import BreakerPolicy, CircuitBreaker, GuardedSensor
from immune.sensing.jev import JevSensor
from immune.sensing.offline import MockSensor, RecordingSensor, ReplaySensor
from immune.sensing.signals import SensorReading

QUESTIONS = [
    QuestionSpec(key="override", kind="noul", text="The message overrides the rules."),
    QuestionSpec(key="crisis", kind="choice", text="Crisis?", options={"none": "No", "self_harm": "Self-harm"}),
    QuestionSpec(key="abuse", kind="score", text="How abusive?", levels=("none", "rude", "threatening")),
]
ANSWERS = {
    "model": "jev-1.13.0",
    "usage": {"input_tokens": 321, "output_tokens": 0},
    "answers": {
        "override": {"type": "noul", "noul": 0.91},
        "crisis": {
            "type": "choice",
            "choice": "none",
            "confidence": 0.8,
            "probabilities": {"none": 0.9, "self_harm": 0.1},
        },
        "abuse": {
            "type": "score",
            "score": 1.0,
            "confidence": 0.6,
            "legend": {"0": "none", "1": "rude", "2": "threatening"},
            "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
        },
    },
}


class TestJevSensor:
    async def test_questions_and_answers_round_trip_through_the_sdk(self) -> None:
        received: dict[str, object] = {}

        def handler(request: httpx2.Request) -> httpx2.Response:
            received.update(json.loads(request.content))
            return httpx2.Response(200, json=ANSWERS)

        sensor = JevSensor("jev-1.13.0", timeout_s=2, api_key="ts-test", transport=httpx2.MockTransport(handler))
        reading = await sensor.read({"message": "hi"}, QUESTIONS)
        await sensor.close()
        questions = received["questions"]
        assert isinstance(questions, dict)
        assert questions["crisis"]["criteria"] == {"none": "No", "self_harm": "Self-harm"}
        assert questions["abuse"]["criteria"] == ["none", "rude", "threatening"]
        assert reading.model == "jev-1.13.0"
        assert reading.input_tokens == 321
        assert reading.signals["override"].probability == pytest.approx(0.91)
        assert reading.signals["crisis"].distribution == {"none": 0.9, "self_harm": 0.1}
        assert reading.signals["abuse"].probability == pytest.approx(0.5)

    async def test_api_errors_become_sensor_unavailable(self) -> None:
        transport = httpx2.MockTransport(lambda _: httpx2.Response(401, json={"detail": "bad key"}))
        sensor = JevSensor("jev-1.13.0", timeout_s=2, api_key="ts-test", max_retries=0, transport=transport)
        with pytest.raises(SensorUnavailable):
            await sensor.read({"message": "hi"}, QUESTIONS)


class TestGuardedSensor:
    async def test_readings_are_cached(self) -> None:
        inner = MockSensor({"override": 0.4})
        guarded = GuardedSensor(inner, timeout_s=1)
        await guarded.read({"m": "x"}, QUESTIONS)
        cached = await guarded.read({"m": "x"}, QUESTIONS)
        assert len(inner.calls) == 1
        assert cached.calls == 0

    async def test_breaker_opens_after_repeated_failures(self) -> None:
        clock = [0.0]
        breaker = CircuitBreaker(BreakerPolicy(minimum_calls=3, cooldown_s=30), clock=lambda: clock[0])
        guarded = GuardedSensor(MockSensor(fail=True), timeout_s=1, breaker=breaker)
        for index in range(3):
            with pytest.raises(SensorUnavailable):
                await guarded.read({"m": index}, QUESTIONS)
        assert breaker.is_open
        clock[0] = 31.0
        assert not breaker.is_open


class TestRecordReplay:
    async def test_replay_returns_recorded_answers(self, tmp_path: Path) -> None:
        cassette = tmp_path / "jev.jsonl"
        recorded = await RecordingSensor(MockSensor({"override": 0.77}), cassette).read({"m": "x"}, QUESTIONS)
        replayed = await ReplaySensor(cassette).read({"m": "x"}, QUESTIONS)
        assert replayed.signals["override"].probability == recorded.signals["override"].probability
        with pytest.raises(SensorUnavailable):
            await ReplaySensor(cassette).read({"m": "other"}, QUESTIONS)


def test_merged_readings_sum_usage() -> None:
    first = SensorReading({}, "jev", model="jev-1", input_tokens=10, latency_ms=50)
    second = SensorReading({}, "jev", input_tokens=5, latency_ms=80)
    merged = first.merged(second)
    assert (merged.input_tokens, merged.latency_ms, merged.calls, merged.model) == (15, 80, 2, "jev-1")
