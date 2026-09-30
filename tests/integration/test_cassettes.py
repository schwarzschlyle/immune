from __future__ import annotations

from pathlib import Path

import pytest

from immune.sensing.offline import ReplaySensor
from immune.testing import ImmuneHarness
from tests.live.conversations import CONVERSATIONS, LiveConversation

CASSETTES = Path(__file__).parents[1] / "live" / "cassettes"
RECORDED = [conversation for conversation in CONVERSATIONS if (CASSETTES / f"{conversation.name}.jsonl").exists()]


@pytest.mark.skipif(not RECORDED, reason="no live cassettes recorded yet; run the live workflow")
@pytest.mark.parametrize("conversation", RECORDED, ids=[item.name for item in RECORDED])
def test_recorded_jev_answers_replay_through_the_pipeline(conversation: LiveConversation, tmp_path: Path) -> None:
    sensor = ReplaySensor(CASSETTES / f"{conversation.name}.jsonl")
    with ImmuneHarness(tmp_path, sensor=sensor, script=conversation.reply) as harness:
        conversation.call(harness)
        verdict = harness.verdict()
    assert verdict is not None
    assert verdict.sensor.name == "replay"
