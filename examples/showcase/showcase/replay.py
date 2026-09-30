"""Incident replays from Immune's scenario library.

Attack examples are never written in this project. They come from `scenarios/` in the Immune repository, which
reconstructs publicly reported incidents; the model's side of each attack is the recorded reply, served by a fake
provider, so no real model is asked to misbehave. Incidents about harmful content or personal crisis are reported
as a verdict line only.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from immune.testing import FakeReply, FakeToolCall, ScenarioLibrary, ScenarioRunner
from immune.testing.scenarios import Scenario
from showcase.config import LIBRARY
from showcase.features import Ops, Session, Turn

SENSITIVE = frozenset({"incident.pak-n-save", "incident.character-crisis"})


@dataclass(frozen=True, slots=True)
class ReplayResult:
    scenario: str
    title: str
    passed: bool
    summary: str
    sensitive: bool


class Replays:
    def __init__(self, library: Path = LIBRARY) -> None:
        self.library = ScenarioLibrary([library]) if library.is_dir() else None

    @property
    def available(self) -> bool:
        return self.library is not None

    def ids(self) -> list[str]:
        return [scenario.id for scenario in self.library.all()] if self.library else []

    def find(self, scenario_id: str) -> Scenario:
        if self.library is None:
            raise LookupError("the scenario library is not available; run the showcase from the Immune repository")
        return self.library.find(scenario_id)

    def run(self, scenario_id: str) -> ReplayResult:
        """Run one library scenario as recorded, and summarize its verdict."""
        scenario = self.find(scenario_id)
        with tempfile.TemporaryDirectory(prefix="showcase-replay-") as scratch:
            result = ScenarioRunner(Path(scratch)).run(scenario)
        verdict = result.turns[-1].verdict
        summary = verdict.explanation if verdict else "no verdict"
        sensitive = scenario.id in SENSITIVE
        title = "(sensitive incident: verdict only)" if sensitive else scenario.title
        return ReplayResult(scenario.id, title, result.passed, summary, sensitive)

    def run_all(self) -> list[ReplayResult]:
        return [self.run(scenario_id) for scenario_id in self.ids() if not scenario_id.startswith("benign.")]

    def poisoned_page(self) -> str:
        """The data item from the library's EchoLeak reconstruction, used as a poisoned help-center article."""
        return self.find("incident.echoleak").turns[0].data[0].text

    def forced_leak(self, ops: Ops, session: Session) -> Turn:
        """ForcedLeak through the ops assistant's own code: the ticket and the model's tool call are the recording."""
        scenario = self.find("incident.forcedleak")
        turn = scenario.turns[0]
        recorded = turn.reply.tool_calls[0]
        ops.tools.tickets["T-99"] = turn.data[0].text
        script = [
            FakeReply(tool_calls=[FakeToolCall("read_ticket", {"ticket_id": "T-99"})]),
            FakeReply(tool_calls=[FakeToolCall("send_email", dict(recorded.arguments))]),
            FakeReply(text="I've handled the ticket."),
        ]
        with ops.llm.replay(script):
            return ops.turn(session, "Please handle ticket T-99.")
