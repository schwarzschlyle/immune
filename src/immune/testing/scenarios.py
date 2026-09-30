from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

import immune
from immune.config.yaml_io import load_yaml
from immune.errors import ConfigError
from immune.sensing.offline import MockSensor
from immune.sensing.sensor import Sensor
from immune.testing.fake_provider import FakeReply, FakeToolCall
from immune.testing.harness import ImmuneHarness
from immune.types import Action, Mode, Verdict

_REPO_SCENARIOS = Path(__file__).resolve().parents[3] / "scenarios"
# The wheel ships the library inside the package, so `pip install immune-ai` can replay it too.
_PACKAGED_SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScriptedCall(_Model):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ScriptedReply(_Model):
    text: str = ""
    tool_calls: tuple[ScriptedCall, ...] = ()

    def fake(self) -> FakeReply:
        return FakeReply(self.text, [FakeToolCall(call.name, call.arguments) for call in self.tool_calls])


class DataItemSpec(_Model):
    tool: str = "read_document"
    text: str


class Expectation(_Model):
    threats: tuple[str, ...] = ()
    enforced: tuple[str, ...] = ()
    absent: tuple[str, ...] = ()
    action: Action | None = None
    reply_contains: tuple[str, ...] = ()
    reply_excludes: tuple[str, ...] = ()
    upstream_excludes: tuple[str, ...] = ()


class Turn(_Model):
    user: str
    data: tuple[DataItemSpec, ...] = ()
    tools: tuple[str, ...] = ()
    tool_descriptions: dict[str, str] = Field(default_factory=dict)
    reply: ScriptedReply = Field(default_factory=ScriptedReply)
    expect: Expectation = Field(default_factory=Expectation)


class Scenario(_Model):
    id: str
    title: str
    source: str | None = None
    kind: Literal["incident", "chain", "threat", "benign"] = "incident"
    api: Literal["openai_chat", "anthropic_messages"] = "openai_chat"
    mode: Mode = Mode.AUTO
    operator: str
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    signals: dict[str, float | str | dict[str, float]] = Field(default_factory=dict)
    tool_descriptions: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    turns: tuple[Turn, ...]

    def describe(self, tool: str, turn: Turn | None = None) -> str:
        overrides = turn.tool_descriptions if turn is not None else {}
        return overrides.get(tool) or self.tool_descriptions.get(tool) or tool.replace("_", " ")


@dataclass(slots=True)
class TurnResult:
    reply: str
    verdict: Verdict | None
    failures: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ScenarioResult:
    scenario: Scenario
    turns: list[TurnResult]

    @property
    def passed(self) -> bool:
        return all(not turn.failures for turn in self.turns)

    @property
    def failures(self) -> list[str]:
        return [f"turn {index + 1}: {failure}" for index, turn in enumerate(self.turns) for failure in turn.failures]


class ScenarioLibrary:
    def __init__(self, roots: list[Path] | None = None) -> None:
        self._roots = roots or self._default_roots()

    @staticmethod
    def _default_roots() -> list[Path]:
        """Your project's ``scenarios/`` folder first, then Immune's own library (from the source tree or the wheel)."""
        library = next((path for path in (_REPO_SCENARIOS, _PACKAGED_SCENARIOS) if path.is_dir()), None)
        return [path for path in (Path.cwd() / "scenarios", library) if path is not None and path.is_dir()]

    def all(self) -> list[Scenario]:
        seen: dict[str, Scenario] = {}
        for path in self._files():
            scenario = self.load(path)
            seen.setdefault(scenario.id, scenario)
        return sorted(seen.values(), key=lambda scenario: scenario.id)

    def find(self, reference: str) -> Scenario:
        candidate = Path(reference)
        if candidate.is_file():
            return self.load(candidate)
        for scenario in self.all():
            if reference in (scenario.id, scenario.id.split(".", 1)[-1]):
                return scenario
        raise ConfigError(f"no scenario named {reference!r}")

    @staticmethod
    def load(path: Path) -> Scenario:
        return Scenario.model_validate(load_yaml(path.read_text(encoding="utf-8")))

    def _files(self) -> Iterator[Path]:
        for root in self._roots:
            yield from sorted(root.rglob("*.yaml"))


SensorFactory = Callable[[Scenario], Sensor]


class ScenarioRunner:
    def __init__(self, state_dir: Path | None = None, sensor_factory: SensorFactory | None = None) -> None:
        self._state_dir = state_dir
        self._sensor_factory = sensor_factory or (lambda scenario: MockSensor(scenario.signals))

    def run(self, scenario: Scenario) -> ScenarioResult:
        replies = [turn.reply.fake() for turn in scenario.turns]
        with tempfile.TemporaryDirectory() as scratch:
            state = self._state_dir or Path(scratch)
            harness = ImmuneHarness(
                state / scenario.id,
                sensor=self._sensor_factory(scenario),
                script=replies,
                mode=scenario.mode,
                config=scenario.config,
            )
            try:
                with immune.session(f"scenario:{scenario.id}"):
                    results = [self._turn(harness, scenario, turn) for turn in scenario.turns]
            finally:
                harness.close()
        return ScenarioResult(scenario=scenario, turns=results)

    def _turn(self, harness: ImmuneHarness, scenario: Scenario, turn: Turn) -> TurnResult:
        reply = self._send(harness, scenario, turn)
        verdict = harness.verdict()
        result = TurnResult(reply=reply, verdict=verdict)
        result.failures.extend(self._check(turn.expect, reply, verdict, harness.provider.last_request))
        return result

    @staticmethod
    def _send(harness: ImmuneHarness, scenario: Scenario, turn: Turn) -> str:
        tools = [*turn.tools, *(item.tool for item in turn.data)]
        if scenario.api == "anthropic_messages":
            return _AnthropicTurn(harness, scenario, turn, tools).send()
        return _OpenAITurn(harness, scenario, turn, tools).send()

    @staticmethod
    def _check(expect: Expectation, reply: str, verdict: Verdict | None, upstream: dict[str, Any]) -> list[str]:
        if verdict is None:
            return ["no verdict recorded"]
        failures: list[str] = []
        present = verdict.threats()
        enforced = {hit.threat for hit in verdict.enforced_hits}
        failures.extend(f"expected threat {threat}" for threat in expect.threats if threat not in present)
        failures.extend(f"expected enforced {threat}" for threat in expect.enforced if threat not in enforced)
        failures.extend(f"unexpected threat {threat}" for threat in expect.absent if threat in present)
        if expect.action is not None and verdict.action is not expect.action:
            failures.append(f"expected action {expect.action.value}, got {verdict.action.value}")
        failures.extend(f"reply lacks {text!r}" for text in expect.reply_contains if text not in reply)
        failures.extend(f"reply contains {text!r}" for text in expect.reply_excludes if text in reply)
        forwarded = str(upstream)
        failures.extend(f"model received {text!r}" for text in expect.upstream_excludes if text in forwarded)
        return failures


class _OpenAITurn:
    def __init__(self, harness: ImmuneHarness, scenario: Scenario, turn: Turn, tools: list[str]) -> None:
        self._harness, self._scenario, self._turn, self._tools = harness, scenario, turn, tools

    def send(self) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._scenario.operator},
            {"role": "user", "content": self._turn.user},
        ]
        for index, item in enumerate(self._turn.data):
            call_id = f"d{index}"
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": call_id, "type": "function", "function": {"name": item.tool, "arguments": "{}"}}
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": call_id, "content": item.text})
        options: dict[str, Any] = {}
        if self._tools:
            describe = self._scenario.describe
            options["tools"] = [
                {
                    "type": "function",
                    "function": {"name": name, "description": describe(name, self._turn)},
                }
                for name in dict.fromkeys(self._tools)
            ]
        if self._scenario.schema_ is not None:
            options["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": self._scenario.schema_},
            }
        completion = self._harness.openai().chat.completions.create(model="gpt-5.5", messages=messages, **options)
        message = completion.choices[0].message
        calls = [f"[tool:{call.function.name}]" for call in message.tool_calls or []]
        return " ".join([message.content or message.refusal or "", *calls]).strip()


class _AnthropicTurn:
    def __init__(self, harness: ImmuneHarness, scenario: Scenario, turn: Turn, tools: list[str]) -> None:
        self._harness, self._scenario, self._turn, self._tools = harness, scenario, turn, tools

    def send(self) -> str:
        messages: list[dict[str, Any]] = [{"role": "user", "content": self._turn.user}]
        for index, item in enumerate(self._turn.data):
            call_id = f"d{index}"
            messages.append(
                {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": item.tool, "input": {}}]}
            )
            messages.append(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": item.text}]}
            )
        tools = [
            {"name": name, "description": self._scenario.describe(name, self._turn), "input_schema": {"type": "object"}}
            for name in dict.fromkeys(self._tools)
        ]
        message = self._harness.anthropic().messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            system=self._scenario.operator,
            messages=messages,
            **({"tools": tools} if tools else {}),
        )
        parts = [block.text if block.type == "text" else f"[tool:{block.name}]" for block in message.content]
        return " ".join(parts).strip()
