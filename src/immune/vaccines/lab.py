from __future__ import annotations

import fnmatch
import json
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import immune
from immune.config.yaml_io import load_yaml
from immune.sensing.offline import MockSensor
from immune.sensing.sensor import Sensor
from immune.testing.fake_provider import FakeReply, FakeToolCall
from immune.testing.harness import ImmuneHarness
from immune.types import Stage, Verdict
from immune.vaccines.loader import LoadedVaccine, VaccineError
from immune.vaccines.model import ToolExample, Vaccine

SensorFactory = Callable[[], Sensor]
SensorKind = Literal["offline", "scripted", "live"]

_URL = "https://api.openai.com/v1/chat/completions"
_OPERATOR = "You are the assistant for this application. Help users with their requests."
_SITE = "vaccine-lab"
_YES, _NO = 0.97, 0.02
_LABEL_WIDTH = 60


@dataclass(frozen=True, slots=True)
class Probe:
    """One message, document, reply or tool call, placed at a vaccine's stage and screened by Immune."""

    text: str = ""
    call: ToolExample | None = None

    @classmethod
    def of(cls, example: str | ToolExample) -> Probe:
        return cls(call=example) if isinstance(example, ToolExample) else cls(text=example)

    @property
    def label(self) -> str:
        if self.call is not None:
            arguments = json.dumps(self.call.arguments, sort_keys=True, default=str)
            label = f"{self.call.tool} {arguments}"
        else:
            label = " ".join(self.text.split())
        return label if len(label) <= _LABEL_WIDTH else f"{label[: _LABEL_WIDTH - 1]}…"

    def request(self, stage: Stage) -> tuple[dict[str, Any], FakeReply]:
        system = {"role": "system", "content": _OPERATOR}
        if stage is Stage.TOOL:
            if self.call is None:
                raise VaccineError(
                    f"a tool vaccine needs tool examples ({{tool: ..., arguments: ...}}), got {self.label!r}"
                )
            body = {
                "messages": [system, {"role": "user", "content": "Please go ahead."}],
                "tools": [{"type": "function", "function": {"name": self.call.tool, "description": self.call.tool}}],
            }
            return body, FakeReply(tool_calls=[FakeToolCall(self.call.tool, dict(self.call.arguments))])
        if self.call is not None:
            raise VaccineError(f"a {stage.value} vaccine needs text examples, got the tool call {self.label!r}")
        if stage is Stage.INPUT:
            return {"messages": [system, {"role": "user", "content": self.text}]}, FakeReply("OK.")
        if stage is Stage.DATA:
            read = {"id": "lab_0", "type": "function", "function": {"name": "read_document", "arguments": "{}"}}
            messages = [
                system,
                {"role": "user", "content": "Please summarize the document."},
                {"role": "assistant", "tool_calls": [read]},
                {"role": "tool", "tool_call_id": "lab_0", "content": self.text},
            ]
            return {"messages": messages}, FakeReply("Here is a summary.")
        return {"messages": [system, {"role": "user", "content": "Hello"}]}, FakeReply(self.text)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    probe: Probe
    expected: bool | None
    fired: bool
    enforced: bool = False
    probability: float | None = None
    evidence: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.expected is None or self.fired is self.expected


@dataclass(frozen=True, slots=True)
class VaccineReport:
    vaccine: str
    source: Path
    sensor: SensorKind
    results: tuple[ProbeResult, ...]
    notes: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def failures(self) -> list[ProbeResult]:
        return [result for result in self.results if not result.passed]


@dataclass(frozen=True, slots=True)
class TrialReport:
    vaccine: str
    sensor: SensorKind
    sources: tuple[str, ...]
    results: tuple[ProbeResult, ...]
    input_tokens: int = 0

    @property
    def screened(self) -> int:
        return len(self.results)

    @property
    def fired(self) -> list[ProbeResult]:
        return [result for result in self.results if result.fired]

    @property
    def rate(self) -> float:
        return len(self.fired) / self.screened if self.screened else 0.0


class SelfSamples:
    """Everyday traffic ("self") that a vaccine should leave alone, used by trials.

    The packaged samples cover common assistant domains. Teams add their own traffic with a corpus file: JSON Lines
    with a string, ``{"text": ...}`` or ``{"tool": ..., "arguments": {...}}`` per line, or plain text with one
    message per line.
    """

    def packaged(self, stage: Stage) -> list[Probe]:
        document = load_yaml(resources.files("immune.vaccines").joinpath("self.yaml").read_text("utf-8"))
        entries = document.get(stage.value, [])
        return [self._probe(entry, f"self.yaml {stage.value}") for entry in entries]

    def corpus(self, path: Path) -> list[Probe]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise VaccineError(f"cannot read corpus {path}: {error}") from error
        probes: list[Probe] = []
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            if path.suffix != ".jsonl":
                probes.append(Probe(text=line.strip()))
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise VaccineError(f"{path}:{number}: not valid JSON ({error.msg})") from error
            probes.append(self._probe(entry, f"{path}:{number}"))
        return probes

    @staticmethod
    def _probe(entry: Any, where: str) -> Probe:
        if isinstance(entry, str):
            return Probe(text=entry)
        if isinstance(entry, Mapping) and isinstance(entry.get("text"), str):
            return Probe(text=entry["text"])
        if isinstance(entry, Mapping) and isinstance(entry.get("tool"), str):
            return Probe(call=ToolExample(tool=entry["tool"], arguments=dict(entry.get("arguments") or {})))
        raise VaccineError(f"{where}: use a string, {{'text': ...}} or {{'tool': ..., 'arguments': {{...}}}}")


class VaccineLab:
    """Runs vaccines against examples in a private Immune runtime, with no model or network calls.

    Deterministic vaccines (keywords, regex, tool rules, Python) are checked for real. Question vaccines need Jev to
    judge the examples; without a live sensor their questions are answered by a scripted sensor, which checks the
    head, stage and site wiring but not the wording of the questions.
    """

    def __init__(self, live: SensorFactory | None = None, site: str | None = None) -> None:
        self._live = live
        self._site = site

    @property
    def live(self) -> bool:
        return self._live is not None

    def test(self, loaded: LoadedVaccine) -> VaccineReport:
        vaccine = loaded.vaccine
        cases = [(Probe.of(example), True) for example in vaccine.tests.positives]
        cases += [(Probe.of(example), False) for example in vaccine.tests.negatives]
        if not cases:
            note = "no tests: add tests.positives (should fire) and tests.negatives (should not)"
            return VaccineReport(vaccine.id, loaded.source, self._kind(vaccine), (), (note,))
        notes: list[str] = []
        if self._live is None and vaccine.detect.confirm == "jev":
            notes.append(
                "a scripted sensor confirmed every match in place of Jev; run with --live to have Jev judge them"
            )
        if self._live is not None or vaccine.detect.kind != "questions":
            results = self._run(loaded, cases, self._sensor(vaccine))
        else:
            notes.append(
                "questions were answered by a scripted sensor, which checks the head, stage and site wiring; "
                "run with --live to judge the examples with Jev"
            )
            positives = [case for case in cases if case[1]]
            negatives = [case for case in cases if not case[1]]
            results = [
                *self._run(loaded, positives, MockSensor(self.answers(vaccine, present=True))),
                *self._run(loaded, negatives, MockSensor(self.answers(vaccine, present=False))),
            ]
        return VaccineReport(vaccine.id, loaded.source, self._kind(vaccine), tuple(results), tuple(notes))

    def trial(self, loaded: LoadedVaccine, probes: Sequence[Probe], sources: Sequence[str]) -> TrialReport:
        vaccine = loaded.vaccine
        if vaccine.detect.kind == "questions" and self._live is None:
            raise VaccineError(f"{vaccine.id} asks Jev questions, so its trial needs Jev: run with --live")
        tokens: list[int] = []
        results = self._run(loaded, [(probe, None) for probe in probes], self._sensor(vaccine), tokens)
        return TrialReport(vaccine.id, self._kind(vaccine), tuple(sources), tuple(results), sum(tokens))

    @staticmethod
    def answers(vaccine: Vaccine, present: bool) -> dict[str, float | str | dict[str, float]]:
        weights = vaccine.detect.head.weights if vaccine.detect.head is not None else {}
        answers: dict[str, float | str | dict[str, float]] = {}
        for question in vaccine.detect.questions:
            key = vaccine.question_key(question.key)
            yes = present if weights.get(question.key, 1.0) >= 0 else not present
            if question.kind == "choice":
                others = [option for option in question.options if option not in question.flag]
                answers[key] = question.flag[0] if yes or not others else others[0]
            else:
                answers[key] = _YES if yes else _NO
        return answers

    def site_for(self, vaccine: Vaccine) -> str:
        patterns = vaccine.applies_to.sites
        if self._site is not None:
            return self._site
        if not patterns or any(fnmatch.fnmatchcase(_SITE, pattern) for pattern in patterns):
            return _SITE
        for pattern in patterns:
            candidate = pattern.replace("*", "").replace("?", "x")
            if candidate and fnmatch.fnmatchcase(candidate, pattern):
                return candidate
        raise VaccineError(f"{vaccine.id}: cannot pick a site that matches {list(patterns)}; pass --site")

    def _run(
        self,
        loaded: LoadedVaccine,
        cases: Sequence[tuple[Probe, bool | None]],
        sensor: Sensor,
        tokens: list[int] | None = None,
    ) -> list[ProbeResult]:
        if not cases:
            return []
        vaccine = loaded.vaccine
        site = self.site_for(vaccine)
        replies: dict[str, FakeReply] = {}
        results: list[ProbeResult] = []
        with tempfile.TemporaryDirectory(prefix="immune-vaccine-lab-") as scratch:
            harness = ImmuneHarness(
                Path(scratch),
                sensor=sensor,
                script=lambda _: replies["next"],
                config=self._config(loaded, site, [probe for probe, _ in cases]),
            )
            client = harness.http_client()
            try:
                previous: Verdict | None = None
                for probe, expected in cases:
                    body, replies["next"] = probe.request(vaccine.stage)
                    with immune.session(f"vaccine-lab-{uuid.uuid4().hex[:8]}"), immune.site(site):
                        client.post(
                            _URL, json={"model": "vaccine-lab", **body}, headers={"authorization": "Bearer lab"}
                        )
                    verdict = harness.verdict()
                    verdict = None if verdict is previous else verdict
                    previous = verdict or previous
                    if tokens is not None and verdict is not None:
                        tokens.append(verdict.sensor.input_tokens)
                    results.append(self._result(vaccine.id, probe, expected, verdict))
            finally:
                client.close()
                harness.close()
        return results

    @staticmethod
    def _config(loaded: LoadedVaccine, site: str, probes: Iterable[Probe]) -> dict[str, Any]:
        vaccine = loaded.vaccine
        settings: dict[str, Any] = {}
        if vaccine.applies_to.organs:
            settings["organs"] = list(vaccine.applies_to.organs)
        if vaccine.stage is Stage.TOOL and vaccine.detect.kind == "questions":
            tools = {probe.call.tool for probe in probes if probe.call is not None}
            settings["tools"] = {tool: {"writes_state": True} for tool in tools}
        return {
            "vaccines": {"paths": [str(loaded.source.resolve())], "entry_points": False, "enabled": [vaccine.id]},
            "sites": {site: settings},
            "telemetry": {"opentelemetry": False, "alerts": {"enabled": False}},
            "promotion": {"enabled": False},
        }

    @staticmethod
    def _result(vaccine_id: str, probe: Probe, expected: bool | None, verdict: Verdict | None) -> ProbeResult:
        hits = [hit for hit in verdict.hits if hit.threat == vaccine_id] if verdict is not None else []
        return ProbeResult(
            probe=probe,
            expected=expected,
            fired=bool(hits),
            enforced=any(hit.enforced for hit in hits),
            probability=max((hit.probability for hit in hits), default=None),
            evidence=tuple(evidence for hit in hits for evidence in hit.evidence),
        )

    def _sensor(self, vaccine: Vaccine) -> Sensor:
        if self._live is not None and self._asks_jev(vaccine):
            return self._live()
        return MockSensor()

    @staticmethod
    def _asks_jev(vaccine: Vaccine) -> bool:
        return vaccine.detect.kind == "questions" or vaccine.detect.confirm == "jev"

    def _kind(self, vaccine: Vaccine) -> SensorKind:
        if not self._asks_jev(vaccine):
            return "offline"
        return "live" if self._live is not None else "scripted"
