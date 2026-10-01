"""The vaccine laboratory: develop, measure and gate the vaccines that ship with Immune.

    python tools/vaccine_lab.py new immune.health.dosage_instructions --kind questions --stage output --owner @me
    python tools/vaccine_lab.py record immune.health.dosage_instructions   # live Jev, once: writes the cassette
    python tools/vaccine_lab.py fit immune.health.dosage_instructions      # fits the head from the cassette
    python tools/vaccine_lab.py measure immune.health.dosage_instructions  # writes card.json and card.md
    python tools/vaccine_lab.py index                                      # bundle.json and the catalog page
    python tools/vaccine_lab.py check --all                                # what CI runs: replay, compare, gates
    python tools/vaccine_lab.py drift --all                                # weekly: re-ask Jev, compare the numbers

Everything except `record` runs offline from the cassette, so CI needs no keys. docs/contributing/vaccine-laboratory.md
is the full guide.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import yaml

from immune.config.settings import SensorSettings, VaccineSettings
from immune.config.spec import QuestionSpec, Spec
from immune.errors import SensorUnavailable
from immune.evidence.examples import DatasetManifest, Example, ExampleSet
from immune.evidence.metrics import Metrics, Scored
from immune.heads.calibration import logit, sigmoid
from immune.heads.optimize import RegularizedLogistic
from immune.reflexes import ReflexSuite
from immune.sensing.jev import JevSensor
from immune.sensing.offline import MockSensor, RecordingSensor, ReplaySensor
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading, Signal
from immune.types import Stage
from immune.vaccines.catalog import BUNDLE, LibraryCatalog
from immune.vaccines.detectors import VaccineContext
from immune.vaccines.lab import Probe, ProbeResult, SelfSamples, VaccineLab
from immune.vaccines.loader import LoadedVaccine, VaccineLoader
from immune.vaccines.model import ToolExample, Vaccine, VaccineError

ROOT = Path(__file__).resolve().parents[1]
JEV_USD_PER_MILLION_INPUT_TOKENS = 0.042
RECORD_TIMEOUT_MS = 15_000
DRIFT_TOLERANCE = 0.05
_ID = re.compile(r"^immune\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")
_KINDS = ("questions", "keywords", "regex", "tool")
_SUBJECT = {"input": "The user message", "data": "The {item}", "tool": "The {call}", "output": "The assistant output"}
_SUMMARY = ("recall", "precision", "hard_negative_fpr", "self_rate", "ece")

# What a vaccine must show to carry each maturity. Starting points, tuned as the library grows.
GATES: dict[str, dict[str, float]] = {
    "experimental": {
        "positives": 25,
        "hard_negatives": 25,
        "recall": 0.80,
        "hard_negative_fpr": 0.10,
        "self_rate": 0.01,
        "questions": 3,
        "p95_ms": 1.0,
    },
    "stable": {
        "positives": 100,
        "hard_negatives": 100,
        "sources": 2,
        "recall": 0.90,
        "hard_negative_fpr": 0.03,
        "self_rate": 0.002,
        "ece": 0.05,
        "questions": 3,
        "p95_ms": 1.0,
    },
}


class LabError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Layout:
    """Where the library, the laboratory and the catalog page live in a checkout."""

    root: Path = ROOT

    @property
    def library(self) -> Path:
        return self.root / "src" / "immune" / "vaccines" / "library"

    @property
    def lab(self) -> Path:
        return self.root / "laboratory"

    @property
    def catalog_page(self) -> Path:
        return self.root / "docs" / "reference" / "vaccine-catalog.md"

    def self_corpus(self, stage: Stage) -> Path:
        return self.lab / "self" / f"{stage.value}.jsonl"

    def workspace(self, vaccine_id: str) -> Workspace:
        if not _ID.match(vaccine_id):
            raise LabError(f"{vaccine_id!r}: library ids are immune.<domain>.<name>, lowercase with underscores")
        return Workspace(self, vaccine_id)

    def vaccine_ids(self) -> list[str]:
        return LibraryCatalog.from_sources(self.library).ids if self.library.is_dir() else []


@dataclass(frozen=True, slots=True)
class Workspace:
    """One vaccine: its library file and its laboratory folder."""

    layout: Layout
    vaccine_id: str

    @property
    def source(self) -> Path:
        _, domain, name = self.vaccine_id.split(".")
        return self.layout.library / domain / f"{name}.yaml"

    @property
    def folder(self) -> Path:
        return self.layout.lab / "vaccines" / self.vaccine_id

    @property
    def evidence(self) -> Path:
        return self.folder / "evidence.jsonl"

    @property
    def manifest(self) -> Path:
        return self.folder / "manifest.yaml"

    @property
    def cassette(self) -> Path:
        return self.folder / "cassette.jsonl"

    @property
    def recording(self) -> Path:
        return self.folder / "recording.json"

    @property
    def card_json(self) -> Path:
        return self.folder / "card.json"

    @property
    def card_md(self) -> Path:
        return self.folder / "card.md"

    @property
    def notes(self) -> Path:
        return self.folder / "NOTES.md"

    def loaded(self) -> LoadedVaccine:
        if not self.source.is_file():
            raise LabError(f"{self.vaccine_id}: no library file at {self.source.relative_to(self.layout.root)}")
        LibraryCatalog.read_source(self.source)
        return VaccineLoader(VaccineSettings(entry_points=False, library=False)).load_file(self.source)

    def examples(self) -> list[Example]:
        if not self.evidence.is_file():
            raise LabError(f"{self.vaccine_id}: no evidence at {self.evidence.relative_to(self.layout.root)}")
        return list(ExampleSet.read(self.evidence))


# Probes and sensors


def probe_for(example: Example, stage: Stage) -> Probe:
    """An evidence row, placed where the vaccine looks."""
    if stage is Stage.OUTPUT:
        if not example.reply:
            raise LabError(f"{example.id}: an output vaccine's evidence needs `reply`")
        return Probe(text=example.reply, user=example.user, operator=example.operator)
    if stage is Stage.INPUT:
        return Probe(text=example.user, operator=example.operator)
    if stage is Stage.DATA:
        if not example.data:
            raise LabError(f"{example.id}: a data vaccine's evidence needs `data`")
        return Probe(text=example.data[0], user=example.user or None, operator=example.operator)
    if not example.calls:
        raise LabError(f"{example.id}: a tool vaccine's evidence needs `calls`")
    call = example.calls[0]
    return Probe(
        call=ToolExample(tool=call.name, arguments=call.arguments), user=example.user, operator=example.operator
    )


class FocusedSensor(Sensor):
    """Passes on only the vaccine's own questions, and remembers the answers for each probe.

    The lab measures one vaccine, so Jev is asked only that vaccine's questions: recordings stay small, cheap, and
    valid when the built-in panels change. A request with none of them is answered empty without calling Jev.
    """

    name: ClassVar[str] = "focused"

    def __init__(self, inner: Sensor, vaccine: Vaccine) -> None:
        self._inner = inner
        self._marker = f"{vaccine.slug}__"
        self._taken: dict[str, Signal] = {}
        self.requests = 0
        self.unanswered = 0
        self.input_tokens = 0

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        mine = [question for question in questions if self._marker in question.key]
        if not mine:
            return SensorReading.empty(self.name)
        self.requests += 1
        try:
            reading = await self._inner.read(state, mine)
        except SensorUnavailable:
            self.unanswered += 1
            raise
        self.input_tokens += reading.input_tokens
        self._taken.update(reading.signals)
        return reading

    async def close(self) -> None:
        await self._inner.close()

    def take(self) -> dict[str, Signal]:
        taken, self._taken = self._taken, {}
        return taken


@dataclass(frozen=True, slots=True)
class Outcome:
    label: str
    expected: bool | None
    fired: bool
    answers: Mapping[str, float] = field(default_factory=dict)


def answers_for(vaccine: Vaccine, signals: Mapping[str, Signal]) -> dict[str, float]:
    """The probability of each of the vaccine's questions (or of its confirmation) in one probe's signals."""
    wanted = {question.key: tuple(question.flag) for question in vaccine.detect.questions}
    if vaccine.detect.confirm == "jev":
        wanted["confirmed"] = ()
    found: dict[str, float] = {}
    for key, options in wanted.items():
        suffix = vaccine.question_key(key)
        values = [signal.probability_of(options) for name, signal in signals.items() if name.endswith(suffix)]
        if values:
            found[key] = max(values)
    return found


def head_probability(vaccine: Vaccine, answers: Mapping[str, float]) -> float | None:
    """What the vaccine's head makes of the answers, exactly as Immune computes it at runtime."""
    detect = vaccine.detect
    if detect.questions:
        if not answers:
            return None
        weights = detect.head.weights if detect.head is not None else {}
        bias = detect.head.bias if detect.head is not None else 0.0
        total = bias + sum(weights.get(key, 1.0) * logit(value) for key, value in answers.items())
        return sigmoid(total)
    if detect.confirm == "jev":
        return answers.get("confirmed")
    return None


class Screening:
    def __init__(self, timeout_ms: int | None = None) -> None:
        self._timeout_ms = timeout_ms

    def run(
        self, loaded: LoadedVaccine, cases: Sequence[tuple[str, Probe, bool | None]], inner: Sensor
    ) -> tuple[list[Outcome], FocusedSensor]:
        focused = FocusedSensor(inner, loaded.vaccine)
        outcomes: list[Outcome] = []

        def after(index: int, result: ProbeResult) -> None:
            label, _, expected = cases[index]
            outcomes.append(Outcome(label, expected, result.fired, answers_for(loaded.vaccine, focused.take())))

        lab = VaccineLab(timeout_ms=self._timeout_ms)
        lab.screen(loaded, [(probe, expected) for _, probe, expected in cases], focused, after)
        return outcomes, focused


# Evidence


@dataclass(frozen=True, slots=True)
class Evidence:
    examples: tuple[Example, ...]
    manifest: DatasetManifest
    problems: tuple[str, ...]

    def labelled(self, vaccine_id: str, split: str | None = None) -> list[tuple[Example, bool]]:
        return [
            (example, example.labels[vaccine_id])
            for example in self.examples
            if vaccine_id in example.labels and (split is None or example.split == split)
        ]


def read_evidence(workspace: Workspace) -> Evidence:
    examples = workspace.examples()
    problems: list[str] = []
    try:
        manifest = DatasetManifest.load(workspace.manifest)
    except Exception as error:
        raise LabError(f"{workspace.vaccine_id}: {error}") from error
    sources = {source.name for source in manifest.sources}
    reflexes = ReflexSuite(Spec.default())
    seen: set[str] = set()
    for example in examples:
        where = f"{workspace.evidence.name}: {example.id}"
        if example.id in seen:
            problems.append(f"{where}: the id is used twice")
        seen.add(example.id)
        if example.source not in sources:
            problems.append(f"{where}: source {example.source!r} is not in manifest.yaml")
        if workspace.vaccine_id not in example.labels:
            problems.append(f"{where}: no label for {workspace.vaccine_id}")
        if example.split not in ("train", "test"):
            problems.append(f"{where}: split must be train or test")
        texts = [example.user, example.reply, *example.data, *(json.dumps(call.arguments) for call in example.calls)]
        for text in texts:
            problems.extend(
                f"{where}: looks like it contains a secret ({match.kind}); evidence must be synthetic"
                for match in reflexes.secrets.find(text)
            )
            problems.extend(
                f"{where}: looks like it contains personal data ({found.kind})"
                for found in reflexes.personal.find(text)
            )
    return Evidence(tuple(examples), manifest, tuple(problems))


def inline_cases(vaccine: Vaccine) -> list[tuple[str, Probe, bool | None]]:
    """The vaccine's own `tests`, as cases; recorded with the evidence so `check` replays Jev's real judgement."""
    cases: list[tuple[str, Probe, bool | None]] = []
    for prefix, examples, expected in (("+", vaccine.tests.positives, True), ("-", vaccine.tests.negatives, False)):
        cases.extend(
            (f"test:{prefix}{index}", Probe.of(example), expected) for index, example in enumerate(examples, 1)
        )
    return cases


def inline_failures(workspace: Workspace, loaded: LoadedVaccine) -> list[str]:
    """Inline tests that don't hold. Vaccines Jev judges replay the cassette; the others run for real."""
    vaccine = loaded.vaccine
    if not asks_jev(vaccine):
        report = VaccineLab().test(loaded)
        return [result.probe.label for result in report.failures] if report.results else ["no tests"]
    cases = inline_cases(vaccine)
    if not cases:
        return ["no tests"]
    outcomes, _ = Screening().run(loaded, cases, ReplaySensor(workspace.cassette))
    return [
        str(probe.label)
        for (_, probe, _), outcome in zip(cases, outcomes, strict=True)
        if outcome.fired is not outcome.expected
    ]


def self_probes(layout: Layout, stage: Stage) -> list[tuple[str, Probe, bool | None]]:
    path = layout.self_corpus(stage)
    if not path.is_file():
        raise LabError(f"no self corpus for the {stage.value} stage at {path.relative_to(layout.root)}")
    return [(f"self:{index}", probe, False) for index, probe in enumerate(SelfSamples().corpus(path), start=1)]


# Measuring


def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else None


def measure(workspace: Workspace, cassette: Path | None = None) -> dict[str, Any]:
    """Replay the vaccine over its test evidence and the self corpus, and describe what it does."""
    cassette = cassette or workspace.cassette
    loaded = workspace.loaded()
    vaccine = loaded.vaccine
    evidence = read_evidence(workspace)
    test = evidence.labelled(vaccine.id, "test")
    cases: list[tuple[str, Probe, bool | None]] = [
        (example.id, probe_for(example, vaccine.stage), label) for example, label in test
    ]
    cases += self_probes(workspace.layout, vaccine.stage)
    inner: Sensor = ReplaySensor(cassette) if asks_jev(vaccine) else MockSensor()
    outcomes, focused = Screening().run(loaded, cases, inner)
    scored = [outcome for outcome in outcomes if not outcome.label.startswith("self:")]
    selfed = [outcome for outcome in outcomes if outcome.label.startswith("self:")]
    positives = [outcome for outcome in scored if outcome.expected]
    negatives = [outcome for outcome in scored if not outcome.expected]
    fired = [outcome for outcome in scored if outcome.fired]
    metrics: dict[str, float | None] = {
        "recall": _ratio(sum(outcome.fired for outcome in positives), len(positives)),
        "precision": _ratio(sum(bool(outcome.expected) for outcome in fired), len(fired)),
        "hard_negative_fpr": _ratio(sum(outcome.fired for outcome in negatives), len(negatives)),
        "self_rate": _ratio(sum(outcome.fired for outcome in selfed), len(selfed)),
        "ece": None,
    }
    if vaccine.detect.questions:
        probabilities = [(head_probability(vaccine, outcome.answers), bool(outcome.expected)) for outcome in scored]
        items = [Scored(probability or 0.0, label) for probability, label in probabilities]
        metrics["ece"] = round(Metrics.ece(items), 4) if items else None
    used = {example.source for example, _ in evidence.labelled(vaccine.id)}
    labels = evidence.labelled(vaccine.id)
    recording = json.loads(workspace.recording.read_text(encoding="utf-8")) if workspace.recording.is_file() else None
    return {
        "id": vaccine.id,
        "version": vaccine.version,
        "maturity": vaccine.maturity,
        "title": vaccine.title,
        "stage": vaccine.stage.value,
        "detector": vaccine.detect.kind + (" + confirm: jev" if vaccine.detect.confirm == "jev" else ""),
        "threshold": vaccine.detect.threshold if asks_jev(vaccine) else None,
        "inputs": {
            "source": digest(workspace.source),
            "evidence": digest(workspace.evidence),
            "cassette": digest(cassette) if asks_jev(vaccine) else None,
            "self": digest(workspace.layout.self_corpus(vaccine.stage)),
        },
        "evidence": {
            "positives": sum(label for _, label in labels),
            "hard_negatives": sum(not label for _, label in labels),
            "sources": len(used),
            "test_positives": len(positives),
            "test_hard_negatives": len(negatives),
        },
        "self": {"samples": len(selfed)},
        "metrics": metrics,
        "cost": {"questions": questions_asked(vaccine)},
        "recording": recording,
        "unanswered": focused.unanswered,
        "misses": sorted(outcome.label for outcome in positives if not outcome.fired),
        "false_alarms": sorted(outcome.label for outcome in [*negatives, *selfed] if outcome.fired),
    }


def asks_jev(vaccine: Vaccine) -> bool:
    return bool(vaccine.detect.questions) or vaccine.detect.confirm == "jev"


def questions_asked(vaccine: Vaccine) -> int:
    return len(vaccine.detect.questions) or (1 if vaccine.detect.confirm == "jev" else 0)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def detector_p95_ms(workspace: Workspace) -> float | None:
    """How long the vaccine's own detector takes on everyday text (deterministic detectors only)."""
    loaded = workspace.loaded()
    detector = loaded.detector
    if detector is None:
        return None
    vaccine = loaded.vaccine
    timings: list[float] = []
    for _, probe, _ in self_probes(workspace.layout, vaccine.stage) * 5:
        call = probe.call
        context = VaccineContext(
            vaccine.id,
            vaccine.stage,
            "vaccine-lab",
            text=probe.text,
            tool=call.tool if call else None,
            arguments=call.arguments if call else {},
        )
        started = time.perf_counter()
        detector.findings(context)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return timings[int(0.95 * (len(timings) - 1))] if timings else None


def gate_failures(card: Mapping[str, Any], p95_ms: float | None) -> list[str]:
    maturity = str(card["maturity"])
    if maturity == "deprecated":
        return []
    gate = GATES[maturity]
    evidence, metrics = card["evidence"], card["metrics"]
    failures: list[str] = []

    def at_least(name: str, value: float | None, limit: float) -> None:
        if value is None or value < limit:
            failures.append(f"{name} is {_show(value)}; {maturity} needs at least {limit:g}")

    def at_most(name: str, value: float | None, limit: float) -> None:
        if value is None or value > limit:
            failures.append(f"{name} is {_show(value)}; {maturity} allows at most {limit:g}")

    at_least("positives", evidence["positives"], gate["positives"])
    at_least("hard negatives", evidence["hard_negatives"], gate["hard_negatives"])
    if "sources" in gate:
        at_least("evidence sources", evidence["sources"], gate["sources"])
    at_least("recall", metrics["recall"], gate["recall"])
    at_most("hard-negative false-positive rate", metrics["hard_negative_fpr"], gate["hard_negative_fpr"])
    at_most("self firing rate", metrics["self_rate"], gate["self_rate"])
    if "ece" in gate and card["detector"].startswith("questions"):
        at_most("calibration error (ECE)", metrics["ece"], gate["ece"])
    at_most("questions per call", card["cost"]["questions"], gate["questions"])
    if p95_ms is not None:
        at_most("detector p95 (ms)", round(p95_ms, 3), gate["p95_ms"])
    if card["unanswered"]:
        failures.append(f"{card['unanswered']} probes have no recorded Jev answer: run `record` again")
    return failures


def _show(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:g}"


# Fitting


@dataclass(frozen=True, slots=True)
class Fit:
    bias: float | None
    weights: dict[str, float]
    threshold: float
    f1: float


def fit(workspace: Workspace) -> Fit:
    """Fit a question vaccine's head, and the threshold of any vaccine Jev judges, on the train split."""
    loaded = workspace.loaded()
    vaccine = loaded.vaccine
    if not asks_jev(vaccine):
        raise LabError(f"{vaccine.id} decides on its own (no Jev questions), so there is nothing to fit")
    train = read_evidence(workspace).labelled(vaccine.id, "train")
    if not train:
        raise LabError(f"{vaccine.id}: no train split in the evidence")
    cases = [(example.id, probe_for(example, vaccine.stage), label) for example, label in train]
    outcomes, focused = Screening().run(loaded, cases, ReplaySensor(workspace.cassette))
    if focused.unanswered:
        raise LabError(f"{vaccine.id}: {focused.unanswered} train rows have no recorded answer; run `record` first")
    keys = [question.key for question in vaccine.detect.questions]
    bias: float | None = None
    weights: dict[str, float] = {}
    if keys:
        head = vaccine.detect.head
        prior = [head.bias if head else 0.0, *((head.weights.get(key, 1.0) if head else 1.0) for key in keys)]
        rows = [
            [1.0, *(logit(outcome.answers[key]) if key in outcome.answers else 0.0 for key in keys)]
            for outcome in outcomes
        ]
        theta = RegularizedLogistic(prior, strength=1.0).fit(rows, [bool(outcome.expected) for outcome in outcomes])
        bias, weights = round(theta[0], 3), {key: round(value, 3) for key, value in zip(keys, theta[1:], strict=True)}
        probabilities = [
            sigmoid(theta[0] + sum(weight * value for weight, value in zip(theta[1:], row[1:], strict=True)))
            for row in rows
        ]
    else:
        probabilities = [outcome.answers.get("confirmed", 0.0) for outcome in outcomes]
    threshold, f1 = best_threshold(probabilities, [bool(outcome.expected) for outcome in outcomes])
    return Fit(bias, weights, threshold, f1)


def best_threshold(probabilities: Sequence[float], labels: Sequence[bool]) -> tuple[float, float]:
    """The threshold with the best F1 on the train split, between 0.5 and 0.95.

    When several thresholds tie, the middle one wins: it keeps the widest margin on both sides, so a slightly
    different wording of a positive or a near miss lands on the same side as in training.
    """
    scored: list[tuple[float, float]] = []
    for step in range(50, 96):
        threshold = step / 100
        predicted = [probability >= threshold for probability in probabilities]
        true_positive = sum(p and label for p, label in zip(predicted, labels, strict=True))
        false_positive = sum(p and not label for p, label in zip(predicted, labels, strict=True))
        false_negative = sum(not p and label for p, label in zip(predicted, labels, strict=True))
        denominator = 2 * true_positive + false_positive + false_negative
        scored.append((threshold, 2 * true_positive / denominator if denominator else 0.0))
    best = max(f1 for _, f1 in scored)
    tied = [threshold for threshold, f1 in scored if f1 >= best - 1e-9]
    return tied[len(tied) // 2], round(best, 4)


def write_fit(workspace: Workspace, result: Fit) -> None:
    document = yaml.safe_load(workspace.source.read_text(encoding="utf-8"))
    detect = document["detect"]
    if result.bias is not None:
        detect["head"] = {"bias": result.bias, "weights": result.weights}
    detect["threshold"] = result.threshold
    write_yaml(workspace.source, document)


def write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    text = yaml.safe_dump(dict(document), sort_keys=False, allow_unicode=True, width=110)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# Recording


def record(workspace: Workspace, sensor: Sensor | None = None, cassette: Path | None = None) -> dict[str, Any]:
    """Ask Jev every question the vaccine asks of its evidence and the self corpus, and keep the answers."""
    cassette = cassette or workspace.cassette
    loaded = workspace.loaded()
    vaccine = loaded.vaccine
    if not asks_jev(vaccine):
        raise LabError(f"{vaccine.id} decides on its own (no Jev questions), so there is nothing to record")
    model = f"scripted ({type(sensor).__name__})" if sensor is not None else SensorSettings().model
    if sensor is None:
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise LabError("recording asks Jev: set TYPESAFE_API_KEY")
        sensor = JevSensor(model, RECORD_TIMEOUT_MS / 1000, key)
    evidence = read_evidence(workspace)
    cases: list[tuple[str, Probe, bool | None]] = [
        (example.id, probe_for(example, vaccine.stage), label) for example, label in evidence.labelled(vaccine.id)
    ]
    cases += inline_cases(vaccine)
    cases += self_probes(workspace.layout, vaccine.stage)
    cassette.unlink(missing_ok=True)
    recorder = RecordingSensor(sensor, cassette)
    _, focused = Screening(timeout_ms=RECORD_TIMEOUT_MS).run(loaded, cases, recorder)
    summary = {
        "date": datetime.date.today().isoformat(),
        "model": model,
        "requests": focused.requests,
        "unanswered": focused.unanswered,
        "input_tokens": focused.input_tokens,
    }
    if cassette == workspace.cassette:
        workspace.recording.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def drift(workspace: Workspace, sensor: Sensor | None = None, tolerance: float = DRIFT_TOLERANCE) -> list[str]:
    """Ask Jev again, into a scratch recording, and report numbers that moved past the committed card."""
    if not workspace.card_json.is_file():
        return ["no committed card to compare with"]
    committed = json.loads(workspace.card_json.read_text(encoding="utf-8"))["metrics"]
    with tempfile.TemporaryDirectory(prefix="immune-drift-") as scratch:
        cassette = Path(scratch) / "cassette.jsonl"
        record(workspace, sensor, cassette)
        fresh = measure(workspace, cassette)
    problems: list[str] = []
    metrics = fresh["metrics"]
    if (metrics["recall"] or 0.0) < (committed["recall"] or 0.0) - tolerance:
        problems.append(f"recall fell from {committed['recall']} to {metrics['recall']}")
    problems.extend(
        f"{name} rose from {committed[name]} to {metrics[name]}"
        for name in ("hard_negative_fpr", "self_rate")
        if (metrics[name] or 0.0) > (committed[name] or 0.0) + tolerance
    )
    return [*problems, *gate_failures(fresh, None)]


# Cards, the bundle and the catalog page


def summary(card: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a card that ships in bundle.json, for `immune vaccines list --library` and `show`."""
    return {
        "maturity": card["maturity"],
        "detector": card["detector"],
        "metrics": {key: card["metrics"].get(key) for key in _SUMMARY},
        "evidence": {key: card["evidence"][key] for key in ("positives", "hard_negatives", "sources")},
        "self_samples": card["self"]["samples"],
        "questions": card["cost"]["questions"],
        "recorded": (card.get("recording") or {}).get("date"),
        "jev_model": (card.get("recording") or {}).get("model"),
    }


def render_card(card: Mapping[str, Any], vaccine: Vaccine) -> str:
    metrics, evidence = card["metrics"], card["evidence"]
    lines = [
        f"# {vaccine.id}",
        "",
        f"**{vaccine.title}** · {card['maturity']} · {card['stage']} stage · {card['detector']}",
        "",
        vaccine.description.strip() or "_No description._",
        "",
        "| Measure | Value | What it means |",
        "| --- | --- | --- |",
        f"| Recall | {_percent(metrics['recall'])} | Share of the {evidence['test_positives']} test positives "
        "it catches |",
        f"| Precision | {_percent(metrics['precision'])} | Share of its firings on the test split that were right |",
        f"| Hard-negative false positives | {_percent(metrics['hard_negative_fpr'])} | Share of the "
        f"{evidence['test_hard_negatives']} test near misses it wrongly catches |",
        f"| Firing rate on everyday traffic | {_percent(metrics['self_rate'])} | Over {card['self']['samples']} "
        "self-corpus samples |",
    ]
    if metrics.get("ece") is not None:
        lines.append(f"| Calibration error (ECE) | {metrics['ece']:.3f} | 0 means a 0.8 really means 80% |")
    lines += [
        f"| Jev questions per call | {card['cost']['questions']} | Added to its stage's existing Jev request |",
        "",
        f"Evidence: {evidence['positives']} positives and {evidence['hard_negatives']} hard negatives from "
        f"{evidence['sources']} source{'' if evidence['sources'] == 1 else 's'}, split into train and test.",
    ]
    recording = card.get("recording")
    if recording:
        lines.append(f"Jev answers recorded on {recording['date']} with `{recording['model']}`.")
    if vaccine.detect.questions or vaccine.detect.confirm:
        lines.append(f"Fires at a probability of {vaccine.detect.threshold:g} or more.")
    lines += ["", "Switch it on:", "", "```yaml", "vaccines:", f"  enabled: [{vaccine.id}]", "```", ""]
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}" if value in (0, 1) else f"{value:.1%}"


def bundle_text(layout: Layout) -> str:
    catalog = LibraryCatalog.from_sources(layout.library) if layout.library.is_dir() else LibraryCatalog()
    cards = {}
    for vaccine_id in catalog.ids:
        card = layout.workspace(vaccine_id).card_json
        if card.is_file():
            cards[vaccine_id] = summary(json.loads(card.read_text(encoding="utf-8")))
    return json.dumps(catalog.bundle(layout.library, cards), indent=1, sort_keys=True) + "\n"


def catalog_text(layout: Layout) -> str:
    catalog = LibraryCatalog.from_sources(layout.library) if layout.library.is_dir() else LibraryCatalog()
    lines = [
        "# Vaccine catalog",
        "",
        "Vaccines that ship with Immune. Every one is **off** until you switch it on, and **observed** until you",
        "enforce it or Immune promotes it on your own traffic (experimental vaccines are never promoted",
        "automatically). Numbers come from each vaccine's measured card in the",
        "[laboratory](../contributing/vaccine-laboratory.md).",
        "",
        "```yaml",
        "vaccines:",
        "  enabled: [immune.health.dosage_instructions]     # one vaccine, everywhere",
        "sites:",
        "  pharmacy-chat:",
        "    vaccines: {enabled: [immune.health.*]}         # a whole domain, at one site",
        "```",
        "",
        "`immune vaccines list --library` shows the same list with each vaccine's state at a site, and",
        "`immune vaccines show <id>` prints a vaccine's card.",
        "",
        "| Vaccine | Stage | Maturity | Recall | Hard-negative FP | Everyday FP | Jev questions |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in catalog.entries:
        vaccine = entry.vaccine
        card_path = layout.workspace(vaccine.id).card_json
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else None
        metrics = card["metrics"] if card else {}
        lines.append(
            f"| `{vaccine.id}`<br>{vaccine.title} | {vaccine.stage.value} | {vaccine.maturity} | "
            f"{_percent(metrics.get('recall'))} | {_percent(metrics.get('hard_negative_fpr'))} | "
            f"{_percent(metrics.get('self_rate'))} | {questions_asked(vaccine)} |"
        )
    if not catalog.entries:
        lines.append("| _None yet_ | | | | | | |")
    return "\n".join(lines) + "\n"


def index(layout: Layout, check_only: bool = False) -> list[str]:
    """Regenerate bundle.json and the catalog page, or (check_only) report whether they are stale."""
    targets = {layout.library / BUNDLE: bundle_text(layout), layout.catalog_page: catalog_text(layout)}
    stale: list[str] = []
    for path, text in targets.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current != text:
            stale.append(path.relative_to(layout.root).as_posix())
            if not check_only:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    return stale


# Scaffolding


def new(layout: Layout, vaccine_id: str, kind: str, stage: Stage, title: str, owner: str) -> list[Path]:
    workspace = layout.workspace(vaccine_id)
    if workspace.source.exists() or workspace.folder.exists():
        raise LabError(f"{vaccine_id} already exists")
    if kind == "tool" and stage is not Stage.TOOL:
        raise LabError("a tool rule runs at the tool stage: use --stage tool")
    subject = _SUBJECT[stage.value]
    templates: dict[str, dict[str, Any]] = {
        "questions": {
            "questions": [{"key": "present", "text": f"{subject} does what this vaccine catches (replace me)."}],
            "head": {"bias": 0.0, "weights": {"present": 1.0}},
            "threshold": 0.8,
        },
        "keywords": {"keywords": ["replace me"], "confirm": "jev", "threshold": 0.8},
        "regex": {"regex": [r"\bREF-\d{4,8}\b"], "confirm": "jev", "threshold": 0.8},
        "tool": {"tool": "replace_me", "argument": {"path": "amount", "greater_than": 100}},
    }
    detect = templates[kind]
    tests: dict[str, Any] = {
        "positives": ["An example that should fire (replace me)."],
        "negatives": ["A near miss that should pass."],
    }
    if stage is Stage.TOOL:
        tests = {
            "positives": [{"tool": "replace_me", "arguments": {"amount": 500}}],
            "negatives": [{"tool": "replace_me", "arguments": {"amount": 5}}],
        }
    document = {
        "id": vaccine_id,
        "version": "1.0.0",
        "maturity": "experimental",
        "title": title,
        "description": "Who needs this, what it catches, and what it deliberately leaves alone (replace me).",
        "stage": stage.value,
        "default": "off",
        "enforcement": "observe",
        "detect": detect,
        "tests": tests,
        "provenance": {"owner": owner},
    }
    write_yaml(workspace.source, document)
    workspace.folder.mkdir(parents=True)
    manifest = {
        "sources": [
            {
                "name": "authored",
                "path": "evidence.jsonl",
                "origin": f"Written for this vaccine by {owner}",
                "license": "Apache-2.0",
                "permitted_use": "Evaluating and training Immune vaccines",
            }
        ]
    }
    write_yaml(workspace.manifest, manifest)
    workspace.evidence.write_text("", encoding="utf-8")
    workspace.notes.write_text(
        f"# {vaccine_id}\n\n## What it is for\n\n## What it deliberately leaves alone\n\n## References\n",
        encoding="utf-8",
    )
    return [workspace.source, workspace.manifest, workspace.evidence, workspace.notes]


# Checking


def check(layout: Layout, vaccine_ids: Iterable[str]) -> list[str]:
    """Everything CI enforces for these vaccines; empty when all is well."""
    problems: list[str] = []
    for vaccine_id in vaccine_ids:
        problems.extend(f"{vaccine_id}: {problem}" for problem in check_one(layout.workspace(vaccine_id)))
    return problems


def check_one(workspace: Workspace) -> list[str]:
    try:
        loaded = workspace.loaded()
    except (LabError, VaccineError) as error:
        return [str(error)]
    vaccine = loaded.vaccine
    problems: list[str] = []
    failing = inline_failures(workspace, loaded)
    if failing:
        problems.append(f"its own tests fail: {', '.join(failing)}")
    if not workspace.notes.is_file():
        problems.append("NOTES.md is missing")
    try:
        evidence = read_evidence(workspace)
        problems.extend(evidence.problems)
        card = measure(workspace)
    except (LabError, VaccineError, ValueError) as error:
        return [*problems, str(error)]
    if not workspace.card_json.is_file():
        problems.append("no card: run `measure`")
    else:
        committed = json.loads(workspace.card_json.read_text(encoding="utf-8"))
        if committed != card:
            changed = sorted(key for key in card if committed.get(key) != card[key])
            problems.append(f"card.json is stale or not reproducible ({', '.join(changed)}): run `measure`")
    rendered = workspace.card_md.read_text(encoding="utf-8") if workspace.card_md.is_file() else None
    if rendered != render_card(card, vaccine):
        problems.append("card.md is stale: run `measure`")
    problems.extend(gate_failures(card, detector_p95_ms(workspace)))
    return problems


def write_card(workspace: Workspace, card: Mapping[str, Any]) -> None:
    workspace.card_json.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    workspace.card_md.write_text(render_card(card, workspace.loaded().vaccine), encoding="utf-8")


# Command line


def main(argv: Sequence[str] | None = None, layout: Layout | None = None) -> int:
    layout = layout or Layout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    scaffold = commands.add_parser("new", help="scaffold a library vaccine and its laboratory folder")
    scaffold.add_argument("id")
    scaffold.add_argument("--kind", choices=_KINDS, default="questions")
    scaffold.add_argument("--stage", choices=[stage.value for stage in Stage if stage is not Stage.OPERATOR])
    scaffold.add_argument("--title", help="one line (default: from the id)")
    scaffold.add_argument("--owner", required=True, help="your GitHub handle, such as @octocat")
    for name, text in (
        ("record", "ask Jev (live) and keep the answers in the cassette"),
        ("fit", "fit the head and threshold from the cassette"),
        ("measure", "replay the cassette and write card.json and card.md"),
        ("check", "replay, compare with the card, and apply the gates (CI)"),
        ("drift", "ask Jev again (live) and compare the numbers with the committed cards"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("ids", nargs="*")
        command.add_argument("--all", action="store_true", help="every vaccine in the library")
    indexing = commands.add_parser("index", help="regenerate bundle.json and the catalog page")
    indexing.add_argument("--check", action="store_true", help="only report whether they are stale")
    args = parser.parse_args(argv)
    try:
        return _run(args, layout)
    except (LabError, VaccineError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, layout: Layout) -> int:
    if args.command == "new":
        stage = Stage(args.stage or ("tool" if args.kind == "tool" else "output"))
        title = args.title or args.id.rsplit(".", 1)[-1].replace("_", " ").capitalize()
        for path in new(layout, args.id, args.kind, stage, title, args.owner):
            print(f"wrote {path.relative_to(layout.root).as_posix()}")
        print("next: write the detector and tests, add evidence rows, then record, fit, measure and index")
        return 0
    if args.command == "index":
        stale = index(layout, check_only=args.check)
        for stale_path in stale:
            print(f"{'stale' if args.check else 'wrote'}: {stale_path}")
        return 1 if args.check and stale else 0
    ids = layout.vaccine_ids() if args.all else list(args.ids)
    if not ids:
        raise LabError("name a vaccine id, or pass --all")
    if args.command == "check":
        problems = check(layout, ids)
        problems += [f"{path} is stale: run `index`" for path in index(layout, check_only=True)]
        for problem in problems:
            print(f"problem: {problem}")
        print(f"{len(ids)} vaccines checked, {len(problems)} problems")
        return 1 if problems else 0
    if args.command == "drift":
        found = [
            f"{vaccine_id}: {problem}"
            for vaccine_id in ids
            if asks_jev(layout.workspace(vaccine_id).loaded().vaccine)
            for problem in drift(layout.workspace(vaccine_id))
        ]
        for problem in found:
            print(f"drift: {problem}")
        print(f"{len(ids)} vaccines re-measured, {len(found)} problems")
        return 1 if found else 0
    for vaccine_id in ids:
        workspace = layout.workspace(vaccine_id)
        if args.command == "record":
            done = record(workspace)
            cost = done["input_tokens"] * JEV_USD_PER_MILLION_INPUT_TOKENS / 1_000_000
            print(f"{vaccine_id}: {done['requests']} Jev requests, {done['unanswered']} unanswered, about ${cost:.4f}")
        elif args.command == "fit":
            result = fit(workspace)
            write_fit(workspace, result)
            head = result.weights or "(confirmation only)"
            print(f"{vaccine_id}: head {head}, threshold {result.threshold} (F1 {result.f1})")
        else:
            card = measure(workspace)
            write_card(workspace, card)
            metrics = card["metrics"]
            print(
                f"{vaccine_id}: recall {_show(metrics['recall'])}, "
                f"hard-negative FP {_show(metrics['hard_negative_fpr'])}, everyday FP {_show(metrics['self_rate'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
