from __future__ import annotations

import argparse
from pathlib import Path

from immune.cli.commands.base import Command
from immune.cli.console import Console
from immune.config.loader import SettingsLoader
from immune.config.spec import Spec
from immune.errors import ConfigError
from immune.evidence import (
    CalibrationStudy,
    DatasetManifest,
    DriftGate,
    ExampleSet,
    FeatureCollector,
    FeatureFile,
    HeadTrainer,
    ModelCard,
)
from immune.evidence.collect import SignalRow
from immune.evidence.examples import Example
from immune.evidence.questions import QuestionSearch, QuestionVariants, VariantSensor
from immune.heads.artifact import HeadsArtifact
from immune.sensing.jev import JevSensor
from immune.sensing.offline import MockSensor, RecordingSensor, ReplaySensor
from immune.sensing.sensor import Sensor


class EvidenceCommand(Command):
    name = "evidence"
    help = "collect features from labeled examples, train heads and check them for drift"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        actions = parser.add_subparsers(dest="action", required=True)
        collect = actions.add_parser("collect", help="run labeled examples through immune and record head features")
        collect.add_argument("examples", type=Path, help="examples JSONL file or a datasets.yaml manifest")
        collect.add_argument("--out", type=Path, required=True)
        collect.add_argument(
            "--sensor", default="jev", help="jev, mock, replay:<cassette> or record:<cassette> (record uses Jev)"
        )
        fit = actions.add_parser("fit", help="train heads and write an artifact and model card")
        fit.add_argument("features", type=Path)
        fit.add_argument("--out", type=Path, required=True)
        fit.add_argument("--card", type=Path)
        fit.add_argument("--jev-model", help="defaults to the configured sensor model")
        drift = actions.add_parser("drift", help="fail when a candidate artifact regresses against a baseline")
        drift.add_argument("baseline", type=Path)
        drift.add_argument("candidate", type=Path)
        drift.add_argument("--tolerance", type=float, default=0.02)
        study = actions.add_parser("study", help="measure how calibrated each Jev question is on its own")
        study.add_argument("features", type=Path)
        questions = actions.add_parser("questions", help="rank alternative phrasings of Jev questions")
        questions.add_argument("examples", type=Path)
        questions.add_argument("--variants", type=Path, required=True, help="YAML: question key -> alternative texts")
        questions.add_argument("--sensor", default="jev")

    def run(self, args: argparse.Namespace, console: Console) -> int:
        actions = {
            "collect": self._collect,
            "fit": self._fit,
            "drift": self._drift,
            "study": self._study,
            "questions": self._questions,
        }
        return actions[args.action](args, console)

    def _collect(self, args: argparse.Namespace, console: Console) -> int:
        examples = self._examples(args.examples)
        count = FeatureFile.write(args.out, FeatureCollector(self._sensors(args.sensor)).collect(examples))
        console.line(f"wrote {count} rows to {args.out}")
        return 0

    def _fit(self, args: argparse.Namespace, console: Console) -> int:
        features, _ = FeatureFile.read(args.features)
        model = args.jev_model or SettingsLoader().load().sensor.model
        artifact, reports = HeadTrainer(Spec.default(), model).train(features)
        artifact.save(args.out)
        if args.card is not None:
            args.card.write_text(ModelCard.render(artifact, reports), encoding="utf-8")
        console.table(
            ("threat", "train", "test", "auc", "ece", "threshold", "target met"),
            [
                (
                    report.threat,
                    report.rows.get("train", 0),
                    report.rows.get("test", 0),
                    report.metrics.get("roc_auc", "-"),
                    report.metrics.get("ece", "-"),
                    report.threshold if report.threshold is not None else report.skipped,
                    "yes" if report.target_met else "NO",
                )
                for report in reports
            ],
        )
        console.line(f"wrote {len(artifact.heads)} trained heads to {args.out}")
        return 0

    @staticmethod
    def _drift(args: argparse.Namespace, console: Console) -> int:
        problems = DriftGate(args.tolerance).regressions(
            HeadsArtifact.load(args.baseline), HeadsArtifact.load(args.candidate)
        )
        for problem in problems:
            console.line(f"regression: {problem}")
        console.line("no regressions" if not problems else f"{len(problems)} regressions")
        return 1 if problems else 0

    @staticmethod
    def _study(args: argparse.Namespace, console: Console) -> int:
        _, signals = FeatureFile.read(args.features)
        results = CalibrationStudy(Spec.default()).run(signals)
        console.table(
            ("question", "samples", "positives", "auc", "ece", "rephrase"),
            [
                (item.key, item.samples, item.positives, item.roc_auc, item.ece, "yes" if item.miscalibrated else "")
                for item in results
            ],
        )
        return 0

    def _questions(self, args: argparse.Namespace, console: Console) -> int:
        variants = QuestionVariants.load(args.variants)
        sensors = self._sensors(args.sensor)
        collector = FeatureCollector(lambda example: VariantSensor(sensors(example), variants))
        signals = [row for row in collector.collect(self._examples(args.examples)) if isinstance(row, SignalRow)]
        rows: list[tuple[object, ...]] = []
        for key, results in sorted(QuestionSearch(Spec.default()).rank(signals).items()):
            rows.extend(
                (
                    key,
                    "current" if result.variant == 0 else f"variant {result.variant}",
                    result.samples,
                    result.roc_auc,
                    result.ece,
                    "best" if position == 0 else "",
                )
                for position, result in enumerate(results)
            )
        console.table(("question", "phrasing", "samples", "auc", "ece", ""), rows)
        return 0

    @staticmethod
    def _examples(path: Path) -> list[Example]:
        if path.suffix in (".yaml", ".yml"):
            return list(DatasetManifest.load(path).examples(path.parent))
        return list(ExampleSet.read(path))

    @staticmethod
    def _sensors(choice: str) -> SensorChoice:
        return SensorChoice(choice)


class SensorChoice:
    def __init__(self, choice: str) -> None:
        kind, _, target = choice.partition(":")
        if kind not in ("jev", "mock", "replay", "record") or (kind in ("replay", "record") and not target):
            raise ConfigError(f"unknown sensor {choice!r}; use jev, mock, replay:<cassette> or record:<cassette>")
        self._kind, self._target = kind, Path(target) if target else None
        self._settings = SettingsLoader().load().sensor

    def __call__(self, example: Example) -> Sensor:
        if self._kind == "mock":
            return MockSensor()
        if self._kind == "replay":
            assert self._target is not None
            return ReplaySensor(self._target)
        live = JevSensor(self._settings.model, self._settings.timeout_s * 5, self._settings.api_key)
        if self._kind == "record":
            assert self._target is not None
            return RecordingSensor(live, self._target)
        return live
