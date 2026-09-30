from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from immune.config.settings import VaccineSettings
from immune.config.spec import Condition, FeatureSpec, HeadSpec, InvariantSpec, QuestionSpec, Spec, ThreatSpec
from immune.config.yaml_io import load_yaml
from immune.errors import ConfigError, SpecError
from immune.types import Stage
from immune.vaccines.detectors import (
    CallableDetector,
    Detector,
    PatternDetector,
    PythonDetector,
    ToolRuleDetector,
    VaccineReflexes,
)
from immune.vaccines.model import Vaccine
from immune.vaccines.patterns import PatternSafety, UnsafePattern

_LOGGER = logging.getLogger("immune")
_ENTRY_POINTS = "immune.vaccines"
_SUFFIXES = (".yaml", ".yml")
_PANELS = {Stage.INPUT: "input", Stage.DATA: "data", Stage.TOOL: "tool", Stage.OUTPUT: "output"}
_CANDIDATES = "candidates"
CUSTOM_INVARIANT = InvariantSpec(name="Custom", statement="Rules the operator added with vaccines.")


class VaccineError(ConfigError):
    pass


@dataclass(frozen=True, slots=True)
class LoadedVaccine:
    vaccine: Vaccine
    source: Path
    digest: str
    detector: Detector | None


@dataclass(frozen=True, slots=True)
class VaccineBundle:
    vaccines: tuple[LoadedVaccine, ...] = ()

    @property
    def label(self) -> str:
        combined = hashlib.sha256("".join(sorted(item.digest for item in self.vaccines)).encode()).hexdigest()
        return f"vaccines-{combined[:8]}"

    @property
    def ids(self) -> list[str]:
        return [item.vaccine.id for item in self.vaccines]

    def get(self, vaccine_id: str) -> LoadedVaccine | None:
        return next((item for item in self.vaccines if item.vaccine.id == vaccine_id), None)

    def compile(self, spec: Spec) -> Spec:
        if not self.vaccines:
            return spec
        threats: dict[str, ThreatSpec] = {}
        heads: dict[str, HeadSpec] = {}
        questions: dict[str, list[QuestionSpec]] = {}
        for item in self.vaccines:
            vaccine = item.vaccine
            if vaccine.invariant != "U0" and vaccine.invariant not in spec.invariants:
                raise VaccineError(f"{item.source}: invariant {vaccine.invariant} does not exist; use U0 to U10")
            detect = vaccine.detect
            confirmed = detect.confirm == "jev"
            threats[vaccine.id] = ThreatSpec(
                id=vaccine.id,
                invariant=vaccine.invariant,
                stage=vaccine.stage,
                detector="jev" if detect.questions else "candidate" if confirmed else "deterministic",
                severity=vaccine.severity,
                threshold=detect.threshold if detect.questions or confirmed else 1.0,
                frameworks=vaccine.frameworks,
                actions=vaccine.respond.actions(vaccine.stage),
            )
            if confirmed:
                key = vaccine.question_key("confirmed")
                questions.setdefault(_CANDIDATES, []).append(
                    QuestionSpec(key=key, kind="noul", text=vaccine.confirmation)
                )
                heads[vaccine.id] = HeadSpec(features=(FeatureSpec(signal=key),))
                continue
            if not detect.questions:
                continue
            when = Condition(sites=vaccine.applies_to.sites, organs=vaccine.applies_to.organs)
            panel = questions.setdefault(_PANELS[vaccine.stage], [])
            weights = detect.head.weights if detect.head is not None else {}
            features: list[FeatureSpec] = []
            for question in detect.questions:
                key = vaccine.question_key(question.key)
                panel.append(
                    QuestionSpec(key=key, kind=question.kind, text=question.text, options=question.options, when=when)
                )
                features.append(FeatureSpec(signal=key, options=question.flag, weight=weights.get(question.key, 1.0)))
            bias = detect.head.bias if detect.head is not None else 0.0
            heads[vaccine.id] = HeadSpec(bias=bias, features=tuple(features))
        try:
            return spec.with_vaccines(threats, heads, questions, {"U0": CUSTOM_INVARIANT}, self.label)
        except SpecError as error:
            raise VaccineError(str(error)) from error

    def reflexes(self) -> VaccineReflexes:
        return VaccineReflexes([item.detector for item in self.vaccines if item.detector is not None])

    def messages(self) -> dict[str, str]:
        return {item.vaccine.id: item.vaccine.respond.message for item in self.vaccines if item.vaccine.respond.message}

    def enforced(self) -> frozenset[str]:
        return frozenset(item.vaccine.id for item in self.vaccines if item.vaccine.enforcement == "enforce")

    def default_off(self) -> frozenset[str]:
        return frozenset(item.vaccine.id for item in self.vaccines if item.vaccine.default == "off")


class VaccineLoader:
    def __init__(self, settings: VaccineSettings, root: Path | None = None) -> None:
        self._settings = settings
        self._root = root or Path.cwd()
        self._safety = PatternSafety()

    def load(self) -> VaccineBundle:
        loaded: dict[str, LoadedVaccine] = {}
        for path in self._files():
            item = self.load_file(path)
            existing = loaded.get(item.vaccine.id)
            if existing is not None:
                raise VaccineError(
                    f"vaccine id {item.vaccine.id} is defined twice: {existing.source} and {item.source}"
                )
            loaded[item.vaccine.id] = item
        if loaded:
            _LOGGER.info("immune: loaded %d vaccines: %s", len(loaded), ", ".join(sorted(loaded)))
        return VaccineBundle(tuple(loaded[key] for key in sorted(loaded)))

    def load_file(self, path: Path) -> LoadedVaccine:
        try:
            raw = path.read_text(encoding="utf-8")
            document = load_yaml(raw)
        except (OSError, yaml.YAMLError) as error:
            raise VaccineError(f"cannot read vaccine {path}: {error}") from error
        if not isinstance(document, dict):
            raise VaccineError(f"{path}: a vaccine file must contain one mapping with id, title, stage and detect")
        try:
            vaccine = Vaccine.model_validate(document)
        except ValidationError as error:
            raise VaccineError(f"{path}: {self._describe(error)}") from error
        return LoadedVaccine(
            vaccine=vaccine,
            source=path,
            digest=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            detector=self._detector(vaccine, path),
        )

    def _detector(self, vaccine: Vaccine, path: Path) -> Detector | None:
        kind = vaccine.detect.kind
        if kind == "questions":
            return None
        if kind == "pattern":
            for pattern in vaccine.detect.regex:
                try:
                    self._safety.check(pattern)
                except UnsafePattern as error:
                    raise VaccineError(f"{path}: detect.regex: {error}") from error
            return PatternDetector(vaccine)
        if kind == "tool":
            return ToolRuleDetector(vaccine)
        assert vaccine.detect.python is not None
        return CallableDetector(vaccine, self._function(vaccine.detect.python, path))

    @staticmethod
    def _function(reference: str, path: Path) -> PythonDetector:
        module_name, _, attribute = reference.partition(":")
        try:
            function: Any = getattr(importlib.import_module(module_name), attribute)
        except (ImportError, AttributeError) as error:
            raise VaccineError(
                f"{path}: detect.python {reference!r} cannot be imported ({error}); "
                "make sure the module is installed or on PYTHONPATH"
            ) from error
        if not callable(function):
            raise VaccineError(f"{path}: detect.python {reference!r} is not callable")
        detector: PythonDetector = function
        return detector

    def _files(self) -> Iterator[Path]:
        for entry in self._paths():
            path = entry if entry.is_absolute() else self._root / entry
            if path.is_dir():
                yield from sorted(item for item in path.rglob("*") if item.suffix in _SUFFIXES and item.is_file())
            elif path.is_file():
                yield path
            else:
                raise VaccineError(f"vaccine path {path} does not exist")

    def _paths(self) -> Iterator[Path]:
        yield from self._settings.paths
        if self._settings.entry_points:
            yield from self._entry_point_paths()

    @staticmethod
    def _entry_point_paths() -> Iterator[Path]:
        for entry_point in importlib.metadata.entry_points(group=_ENTRY_POINTS):
            try:
                provided = entry_point.load()
                values = provided() if callable(provided) else provided
            except Exception as error:
                raise VaccineError(f"vaccine entry point {entry_point.name} failed to load: {error}") from error
            for value in _as_paths(values):
                _LOGGER.info("immune: vaccines from entry point %s: %s", entry_point.name, value)
                yield value

    @staticmethod
    def _describe(error: ValidationError) -> str:
        problems = []
        for issue in error.errors():
            location = ".".join(str(part) for part in issue["loc"]) or "file"
            message = str(issue["msg"]).removeprefix("Value error, ")
            problems.append(f"{location}: {message}")
        return "; ".join(problems)


def _as_paths(values: Any) -> Iterable[Path]:
    if isinstance(values, (str, PathLike)):
        return [Path(values)]
    if isinstance(values, Sequence):
        return [Path(value) for value in values]
    raise VaccineError("a vaccine entry point must provide a path or a list of paths")
