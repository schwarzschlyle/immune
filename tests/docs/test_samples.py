from __future__ import annotations

import ast
import importlib
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

import immune
from immune.config.loader import SettingsLoader
from immune.config.settings import Settings
from immune.testing import ImmuneHarness, ScenarioRunner
from immune.testing.scenarios import Scenario
from immune.vaccines import Vaccine

ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
_FENCE = re.compile(r"^```(?P<language>[a-z]+)[^\n]*\n(?P<body>.*?)^```", re.M | re.S)
_SETTINGS_KEYS = frozenset(Settings.model_fields)


@dataclass(frozen=True, slots=True)
class Sample:
    document: Path
    line: int
    language: str
    body: str

    @property
    def label(self) -> str:
        return f"{self.document.relative_to(ROOT)}:{self.line}"

    @property
    def is_test(self) -> bool:
        # Samples that locate files through __file__ are copies of examples/starter, tested there.
        return "def test_" in self.body and "__file__" not in self.body

    def yaml_document(self) -> Any:
        return yaml.safe_load(self.body)


def samples(language: str) -> Iterator[Sample]:
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        for match in _FENCE.finditer(text):
            if match.group("language") == language:
                yield Sample(document, text.count("\n", 0, match.start()) + 1, language, match.group("body"))


def parametrize(language: str, keep: Callable[[Sample], bool] = lambda _: True) -> Any:
    chosen = [sample for sample in samples(language) if keep(sample)]
    return pytest.mark.parametrize("sample", chosen, ids=[sample.label for sample in chosen])


class ApiReferences(ast.NodeVisitor):
    def __init__(self) -> None:
        self.attributes: set[str] = set()
        self.imports: set[tuple[str, str]] = set()

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "immune":
            self.attributes.add(node.attr)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and node.module.split(".")[0] == "immune":
            self.imports.update((node.module, alias.name) for alias in node.names)


def is_settings(sample: Sample) -> bool:
    document = sample.yaml_document()
    return isinstance(document, dict) and bool(document) and set(document) <= _SETTINGS_KEYS


def is_vaccine(sample: Sample) -> bool:
    document = sample.yaml_document()
    return isinstance(document, dict) and {"id", "stage", "detect"} <= set(document)


def is_scenario(sample: Sample) -> bool:
    document = sample.yaml_document()
    return isinstance(document, dict) and {"id", "operator", "turns"} <= set(document)


@parametrize("python")
def test_python_samples_compile(sample: Sample) -> None:
    compile(sample.body, sample.label, "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)


@parametrize("python")
def test_python_samples_use_the_real_api(sample: Sample) -> None:
    references = ApiReferences()
    references.visit(compile(sample.body, sample.label, "exec", ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT))
    missing = [name for name in sorted(references.attributes) if not hasattr(immune, name)]
    for module, name in sorted(references.imports):
        if not hasattr(importlib.import_module(module), name):
            missing.append(f"{module}.{name}")
    assert not missing, f"{sample.label} refers to {missing}"


@parametrize("python", lambda sample: sample.is_test)
def test_test_samples_pass(sample: Sample, immune_harness: Callable[..., ImmuneHarness]) -> None:
    namespace: dict[str, Any] = {}
    exec(compile(sample.body, sample.label, "exec"), namespace)
    tests = [value for name, value in namespace.items() if name.startswith("test_") and callable(value)]
    assert tests
    for test in tests:
        test(immune_harness)


@parametrize("yaml", is_settings)
def test_configuration_samples_validate(sample: Sample) -> None:
    SettingsLoader(environ={}).load(sample.yaml_document())


@parametrize("yaml", is_vaccine)
def test_vaccine_samples_validate(sample: Sample) -> None:
    Vaccine.model_validate(sample.yaml_document())


@parametrize("yaml", is_scenario)
def test_scenario_samples_pass(sample: Sample, tmp_path: Path) -> None:
    scenario = Scenario.model_validate(sample.yaml_document())
    result = ScenarioRunner(tmp_path).run(scenario)
    assert result.passed, result.failures
