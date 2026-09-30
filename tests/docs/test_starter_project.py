from __future__ import annotations

import importlib.util
import io
import shlex
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from immune.cli.console import Console
from immune.cli.main import CommandLine
from immune.testing import ImmuneHarness, ScenarioLibrary, ScenarioRunner

ROOT = Path(__file__).resolve().parents[2]
STARTER = ROOT / "examples" / "starter"
GUIDE = ROOT / "docs" / "guides" / "setup.md"
CI_FILES = (".github/workflows/immune.yml", ".gitlab-ci.yml", ".pre-commit-config.yaml")


def files() -> Iterator[Path]:
    for path in sorted(STARTER.rglob("*")):
        if path.is_file() and path.name != "README.md" and "__pycache__" not in path.parts:
            yield path


def run(*argv: str) -> tuple[int, str]:
    buffer = io.StringIO()
    code = CommandLine(console=Console(buffer)).run(list(argv))
    return code, buffer.getvalue()


@pytest.fixture
def in_starter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(STARTER)
    monkeypatch.delenv("IMMUNE_CONFIG", raising=False)
    monkeypatch.delenv("IMMUNE_MODE", raising=False)


@pytest.mark.parametrize("path", list(files()), ids=lambda path: str(path.relative_to(STARTER)))
def test_the_guide_shows_every_starter_file_verbatim(path: Path) -> None:
    assert path.read_text(encoding="utf-8").strip() in GUIDE.read_text(encoding="utf-8")


@pytest.mark.usefixtures("in_starter")
def test_the_starter_configuration_and_vaccines_pass() -> None:
    assert run("config", "validate", "immune.yaml")[0] == 0
    code, output = run("vaccines", "test")
    assert code == 0, output
    assert "2 vaccines, 4 examples: all passed" in output
    code, output = run(
        "vaccines",
        "trial",
        "bobs.no_competitor_mentions",
        "--corpus",
        "tests/fixtures/replies.jsonl",
        "--max-rate",
        "0.01",
    )
    assert code == 0, output


def test_the_starter_scenarios_pass(tmp_path: Path) -> None:
    scenarios = ScenarioLibrary([STARTER / "scenarios"]).all()
    assert [scenario.id for scenario in scenarios] == ["bobs.menu-question", "bobs.off-topic"]
    for scenario in scenarios:
        result = ScenarioRunner(tmp_path).run(scenario)
        assert result.passed, (scenario.id, result.failures)


def test_the_starter_tests_pass(immune_harness: Callable[..., ImmuneHarness]) -> None:
    spec = importlib.util.spec_from_file_location("starter_tests", STARTER / "tests" / "test_immune.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tests = [value for name, value in vars(module).items() if name.startswith("test_") and callable(value)]
    assert len(tests) == 4
    for test in tests:
        test(immune_harness)


def test_the_app_runs_from_any_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IMMUNE_CONFIG", str(STARTER / "immune.yaml"))
    code, output = run("vaccines", "list", "--custom")
    assert code == 0
    assert "2 vaccines loaded" in output


@pytest.mark.parametrize("name", CI_FILES)
def test_ci_files_only_use_real_commands(name: str) -> None:
    document: Any = yaml.safe_load((STARTER / name).read_text(encoding="utf-8"))
    commands = [line for line in _strings(document) if line.startswith("immune ")]
    assert commands
    parser = CommandLine().parser()
    for command in commands:
        parser.parse_args(shlex.split(command)[1:])


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
