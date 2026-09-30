from __future__ import annotations

import importlib
import pkgutil
import re
import time

import pytest

import immune
from immune.config.spec import Spec

SMALL, LARGE = 10_000, 100_000
LIMIT_S = 1.0
GROWTH = 25.0
# Timings below this are dominated by scheduler and cache noise on shared CI runners.
NOISE_FLOOR_S = 0.001
REPEATS = 5
FAMILIES = (
    "a",
    "a@",
    "1.",
    "a.",
    "<a ",
    "[(",
    "%41",
    "a/",
    " ",
    "-",
    "=",
    "\\",
    "a-a_",
    "![",
    "](",
    "{{",
    "'or'1'='1",
    "../",
    "at ",
    "sk-",
    "$(",
    "a" + " " * 50,
    "a at ",
)


def compiled_patterns() -> dict[str, re.Pattern[str]]:
    found: dict[str, re.Pattern[str]] = {}
    for module_info in pkgutil.walk_packages(immune.__path__, "immune."):
        if module_info.name.startswith(("immune.cli.main", "immune.testing.pytest_plugin")):
            continue
        module = importlib.import_module(module_info.name)
        for name, value in vars(module).items():
            if isinstance(value, re.Pattern) and isinstance(value.pattern, str):
                found[f"{module_info.name}.{name}"] = value
    reflexes = Spec.default().reflexes
    for group in ("secrets", "template_tokens", "guard_addressed", "sql", "paths"):
        for name, pattern in getattr(reflexes, group).items():
            found[f"spec.{group}.{name}"] = re.compile(pattern)
    return found


PATTERNS = compiled_patterns()


def scan_time(pattern: re.Pattern[str], text: str, good_enough: float = 0.0) -> float:
    """The fastest of a few scans: one pause on a busy machine shouldn't look like backtracking."""
    best = float("inf")
    for _ in range(REPEATS):
        started = time.perf_counter()
        for _ in pattern.finditer(text):
            pass
        best = min(best, time.perf_counter() - started)
        if best <= good_enough:
            break
    return best


def test_the_audit_sees_every_reflex_family() -> None:
    assert len(PATTERNS) >= 60


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_patterns_stay_linear_on_adversarial_input(name: str) -> None:
    pattern = PATTERNS[name]
    for family in FAMILIES:
        small = scan_time(pattern, (family * (SMALL // len(family) + 1))[:SMALL])
        allowed = max(small, NOISE_FLOOR_S) * GROWTH
        large = scan_time(pattern, (family * (LARGE // len(family) + 1))[:LARGE], good_enough=allowed)
        assert large < LIMIT_S, f"{name} took {large:.3f}s on {family!r}"
        assert large <= allowed, f"{name} grew super-linearly on {family!r}: {small:.5f}s -> {large:.5f}s"
