from __future__ import annotations

import fnmatch
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from immune.core.conversation import ToolCall
from immune.reflexes.findings import Finding
from immune.telemetry.throttle import ThrottledLog
from immune.types import Stage
from immune.vaccines.model import ArgumentRule, Vaccine
from immune.vaccines.patterns import TextMatcher

_LOGGER = logging.getLogger("immune")
_THROTTLED = ThrottledLog(_LOGGER)
_MISSING = object()


@dataclass(frozen=True, slots=True)
class VaccineContext:
    vaccine: str
    stage: Stage
    site: str
    text: str = ""
    tool: str | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    tainted: bool = False


PythonDetector = Callable[[VaccineContext], "bool | str | Iterable[str] | None"]


class Detector(ABC):
    def __init__(self, vaccine: Vaccine) -> None:
        self.vaccine = vaccine

    @abstractmethod
    def findings(self, context: VaccineContext) -> list[Finding]: ...


class PatternDetector(Detector):
    def __init__(self, vaccine: Vaccine) -> None:
        super().__init__(vaccine)
        self._matcher: TextMatcher | None = None

    def findings(self, context: VaccineContext) -> list[Finding]:
        # Compiled on first use, so a library vaccine that stays switched off never compiles its patterns.
        if self._matcher is None:
            self._matcher = TextMatcher(self.vaccine.detect.keywords, self.vaccine.detect.regex)
        text = context.text
        if context.tool is not None:
            text = f"{context.tool} {json.dumps(context.arguments, ensure_ascii=False, default=str)}"
        return [Finding(self.vaccine.id, evidence, span) for evidence, span in self._matcher.matches(text)]


class ToolRuleDetector(Detector):
    def findings(self, context: VaccineContext) -> list[Finding]:
        pattern = self.vaccine.detect.tool
        if context.tool is None or pattern is None or not fnmatch.fnmatchcase(context.tool, pattern):
            return []
        rule = self.vaccine.detect.argument
        if rule is None:
            return [Finding(self.vaccine.id, f"call to {context.tool}")]
        value = self._lookup(context.arguments, rule.path)
        if value is _MISSING or not self._holds(rule, value):
            return []
        return [Finding(self.vaccine.id, f"{context.tool}: {rule.path}={value!r}")]

    @staticmethod
    def _lookup(arguments: Mapping[str, Any], path: str) -> Any:
        value: Any = arguments
        for part in path.split("."):
            if isinstance(value, Mapping) and part in value:
                value = value[part]
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                return _MISSING
        return value

    @staticmethod
    def _holds(rule: ArgumentRule, value: Any) -> bool:
        checks: list[bool] = []
        if rule.equals is not None:
            checks.append(value == rule.equals)
        if rule.one_of:
            checks.append(value in rule.one_of)
        if rule.contains is not None:
            checks.append(rule.contains.lower() in str(value).lower())
        number = _number(value)
        if rule.greater_than is not None:
            checks.append(number is not None and number > rule.greater_than)
        if rule.less_than is not None:
            checks.append(number is not None and number < rule.less_than)
        return all(checks)


class CallableDetector(Detector):
    def __init__(self, vaccine: Vaccine, function: PythonDetector) -> None:
        super().__init__(vaccine)
        self._function = function
        self._budget_s = vaccine.detect.budget_ms / 1000

    def findings(self, context: VaccineContext) -> list[Finding]:
        started = time.perf_counter()
        try:
            result = self._function(context)
        except Exception as error:
            _THROTTLED.warning(
                f"vaccine:{self.vaccine.id}:error",
                "immune: vaccine %s raised %s: %s; skipped for this call",
                self.vaccine.id,
                type(error).__name__,
                error,
            )
            return []
        elapsed = time.perf_counter() - started
        if elapsed > self._budget_s:
            _THROTTLED.warning(
                f"vaccine:{self.vaccine.id}:slow",
                "immune: vaccine %s took %.0f ms (budget %.0f ms)",
                self.vaccine.id,
                elapsed * 1000,
                self._budget_s * 1000,
            )
        return [Finding(self.vaccine.id, evidence) for evidence in self._evidence(result)]

    def _evidence(self, result: bool | str | Iterable[str] | None) -> list[str]:
        if result is None or result is False:
            return []
        if result is True:
            return [f"{self.vaccine.id} matched"]
        if isinstance(result, str):
            return [result] if result else []
        return [str(item) for item in result if item]


class DeferredCallableDetector(Detector):
    """A Python detector imported the first time it runs, for library vaccines that are usually switched off."""

    def __init__(self, vaccine: Vaccine, resolve: Callable[[], PythonDetector]) -> None:
        super().__init__(vaccine)
        self._resolve = resolve
        self._detector: CallableDetector | None = None
        self._failed = False

    def findings(self, context: VaccineContext) -> list[Finding]:
        if self._failed:
            return []
        if self._detector is None:
            try:
                self._detector = CallableDetector(self.vaccine, self._resolve())
            except Exception as error:
                self._failed = True
                _LOGGER.warning("immune: vaccine %s is switched off because it cannot load: %s", self.vaccine.id, error)
                return []
        return self._detector.findings(context)


class VaccineReflexes:
    def __init__(self, detectors: Sequence[Detector] = ()) -> None:
        self._text: dict[Stage, list[Detector]] = {}
        self._tool: list[Detector] = []
        for detector in detectors:
            if detector.vaccine.stage is Stage.TOOL:
                self._tool.append(detector)
            else:
                self._text.setdefault(detector.vaccine.stage, []).append(detector)

    @property
    def empty(self) -> bool:
        return not self._text and not self._tool

    def text_findings(
        self, stage: Stage, text: str, site: str, organs: frozenset[str], off: frozenset[str] = frozenset()
    ) -> list[Finding]:
        """Findings of the vaccines that apply here. Vaccines in `off` are switched off at this site and don't run."""
        if not text:
            return []
        return [
            finding
            for detector in self._text.get(stage, [])
            if detector.vaccine.id not in off and _applies(detector.vaccine, site, organs)
            for finding in detector.findings(VaccineContext(detector.vaccine.id, stage, site, text=text))
        ]

    def call_findings(
        self, call: ToolCall, site: str, organs: frozenset[str], tainted: bool, off: frozenset[str] = frozenset()
    ) -> list[Finding]:
        found: list[Finding] = []
        for detector in self._tool:
            if detector.vaccine.id in off or not _applies(detector.vaccine, site, organs):
                continue
            context = VaccineContext(
                detector.vaccine.id, Stage.TOOL, site, tool=call.name, arguments=call.arguments, tainted=tainted
            )
            found.extend(detector.findings(context))
        return found


def _applies(vaccine: Vaccine, site: str, organs: frozenset[str]) -> bool:
    scope = vaccine.applies_to
    if scope.sites and not any(fnmatch.fnmatchcase(site, pattern) for pattern in scope.sites):
        return False
    return not scope.organs or bool(set(scope.organs) & organs)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
