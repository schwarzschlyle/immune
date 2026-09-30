from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from immune.types import Verdict


class ImmuneError(Exception):
    pass


class ConfigError(ImmuneError):
    pass


class SpecError(ImmuneError):
    pass


class CodecError(ImmuneError):
    pass


class SensorError(ImmuneError):
    pass


class SensorUnavailable(SensorError):
    pass


class Blocked(ImmuneError):
    def __init__(self, verdict: Verdict) -> None:
        super().__init__(verdict.explanation or f"blocked by immune: {verdict.action.value}")
        self.verdict = verdict
