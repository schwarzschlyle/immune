from __future__ import annotations

from dataclasses import dataclass

from immune.core.conversation import Span


@dataclass(frozen=True, slots=True)
class Finding:
    threat: str
    evidence: str
    span: Span | None = None
    payload: str | None = None
    subject: str | None = None


@dataclass(frozen=True, slots=True)
class Cleaned:
    text: str
    hidden: str = ""
    findings: tuple[Finding, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.findings)
