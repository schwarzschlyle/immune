"""Candidates: what deterministic code found, handed to Jev to decide.

Reflexes and organs locate things precisely (a link, a key-shaped string, a copied passage, a repeated tool call), but
they don't decide whether it is a threat. Each finding for a candidate threat becomes a Candidate, Jev is asked a
question about it, and a head turns the answer into a probability. Only user-written vaccines and checks already
derived from Jev's answers stay deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from immune.config.spec import Spec
from immune.reflexes.findings import Finding
from immune.types import Stage

_EXCERPT_CONTEXT = 100
_PLACEHOLDER = re.compile(r"example|sample|dummy|placeholder|your[-_ ]?(?:api[-_ ]?)?key|x{4,}|\*{4,}|0{8,}", re.I)
_SECRET_THREATS = frozenset({"output.secret_leak"})


@dataclass(frozen=True, slots=True)
class Candidate:
    """One nominated finding. Its id scopes Jev's answers (``cand_3__secret_real``).

    ``stage`` is where it was found: the user message, a data item, a tool call or description, or the reply.
    """

    id: str
    finding: Finding
    stage: Stage
    subject: str | None
    where: str
    excerpt: str
    hints: Mapping[str, object] = field(default_factory=dict)

    @property
    def threat(self) -> str:
        return self.finding.threat


class Nominator:
    """Splits findings into candidates, which Jev decides, and findings that stay deterministic."""

    def __init__(self, spec: Spec) -> None:
        self._candidate_threats = frozenset(
            threat.id for threat in spec.threats.values() if threat.detector == "candidate"
        )

    def is_candidate(self, finding: Finding) -> bool:
        return finding.threat in self._candidate_threats

    def deterministic(self, findings: Iterable[Finding]) -> list[Finding]:
        return [finding for finding in findings if not self.is_candidate(finding)]

    def nominate(
        self,
        findings: Iterable[Finding],
        numbering: CandidateNumbers,
        stage: Stage,
        where: str,
        text: str = "",
        subject: str | None = None,
    ) -> list[Candidate]:
        return [
            Candidate(
                id=numbering.next(),
                finding=finding,
                stage=stage,
                subject=finding.subject or subject,
                where=where,
                excerpt=_excerpt(finding, text),
                hints=_hints(finding, text),
            )
            for finding in findings
            if self.is_candidate(finding)
        ]


class CandidateNumbers:
    def __init__(self) -> None:
        self._count = 0

    def next(self) -> str:
        self._count += 1
        return f"cand_{self._count}"


def _excerpt(finding: Finding, text: str) -> str:
    if finding.payload:
        return finding.payload
    span = finding.span
    if span is None or not text:
        return text or finding.evidence
    start, end = max(0, span.start - _EXCERPT_CONTEXT), min(len(text), span.end + _EXCERPT_CONTEXT)
    return f"{text[start : span.start]}«{text[span.start : span.end]}»{text[span.end : end]}"


def _hints(finding: Finding, text: str) -> dict[str, object]:
    if finding.threat not in _SECRET_THREATS or finding.span is None or not text:
        return {}
    value = text[finding.span.start : finding.span.end]
    return {"looks_like_example_or_placeholder": bool(_PLACEHOLDER.search(value))}
