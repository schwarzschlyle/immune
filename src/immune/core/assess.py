from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from immune.config.spec import Spec, ThreatSpec
from immune.core.conversation import Span
from immune.heads.model import HeadRegistry, HeadScore
from immune.reflexes.findings import Finding
from immune.sensing.signals import SensorReading
from immune.types import Stage

if TYPE_CHECKING:
    from immune.core.candidates import Candidate

_TOLERANCE = 1e-9
_UNDECIDED = "acted without Jev: the sensor was unavailable (sensor.on_outage: act_on_candidates)"


def reaches(probability: float, threshold: float) -> bool:
    return probability >= threshold - _TOLERANCE


@dataclass(frozen=True, slots=True)
class Assessment:
    threat: ThreatSpec
    probability: float
    evidence: tuple[str, ...]
    subject: str | None = None
    span: Span | None = None
    views_agree: bool | None = None
    deterministic: bool = False

    @property
    def triggered(self) -> bool:
        return reaches(self.probability, self.threat.threshold)


class Assessor:
    def __init__(self, spec: Spec, heads: HeadRegistry) -> None:
        self._spec = spec
        self._heads = heads

    @property
    def heads(self) -> HeadRegistry:
        return self._heads

    @property
    def spec(self) -> Spec:
        return self._spec

    def with_heads(self, heads: HeadRegistry) -> Assessor:
        return Assessor(self._spec, heads)

    def deterministic(self, findings: Iterable[Finding], subject: str | None = None) -> list[Assessment]:
        assessments: list[Assessment] = []
        for finding in findings:
            threat = self._spec.threats.get(finding.threat)
            if threat is None:
                continue
            assessments.append(
                Assessment(
                    threat=threat,
                    probability=1.0,
                    evidence=(finding.evidence,),
                    subject=finding.subject or subject,
                    span=finding.span,
                    deterministic=True,
                )
            )
        return assessments

    def judged(
        self,
        stage: Stage,
        reading: SensorReading,
        findings: Iterable[Finding],
        organs: frozenset[str],
        subject: str | None = None,
        site: str | None = None,
        off: frozenset[str] = frozenset(),
    ) -> list[Assessment]:
        if reading.is_empty:
            return []
        finding_ids = {finding.threat for finding in findings}
        scores = self._heads.score(self.judged_threats(stage, organs, off), reading, finding_ids, site)
        return [self._from_score(score, subject) for score in scores.values()]

    def candidates(
        self,
        candidates: Iterable[Candidate],
        reading: SensorReading | None,
        unavailable: bool,
        act_on_outage: bool,
        site: str | None = None,
    ) -> list[Assessment]:
        """Jev's decision on each candidate. Without Jev, floor candidates act only if the outage policy says so."""
        assessments: list[Assessment] = []
        for candidate in candidates:
            threat = self._spec.threats.get(candidate.threat)
            if threat is None:
                continue
            score = None
            if reading is not None and not reading.is_empty:
                score = self._heads.score([threat.id], reading.scoped(candidate.id), set(), site).get(threat.id)
            if score is not None:
                assessments.append(
                    Assessment(
                        threat=threat,
                        probability=score.probability,
                        evidence=(candidate.finding.evidence, *score.evidence),
                        subject=candidate.subject,
                        span=candidate.finding.span,
                        views_agree=score.views_agree,
                    )
                )
            elif unavailable and act_on_outage and threat.floor is not None:
                assessments.append(
                    Assessment(
                        threat=threat,
                        probability=1.0,
                        evidence=(candidate.finding.evidence, _UNDECIDED),
                        subject=candidate.subject,
                        span=candidate.finding.span,
                        deterministic=True,
                    )
                )
        return assessments

    def judged_threats(self, stage: Stage, organs: frozenset[str], off: frozenset[str] = frozenset()) -> list[str]:
        return [
            threat.id
            for threat in self._spec.threats_for(stage)
            if threat.detector == "jev" and (threat.organ is None or threat.organ in organs) and threat.id not in off
        ]

    def evaluated_threats(self, stage: Stage, organs: frozenset[str], off: frozenset[str] = frozenset()) -> list[str]:
        """Threats checked at this stage. Vaccines in `off` weren't checked, so they build no promotion record."""
        return [
            threat.id
            for threat in self._spec.threats_for(stage)
            if (threat.organ is None or threat.organ in organs) and threat.id not in off
        ]

    def _from_score(self, score: HeadScore, subject: str | None) -> Assessment:
        return Assessment(
            threat=self._spec.threat(score.threat),
            probability=score.probability,
            evidence=score.evidence,
            subject=subject,
            views_agree=score.views_agree,
        )

    def session(self, probability: float) -> Assessment:
        threat = self._spec.threat("session.adversarial")
        return Assessment(threat=threat, probability=probability, evidence=(f"session_risk={probability:.2f}",))


def strongest(assessments: Iterable[Assessment], threat_ids: Iterable[str]) -> float:
    wanted = set(threat_ids)
    return max((item.probability for item in assessments if item.threat.id in wanted), default=0.0)


def by_subject(assessments: Iterable[Assessment]) -> Mapping[str | None, list[Assessment]]:
    grouped: dict[str | None, list[Assessment]] = {}
    for assessment in assessments:
        grouped.setdefault(assessment.subject, []).append(assessment)
    return grouped
