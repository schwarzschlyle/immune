from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from dataclasses import dataclass

from immune.config.settings import SiteSettings
from immune.config.spec import FloorSpec, ThreatSpec
from immune.core.assess import Assessment, reaches
from immune.profiling.profile import SiteProfile
from immune.telemetry.stats import PromotionLedger
from immune.types import Action, Hit, Mode, Sink
from immune.vaccines.switchboard import Switchboard

_OBSERVE_MODE_FLOOR = frozenset({"F9"})


@dataclass(frozen=True, slots=True)
class Decision:
    assessment: Assessment
    hit: Hit

    @property
    def enforced_action(self) -> Action:
        return self.hit.action if self.hit.enforced else Action.ALLOW


class EnforcementPolicy:
    def __init__(
        self,
        mode: Mode,
        promotions: PromotionLedger,
        switchboard: Switchboard | None = None,
        always: frozenset[str] = frozenset(),
        unpromoted: frozenset[str] = frozenset(),
    ) -> None:
        self._mode = mode
        self._promotions = promotions
        self._switchboard = switchboard
        self._always = always
        self._unpromoted = unpromoted

    def disabled(self, threat_id: str, settings: SiteSettings | None) -> bool:
        return self._switchboard is not None and self._switchboard.disabled(threat_id, settings)

    def enforced(self, assessment: Assessment, profile: SiteProfile, site: str, settings: SiteSettings | None) -> bool:
        threat_id = assessment.threat.id
        if settings is not None and self._matches(threat_id, settings.observe):
            return False
        if settings is not None and self._matches(threat_id, settings.enforce):
            return True
        if threat_id in self._always and self._mode in (Mode.AUTO, Mode.STRICT):
            return True
        if threat_id == "output.echo_disagreement" and settings is not None and settings.echo.enforce:
            return True
        floor = assessment.threat.floor
        floor_holds = floor is not None and self._floor_holds(floor, assessment, profile)
        if self._mode is Mode.STRICT:
            return True
        if self._mode is Mode.OBSERVE:
            return floor_holds and floor is not None and floor.id in _OBSERVE_MODE_FLOOR
        return floor_holds or self._promoted(site, threat_id)

    def could_enforce(self, threat: ThreatSpec, site: str, settings: SiteSettings | None) -> bool:
        if settings is not None and self._matches(threat.id, settings.observe):
            return False
        if settings is not None and self._matches(threat.id, settings.enforce):
            return True
        if threat.id in self._always and self._mode is Mode.AUTO:
            return True
        if self._mode is Mode.STRICT:
            return True
        if self._mode is Mode.OBSERVE:
            return threat.floor is not None and threat.floor.id in _OBSERVE_MODE_FLOOR
        return threat.floor is not None or self._promoted(site, threat.id)

    def _promoted(self, site: str, threat_id: str) -> bool:
        # Experimental library vaccines are enforced only by `sites.<site>.enforce`, never by promotion.
        return threat_id not in self._unpromoted and self._promotions.promoted(site, threat_id)

    @staticmethod
    def _floor_holds(floor: FloorSpec, assessment: Assessment, profile: SiteProfile) -> bool:
        if not reaches(assessment.probability, floor.threshold):
            return False
        if floor.requires == "user_facing":
            return profile.user_facing
        if floor.requires == "rendered":
            return profile.rendered
        if floor.requires == "view_agreement":
            return assessment.views_agree is not False
        return True

    @staticmethod
    def _matches(threat_id: str, patterns: Iterable[str]) -> bool:
        return any(fnmatch.fnmatchcase(threat_id, pattern) for pattern in patterns)


class Decider:
    def __init__(self, policy: EnforcementPolicy) -> None:
        self._policy = policy

    @property
    def policy(self) -> EnforcementPolicy:
        return self._policy

    def decide(
        self,
        assessments: Iterable[Assessment],
        sink: Sink,
        profile: SiteProfile,
        site: str,
        settings: SiteSettings | None,
    ) -> list[Decision]:
        decisions: list[Decision] = []
        for assessment in assessments:
            if not assessment.triggered or self._policy.disabled(assessment.threat.id, settings):
                continue
            threat = assessment.threat
            action = threat.action_for(sink)
            hit = Hit(
                threat=threat.id,
                invariant=threat.invariant,
                stage=threat.stage,
                probability=round(assessment.probability, 4),
                action=action,
                enforced=self._policy.enforced(assessment, profile, site, settings),
                evidence=assessment.evidence,
                frameworks=threat.frameworks,
                organ=threat.organ,
                floor=threat.floor.id if threat.floor else None,
            )
            decisions.append(Decision(assessment=assessment, hit=hit))
        return decisions


def applied(decisions: Iterable[Decision]) -> Action:
    return Action.most_severe([decision.enforced_action for decision in decisions])


def would(decisions: Iterable[Decision]) -> Action:
    return Action.most_severe([decision.hit.action for decision in decisions])
