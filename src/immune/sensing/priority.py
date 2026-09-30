from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from immune.config.settings import SiteSettings
from immune.config.spec import QuestionSpec, Spec, ThreatSpec
from immune.core.decide import EnforcementPolicy
from immune.sensing.quota import RequestPriority
from immune.sensing.signals import DEFANGED_SUFFIX

_SCOPED = re.compile(r"^(?:item|call|cand)_\d+__")
_PROFILE_PANEL = "operator"


class RequestPrioritizer:
    def __init__(self, spec: Spec, policy: EnforcementPolicy) -> None:
        self._policy = policy
        self._threats: defaultdict[str, list[ThreatSpec]] = defaultdict(list)
        for threat_id, head in spec.heads.items():
            threat = spec.threats.get(threat_id)
            if threat is None:
                continue
            for feature in head.features:
                if feature.signal:
                    self._threats[feature.signal].append(threat)
        self._profile_keys = frozenset(question.key for question in spec.panel(_PROFILE_PANEL).questions)

    def priority(self, questions: Iterable[QuestionSpec], site: str, settings: SiteSettings | None) -> RequestPriority:
        highest = RequestPriority.OBSERVED
        for question in questions:
            key = self.base_key(question.key)
            if key in self._profile_keys:
                highest = max(highest, RequestPriority.ENFORCED)
            for threat in self._threats.get(key, ()):
                if threat.floor is not None and self._policy.could_enforce(threat, site, settings):
                    return RequestPriority.FLOOR
                if self._policy.could_enforce(threat, site, settings):
                    highest = max(highest, RequestPriority.ENFORCED)
        return highest

    @staticmethod
    def base_key(key: str) -> str:
        return _SCOPED.sub("", key).removesuffix(DEFANGED_SUFFIX)
