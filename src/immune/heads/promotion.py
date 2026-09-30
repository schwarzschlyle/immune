from __future__ import annotations

import math
from dataclasses import dataclass, field

_SECONDS_PER_DAY = 86_400.0


@dataclass(slots=True)
class ThreatStats:
    screened: int = 0
    fired: int = 0
    first_seen: float | None = None
    last_seen: float | None = None

    def observe(self, fired: bool, now: float) -> None:
        if self.first_seen is None:
            self.first_seen = now
        self.last_seen = now
        self.screened += 1
        self.fired += int(fired)

    @property
    def rate(self) -> float:
        return self.fired / self.screened if self.screened else 0.0


class WilsonBound:
    @staticmethod
    def upper(successes: int, trials: int, z: float = 1.96) -> float:
        if trials <= 0:
            return 1.0
        proportion = successes / trials
        denominator = 1 + z * z / trials
        centre = proportion + z * z / (2 * trials)
        margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials))
        return min(1.0, (centre + margin) / denominator)


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    promoted: bool
    reason: str
    upper_bound: float


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    max_firing_rate: float = 0.001
    min_calls: int = 5_000
    min_days: float = 14.0
    excluded: frozenset[str] = field(default_factory=frozenset)

    def decide(self, threat: str, stats: ThreatStats, now: float) -> PromotionDecision:
        bound = WilsonBound.upper(stats.fired, stats.screened)
        if threat in self.excluded:
            return PromotionDecision(False, "excluded by configuration", bound)
        if stats.screened < self.min_calls:
            return PromotionDecision(False, f"{stats.screened}/{self.min_calls} screened calls", bound)
        days = (now - stats.first_seen) / _SECONDS_PER_DAY if stats.first_seen is not None else 0.0
        if days < self.min_days:
            return PromotionDecision(False, f"{days:.1f}/{self.min_days:.0f} days observed", bound)
        if bound > self.max_firing_rate:
            return PromotionDecision(False, f"firing-rate bound {bound:.4%} above {self.max_firing_rate:.2%}", bound)
        return PromotionDecision(True, f"firing-rate bound {bound:.4%}", bound)
