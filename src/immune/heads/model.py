from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from immune.config.spec import FeatureSpec, HeadSpec, Spec
from immune.heads.calibration import Calibrator, IdentityCalibrator, logit, sigmoid
from immune.sensing.signals import DEFANGED_SUFFIX, SensorReading, Signal


@dataclass(frozen=True, slots=True)
class HeadScore:
    threat: str
    probability: float
    raw_probability: float
    evidence: tuple[str, ...]
    views_agree: bool | None = None


@dataclass(frozen=True, slots=True)
class _FeatureValue:
    value: float
    evidence: str | None
    agreement: bool | None
    observed: bool


class Head:
    def __init__(self, threat: str, spec: HeadSpec, calibrator: Calibrator | None = None) -> None:
        self.threat = threat
        self.spec = spec
        self.calibrator = calibrator or IdentityCalibrator()

    def vector(self, reading: SensorReading, findings: set[str]) -> list[float] | None:
        computed = [self._feature(feature, reading, findings) for feature in self.spec.features]
        if not any(item.observed for item in computed):
            return None
        return [item.value for item in computed]

    def score(self, reading: SensorReading, findings: set[str]) -> HeadScore | None:
        total = self.spec.bias
        evidence: list[str] = []
        agreements: list[bool] = []
        observed = False
        for feature in self.spec.features:
            computed = self._feature(feature, reading, findings)
            observed = observed or computed.observed
            total += feature.weight * computed.value
            if computed.evidence:
                evidence.append(computed.evidence)
            if computed.agreement is not None:
                agreements.append(computed.agreement)
        if not observed:
            return None
        raw = sigmoid(total)
        return HeadScore(
            threat=self.threat,
            probability=self.calibrator.apply(raw),
            raw_probability=raw,
            evidence=tuple(evidence),
            views_agree=all(agreements) if agreements else None,
        )

    def _feature(self, feature: FeatureSpec, reading: SensorReading, findings: set[str]) -> _FeatureValue:
        if feature.finding is not None:
            present = feature.finding in findings
            return _FeatureValue(
                1.0 if present else 0.0, f"finding:{feature.finding}" if present else None, None, False
            )
        assert feature.signal is not None
        raw = reading.get(feature.signal)
        defanged = reading.get(f"{feature.signal}{DEFANGED_SUFFIX}")
        if raw is None and defanged is None:
            return _FeatureValue(0.0, None, None, False)
        raw_p = self._probability(raw or defanged, feature)
        defanged_p = self._probability(defanged, feature) if defanged is not None else None
        agreement = None if defanged_p is None else (raw_p >= 0.5) == (defanged_p >= 0.5)
        if feature.reduce == "divergence":
            divergence = abs(logit(raw_p) - logit(defanged_p)) if defanged_p is not None else 0.0
            return _FeatureValue(
                divergence, f"{feature.signal}:divergence={divergence:.2f}" if divergence else None, None, True
            )
        probability = max(raw_p, defanged_p) if feature.reduce == "max_view" and defanged_p is not None else raw_p
        value = logit(probability)
        if feature.transform == "positive":
            value = max(0.0, value)
        answered = 1.0 - probability if feature.transform == "invert" else probability
        return _FeatureValue(value, f"{feature.signal}={answered:.2f}", agreement, True)

    @staticmethod
    def _probability(signal: Signal | None, feature: FeatureSpec) -> float:
        if signal is None:
            return 0.0
        probability = signal.probability_of(feature.options)
        return 1.0 - probability if feature.transform == "invert" else probability


class HeadRegistry:
    def __init__(self, heads: Mapping[str, Head], sites: Mapping[str, Mapping[str, Calibrator]] | None = None) -> None:
        self._heads = dict(heads)
        self._sites = {site: dict(calibrators) for site, calibrators in (sites or {}).items()}

    @classmethod
    def from_spec(
        cls,
        spec: Spec,
        calibrators: Mapping[str, Calibrator] | None = None,
        sites: Mapping[str, Mapping[str, Calibrator]] | None = None,
    ) -> HeadRegistry:
        overrides = calibrators or {}
        return cls({threat: Head(threat, head, overrides.get(threat)) for threat, head in spec.heads.items()}, sites)

    def get(self, threat: str) -> Head | None:
        return self._heads.get(threat)

    def threats(self) -> list[str]:
        return list(self._heads)

    def score(
        self, threats: list[str], reading: SensorReading, findings: set[str], site: str | None = None
    ) -> dict[str, HeadScore]:
        local = self._sites.get(site or "", {})
        scores: dict[str, HeadScore] = {}
        for threat in threats:
            head = self._heads.get(threat)
            result = head.score(reading, findings) if head else None
            if result is None:
                continue
            calibrator = local.get(threat)
            if calibrator is not None:
                result = HeadScore(
                    threat=result.threat,
                    probability=calibrator.apply(result.probability),
                    raw_probability=result.raw_probability,
                    evidence=result.evidence,
                    views_agree=result.views_agree,
                )
            scores[threat] = result
        return scores

    def with_calibrators(self, calibrators: Mapping[str, Calibrator]) -> HeadRegistry:
        return HeadRegistry(
            {
                threat: Head(threat, head.spec, calibrators.get(threat, head.calibrator))
                for threat, head in self._heads.items()
            },
            self._sites,
        )
