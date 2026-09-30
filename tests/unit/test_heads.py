from __future__ import annotations

import random

import pytest

from immune.config.spec import FeatureSpec, HeadSpec, Spec
from immune.heads import (
    CalibrationReport,
    Head,
    HeadHealth,
    HeadRegistry,
    IsotonicCalibrator,
    PlattCalibrator,
    PromotionPolicy,
    ThreatStats,
    WilsonBound,
    expected_calibration_error,
)
from immune.sensing.signals import SensorReading, Signal


def reading(**probabilities: float) -> SensorReading:
    return SensorReading({key: Signal(key, "noul", value) for key, value in probabilities.items()}, source="test")


class TestHead:
    def test_single_logit_feature_is_identity(self) -> None:
        head = Head("t", HeadSpec(features=(FeatureSpec(signal="x"),)))
        score = head.score(reading(x=0.83), set())
        assert score is not None
        assert score.probability == pytest.approx(0.83, abs=1e-6)

    def test_invert_transform(self) -> None:
        head = Head("t", HeadSpec(features=(FeatureSpec(signal="fidelity", transform="invert"),)))
        score = head.score(reading(fidelity=0.1), set())
        assert score is not None
        assert score.probability == pytest.approx(0.9, abs=1e-6)

    def test_max_view_takes_riskier_view_and_reports_agreement(self) -> None:
        head = Head("t", HeadSpec(features=(FeatureSpec(signal="x", reduce="max_view"),)))
        score = head.score(reading(x=0.1, x__defanged=0.95), set())
        assert score is not None
        assert score.probability == pytest.approx(0.95, abs=1e-6)
        assert score.views_agree is False

    def test_choice_options_are_summed(self) -> None:
        head = Head("t", HeadSpec(features=(FeatureSpec(signal="harm", options=("a", "b")),)))
        signal = Signal("harm", "choice", 0.5, {"none": 0.2, "a": 0.5, "b": 0.3}, "a")
        score = head.score(SensorReading({"harm": signal}, "test"), set())
        assert score is not None
        assert score.probability == pytest.approx(0.8, abs=1e-6)

    def test_findings_raise_the_score(self) -> None:
        spec = HeadSpec(features=(FeatureSpec(signal="x"), FeatureSpec(finding="tokens", weight=2.0)))
        head = Head("t", spec)
        plain, boosted = head.score(reading(x=0.3), set()), head.score(reading(x=0.3), {"tokens"})
        assert plain is not None
        assert boosted is not None
        assert boosted.probability > plain.probability

    def test_positive_transform_never_lowers_the_score(self) -> None:
        spec = HeadSpec(features=(FeatureSpec(signal="x"), FeatureSpec(signal="y", transform="positive", weight=0.5)))
        head = Head("t", spec)
        score = head.score(reading(x=0.9, y=0.01), set())
        assert score is not None
        assert score.probability == pytest.approx(0.9, abs=1e-6)

    def test_missing_signals_give_no_score(self) -> None:
        assert Head("t", HeadSpec(features=(FeatureSpec(signal="x"),))).score(reading(), set()) is None

    def test_registry_covers_every_jev_threat(self, spec: Spec) -> None:
        registry = HeadRegistry.from_spec(spec)
        assert all(registry.get(threat.id) for threat in spec.threats.values() if threat.detector == "jev")


class TestCalibration:
    @staticmethod
    def overconfident(samples: int = 2000) -> tuple[list[float], list[bool]]:
        generator = random.Random(3)
        probabilities, labels = [], []
        for _ in range(samples):
            truth = generator.random()
            probabilities.append(min(0.999, truth**0.4))
            labels.append(generator.random() < truth)
        return probabilities, labels

    def test_platt_reduces_calibration_error(self) -> None:
        probabilities, labels = self.overconfident()
        calibrator = PlattCalibrator.fit(probabilities, labels)
        before = expected_calibration_error(probabilities, labels)
        after = expected_calibration_error([calibrator.apply(p) for p in probabilities], labels)
        assert after < before / 2

    def test_isotonic_is_monotone(self) -> None:
        probabilities, labels = self.overconfident()
        calibrator = IsotonicCalibrator.fit(probabilities, labels)
        outputs = [calibrator.apply(step / 100) for step in range(101)]
        assert outputs == sorted(outputs)

    def test_report_and_health(self) -> None:
        probabilities, labels = self.overconfident(500)
        calibrator = PlattCalibrator.fit(probabilities, labels)
        report = CalibrationReport.measure([calibrator.apply(p) for p in probabilities], labels, rounds=50)
        assert report.lower <= report.ece <= report.upper
        assert HeadHealth.of(None) is HeadHealth.PROVISIONAL
        assert HeadHealth.of(report, model_changed=True) is HeadHealth.DEGRADED


class TestPromotion:
    def test_wilson_upper_bound(self) -> None:
        assert WilsonBound.upper(0, 5000) < 0.001
        assert WilsonBound.upper(20, 5000) > 0.001

    def test_policy_requires_volume_time_and_low_rate(self) -> None:
        policy = PromotionPolicy()
        day = 86_400.0
        quiet = ThreatStats(screened=6000, fired=0, first_seen=0.0)
        assert policy.decide("x", quiet, now=15 * day).promoted
        assert not policy.decide("x", quiet, now=3 * day).promoted
        assert not policy.decide("x", ThreatStats(screened=100, fired=0, first_seen=0.0), now=15 * day).promoted
        assert not policy.decide("x", ThreatStats(screened=6000, fired=30, first_seen=0.0), now=15 * day).promoted


def test_evidence_reports_the_answer_jev_gave_for_inverted_questions() -> None:
    spec = Spec.default()
    head = HeadRegistry.from_spec(spec).get("output.task_deviation")
    assert head is not None
    reading = SensorReading(
        signals={"task_fidelity": Signal(key="task_fidelity", kind="noul", probability=0.05)}, source="test"
    )
    score = head.score(reading, set())
    assert score is not None
    assert score.probability > 0.9
    assert score.evidence == ("task_fidelity=0.05",)
