from __future__ import annotations

import datetime
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from immune.config.spec import Spec, ThreatSpec
from immune.evidence.collect import FeatureRow
from immune.evidence.metrics import Metrics, Scored, finite
from immune.heads.artifact import HeadsArtifact, TrainedHead
from immune.heads.calibration import Calibrator, IdentityCalibrator, IsotonicCalibrator, PlattCalibrator, sigmoid
from immune.heads.optimize import RegularizedLogistic

FLOOR_PRECISION = 0.99
CRISIS_RECALL = 0.95
CRISIS_THREATS = frozenset({"input.crisis"})
_ISOTONIC_MINIMUM = 1000
_MINIMUM_ROWS = 20


@dataclass(frozen=True, slots=True)
class LogisticModel:
    weights: tuple[float, ...]
    bias: float

    def probability(self, vector: Sequence[float]) -> float:
        return sigmoid(self.bias + sum(weight * value for weight, value in zip(self.weights, vector, strict=True)))

    @classmethod
    def fit(
        cls, rows: Sequence[FeatureRow], prior: Sequence[float], prior_bias: float, strength: float = 1.0
    ) -> LogisticModel:
        solver = RegularizedLogistic([prior_bias, *prior], strength)
        theta = solver.fit([(1.0, *row.vector) for row in rows], [row.label for row in rows])
        return cls(tuple(theta[1:]), theta[0])


@dataclass(slots=True)
class HeadReport:
    threat: str
    rows: dict[str, int] = field(default_factory=dict)
    positives: dict[str, int] = field(default_factory=dict)
    metrics: dict[str, float | None] = field(default_factory=dict)
    threshold: float | None = None
    target: str = ""
    target_met: bool = True
    skipped: str = ""


class HeadTrainer:
    def __init__(self, spec: Spec, jev_model: str) -> None:
        self._spec = spec
        self._jev_model = jev_model

    def train(
        self, rows: Sequence[FeatureRow], data: dict[str, object] | None = None
    ) -> tuple[HeadsArtifact, list[HeadReport]]:
        grouped: dict[str, list[FeatureRow]] = defaultdict(list)
        for row in rows:
            grouped[row.threat].append(row)
        heads: dict[str, TrainedHead] = {}
        reports: list[HeadReport] = []
        for threat_id, items in sorted(grouped.items()):
            trained, report = self._head(threat_id, items)
            reports.append(report)
            if trained is not None:
                heads[threat_id] = trained
        summary = {
            "examples": len({row.example for row in rows}),
            "sources": dict(Counter(row.source for row in rows)),
            "languages": dict(Counter(row.language for row in rows)),
            **(data or {}),
        }
        artifact = HeadsArtifact(
            spec_version=self._spec.version,
            jev_model=self._jev_model,
            heads=heads,
            data=summary,
            created_at=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        return artifact, reports

    def _head(self, threat_id: str, rows: Sequence[FeatureRow]) -> tuple[TrainedHead | None, HeadReport]:
        report = HeadReport(threat_id)
        splits = {name: [row for row in rows if row.split == name] for name in ("train", "dev", "test")}
        report.rows = {name: len(items) for name, items in splits.items()}
        report.positives = {name: sum(row.label for row in items) for name, items in splits.items()}
        head = self._spec.heads.get(threat_id)
        threat = self._spec.threats.get(threat_id)
        if head is None or threat is None:
            report.skipped = "not a trainable head"
            return None, report
        train, dev, test = splits["train"], splits["dev"] or splits["train"], splits["test"] or splits["dev"]
        if len(train) < _MINIMUM_ROWS or len({row.label for row in train}) < 2:
            report.skipped = f"needs at least {_MINIMUM_ROWS} training rows with both labels"
            return None, report
        model = LogisticModel.fit(train, [feature.weight for feature in head.features], head.bias)
        calibrator = self._calibrate(model, dev)
        scored_dev = [Scored(calibrator.apply(model.probability(row.vector)), row.label) for row in dev]
        report.threshold, report.target, report.target_met = self._threshold(threat, scored_dev)
        scored_test = [Scored(calibrator.apply(model.probability(row.vector)), row.label) for row in test]
        precision, recall, lower = Metrics.at_threshold(scored_test, report.threshold)
        report.metrics = {
            "roc_auc": finite(Metrics.roc_auc(scored_test)),
            "average_precision": finite(Metrics.average_precision(scored_test)),
            "ece": finite(Metrics.ece(scored_test)),
            "brier": finite(Metrics.brier(scored_test)),
            "precision": round(precision, 4),
            "precision_lower_95": round(lower, 4),
            "recall": round(recall, 4),
        }
        metrics = {key: value for key, value in report.metrics.items() if value is not None}
        return TrainedHead(model.weights, model.bias, calibrator, report.threshold, metrics), report

    @staticmethod
    def _calibrate(model: LogisticModel, rows: Sequence[FeatureRow]) -> Calibrator:
        if len({row.label for row in rows}) < 2:
            return IdentityCalibrator()
        probabilities = [model.probability(row.vector) for row in rows]
        labels = [row.label for row in rows]
        if len(rows) >= _ISOTONIC_MINIMUM:
            return IsotonicCalibrator.fit(probabilities, labels)
        return PlattCalibrator.fit(probabilities, labels)

    @staticmethod
    def _threshold(threat: ThreatSpec, scored: Sequence[Scored]) -> tuple[float, str, bool]:
        candidates = sorted({round(item.probability, 4) for item in scored} | {threat.threshold})
        if threat.floor is not None and threat.id not in CRISIS_THREATS:
            target = f"precision lower bound >= {FLOOR_PRECISION:.0%}"
            for candidate in candidates:
                _, _, lower = Metrics.at_threshold(scored, candidate)
                if lower >= FLOOR_PRECISION:
                    return candidate, target, True
            return max(threat.threshold, candidates[-1]), target, False
        if threat.id in CRISIS_THREATS:
            target = f"recall >= {CRISIS_RECALL:.0%}"
            for candidate in reversed(candidates):
                _, recall, _ = Metrics.at_threshold(scored, candidate)
                if recall >= CRISIS_RECALL:
                    return candidate, target, True
            return candidates[0], target, False
        return threat.threshold, "specification threshold", not math.isnan(Metrics.roc_auc(scored))
