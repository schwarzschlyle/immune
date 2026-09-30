from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from immune.config.spec import Spec
from immune.evidence.collect import SignalRow
from immune.evidence.metrics import Metrics, Scored, finite
from immune.sensing.priority import RequestPrioritizer

MISCALIBRATED = 0.15


@dataclass(frozen=True, slots=True)
class QuestionStudy:
    key: str
    threats: tuple[str, ...]
    samples: int
    positives: int
    roc_auc: float | None
    ece: float | None

    @property
    def miscalibrated(self) -> bool:
        return self.ece is not None and self.ece > MISCALIBRATED


class CalibrationStudy:
    def __init__(self, spec: Spec) -> None:
        self._threats: dict[str, set[str]] = defaultdict(set)
        for threat, head in spec.heads.items():
            for feature in head.features:
                if feature.signal:
                    self._threats[feature.signal].add(threat)

    def run(self, rows: Sequence[SignalRow]) -> list[QuestionStudy]:
        grouped: dict[str, list[Scored]] = defaultdict(list)
        for row in rows:
            key = RequestPrioritizer.base_key(row.key)
            labels = [row.labels[threat] for threat in self._threats.get(key, ()) if threat in row.labels]
            if labels:
                grouped[key].append(Scored(row.probability, any(labels)))
        return [
            QuestionStudy(
                key=key,
                threats=tuple(sorted(self._threats[key])),
                samples=len(items),
                positives=sum(item.positive for item in items),
                roc_auc=finite(Metrics.roc_auc(items)),
                ece=finite(Metrics.ece(items)),
            )
            for key, items in sorted(grouped.items())
        ]
