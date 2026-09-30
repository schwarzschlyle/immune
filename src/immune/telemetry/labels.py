from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from immune.errors import ConfigError
from immune.heads.calibration import CalibrationReport, Calibrator, IsotonicCalibrator, PlattCalibrator
from immune.telemetry.state import StateStore
from immune.types import Verdict

Label = Literal["false_positive", "missed", "correct"]
_LABELS: tuple[str, ...] = ("false_positive", "missed", "correct")
_ISOTONIC_MINIMUM = 1000


@dataclass(frozen=True, slots=True)
class LabeledScore:
    threat: str
    probability: float
    positive: bool
    site: str = ""


class LabelStore:
    def __init__(self, store: StateStore) -> None:
        self._store = store

    def add(self, verdict: Verdict, label: str, note: str | None = None, threat: str | None = None) -> None:
        if label not in _LABELS:
            raise ConfigError(f"label must be one of {', '.join(_LABELS)}")
        hits = [hit for hit in verdict.hits if threat is None or hit.threat == threat]
        self._store.append_line(
            "labels",
            {
                "trace_id": verdict.trace_id,
                "site": verdict.site,
                "label": label,
                "note": note,
                "at": time.time(),
                "scores": [{"threat": hit.threat, "probability": hit.probability} for hit in hits],
                "threat": threat,
            },
        )

    def samples(self) -> list[LabeledScore]:
        return [
            LabeledScore(
                score["threat"],
                float(score["probability"]),
                record.get("label") in ("correct", "missed"),
                str(record.get("site", "")),
            )
            for record in self._store.read_lines("labels")
            for score in record.get("scores", [])
        ]


class CalibrationFitter:
    def __init__(self, minimum_samples: int = 50, site_minimum: int = 200) -> None:
        self._minimum = minimum_samples
        self._site_minimum = site_minimum

    def fit(self, samples: list[LabeledScore]) -> dict[str, tuple[Calibrator, CalibrationReport]]:
        return self._fit(samples, self._minimum)

    def fit_by_site(self, samples: list[LabeledScore]) -> dict[str, dict[str, tuple[Calibrator, CalibrationReport]]]:
        grouped: dict[str, list[LabeledScore]] = defaultdict(list)
        for sample in samples:
            if sample.site:
                grouped[sample.site].append(sample)
        fitted = {site: self._fit(items, self._site_minimum) for site, items in grouped.items()}
        return {site: calibrators for site, calibrators in fitted.items() if calibrators}

    @staticmethod
    def _fit(samples: list[LabeledScore], minimum: int) -> dict[str, tuple[Calibrator, CalibrationReport]]:
        grouped: dict[str, list[LabeledScore]] = defaultdict(list)
        for sample in samples:
            grouped[sample.threat].append(sample)
        fitted: dict[str, tuple[Calibrator, CalibrationReport]] = {}
        for threat, items in grouped.items():
            if len(items) < minimum:
                continue
            probabilities = [item.probability for item in items]
            labels = [item.positive for item in items]
            calibrator: Calibrator = (
                IsotonicCalibrator.fit(probabilities, labels)
                if len(items) >= _ISOTONIC_MINIMUM
                else PlattCalibrator.fit(probabilities, labels)
            )
            calibrated = [calibrator.apply(probability) for probability in probabilities]
            fitted[threat] = (calibrator, CalibrationReport.measure(calibrated, labels))
        return fitted
