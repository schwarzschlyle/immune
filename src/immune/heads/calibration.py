from __future__ import annotations

import bisect
import math
import random
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from immune.heads.optimize import RegularizedLogistic

_EPSILON = 1e-4
_LOGIT_LIMIT = 6.0


def clamp(probability: float) -> float:
    return min(max(probability, _EPSILON), 1 - _EPSILON)


def logit(probability: float) -> float:
    value = clamp(probability)
    return max(-_LOGIT_LIMIT, min(_LOGIT_LIMIT, math.log(value / (1 - value))))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1 + exponent)


class Calibrator(ABC):
    @abstractmethod
    def apply(self, probability: float) -> float: ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Calibrator:
        kind = data.get("kind")
        if kind == "platt":
            return PlattCalibrator(float(data["slope"]), float(data["intercept"]))
        if kind == "isotonic":
            return IsotonicCalibrator(tuple(data["thresholds"]), tuple(data["values"]))
        return IdentityCalibrator()


class IdentityCalibrator(Calibrator):
    def apply(self, probability: float) -> float:
        return probability

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "identity"}


class PlattCalibrator(Calibrator):
    def __init__(self, slope: float = 1.0, intercept: float = 0.0) -> None:
        self.slope = slope
        self.intercept = intercept

    def apply(self, probability: float) -> float:
        return sigmoid(self.slope * logit(probability) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "platt", "slope": self.slope, "intercept": self.intercept}

    @classmethod
    def fit(
        cls, probabilities: Sequence[float], labels: Sequence[bool], iterations: int = 100, prior_strength: float = 1.0
    ) -> PlattCalibrator:
        rows = [(1.0, logit(probability)) for probability in probabilities]
        intercept, slope = RegularizedLogistic([0.0, 1.0], prior_strength, iterations).fit(rows, labels)
        return cls(slope, intercept)


class IsotonicCalibrator(Calibrator):
    def __init__(self, thresholds: Sequence[float], values: Sequence[float]) -> None:
        self.thresholds = tuple(thresholds)
        self.values = tuple(values)

    def apply(self, probability: float) -> float:
        if not self.thresholds:
            return probability
        index = bisect.bisect_right(self.thresholds, probability) - 1
        return self.values[max(index, 0)]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "isotonic", "thresholds": list(self.thresholds), "values": list(self.values)}

    @classmethod
    def fit(cls, probabilities: Sequence[float], labels: Sequence[bool]) -> IsotonicCalibrator:
        pairs = sorted(zip(probabilities, labels, strict=True), key=lambda item: item[0])
        blocks: list[list[float]] = []
        for probability, label in pairs:
            blocks.append([probability, 1.0 if label else 0.0, 1.0])
            while len(blocks) > 1 and blocks[-2][1] / blocks[-2][2] > blocks[-1][1] / blocks[-1][2]:
                last = blocks.pop()
                blocks[-1][1] += last[1]
                blocks[-1][2] += last[2]
        return cls([block[0] for block in blocks], [block[1] / block[2] for block in blocks])


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    ece: float
    lower: float
    upper: float
    samples: int

    @classmethod
    def measure(
        cls, probabilities: Sequence[float], labels: Sequence[bool], rounds: int = 200, seed: int = 7
    ) -> CalibrationReport:
        ece = expected_calibration_error(probabilities, labels)
        generator = random.Random(seed)
        indices = range(len(probabilities))
        samples = []
        for _ in range(rounds if probabilities else 0):
            chosen = [generator.choice(indices) for _ in indices]
            samples.append(expected_calibration_error([probabilities[i] for i in chosen], [labels[i] for i in chosen]))
        samples.sort()
        lower = samples[int(0.025 * len(samples))] if samples else ece
        upper = samples[min(int(0.975 * len(samples)), len(samples) - 1)] if samples else ece
        return cls(ece=ece, lower=lower, upper=upper, samples=len(probabilities))


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[bool], bins: int = 10) -> float:
    if not probabilities:
        return 0.0
    totals = [[0.0, 0.0, 0] for _ in range(bins)]
    for probability, label in zip(probabilities, labels, strict=True):
        bucket = totals[min(int(probability * bins), bins - 1)]
        bucket[0] += probability
        bucket[1] += 1.0 if label else 0.0
        bucket[2] += 1
    count = len(probabilities)
    return sum(abs(confidence - hits) / count for confidence, hits, size in totals if size)


class HeadHealth(StrEnum):
    CALIBRATED = "calibrated"
    PROVISIONAL = "provisional"
    DEGRADED = "degraded"

    @classmethod
    def of(cls, report: CalibrationReport | None, model_changed: bool = False) -> HeadHealth:
        if report is None:
            return cls.PROVISIONAL
        if model_changed:
            return cls.DEGRADED
        return cls.CALIBRATED if report.ece <= 0.05 and report.upper <= 0.07 else cls.DEGRADED
