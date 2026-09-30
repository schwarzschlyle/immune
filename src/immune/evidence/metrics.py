from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from immune.heads.calibration import expected_calibration_error
from immune.heads.promotion import WilsonBound


@dataclass(frozen=True, slots=True)
class Scored:
    probability: float
    positive: bool


class Metrics:
    @staticmethod
    def roc_auc(items: Sequence[Scored]) -> float:
        positives = [item.probability for item in items if item.positive]
        negatives = [item.probability for item in items if not item.positive]
        if not positives or not negatives:
            return float("nan")
        ranked = sorted((probability, index) for index, probability in enumerate(positives + negatives))
        ranks = [0.0] * len(ranked)
        position = 0
        while position < len(ranked):
            end = position
            while end + 1 < len(ranked) and ranked[end + 1][0] == ranked[position][0]:
                end += 1
            average = (position + end) / 2 + 1
            for cursor in range(position, end + 1):
                ranks[ranked[cursor][1]] = average
            position = end + 1
        positive_ranks = sum(ranks[: len(positives)])
        return (positive_ranks - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives))

    @staticmethod
    def average_precision(items: Sequence[Scored]) -> float:
        ordered = sorted(items, key=lambda item: item.probability, reverse=True)
        total = sum(item.positive for item in ordered)
        if total == 0:
            return float("nan")
        hits, precision_sum = 0, 0.0
        for rank, item in enumerate(ordered, start=1):
            if item.positive:
                hits += 1
                precision_sum += hits / rank
        return precision_sum / total

    @staticmethod
    def brier(items: Sequence[Scored]) -> float:
        return sum((item.probability - item.positive) ** 2 for item in items) / len(items) if items else float("nan")

    @staticmethod
    def ece(items: Sequence[Scored]) -> float:
        return expected_calibration_error([item.probability for item in items], [item.positive for item in items])

    @staticmethod
    def at_threshold(items: Sequence[Scored], threshold: float) -> tuple[float, float, float]:
        predicted = [item for item in items if item.probability >= threshold]
        true_positive = sum(item.positive for item in predicted)
        positives = sum(item.positive for item in items)
        precision = true_positive / len(predicted) if predicted else 1.0
        recall = true_positive / positives if positives else 1.0
        lower = WilsonLower.of(true_positive, len(predicted)) if predicted else 0.0
        return precision, recall, lower


class WilsonLower:
    @staticmethod
    def of(successes: int, trials: int, z: float = 1.96) -> float:
        if trials <= 0:
            return 0.0
        return max(0.0, 1.0 - WilsonBound.upper(trials - successes, trials, z))


def finite(value: float) -> float | None:
    return None if math.isnan(value) else round(value, 4)
