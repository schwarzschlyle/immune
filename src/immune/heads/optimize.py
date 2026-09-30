from __future__ import annotations

import math
from collections.abc import Sequence

_MIN_SCALE = 1e-4
_TOLERANCE = 1e-9


class RegularizedLogistic:
    def __init__(self, prior: Sequence[float], strength: float = 1.0, iterations: int = 50) -> None:
        self._prior = list(prior)
        self._strength = strength
        self._iterations = iterations

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[bool]) -> list[float]:
        theta = list(self._prior)
        loss = self._loss(theta, rows, labels)
        for _ in range(self._iterations):
            gradient, hessian = self._derivatives(theta, rows, labels)
            step = _solve(hessian, gradient)
            scale = 1.0
            while scale >= _MIN_SCALE:
                candidate = [value - scale * delta for value, delta in zip(theta, step, strict=True)]
                candidate_loss = self._loss(candidate, rows, labels)
                if candidate_loss <= loss:
                    break
                scale /= 2
            else:
                break
            improvement = loss - candidate_loss
            theta, loss = candidate, candidate_loss
            if improvement < _TOLERANCE:
                break
        return theta

    def _loss(self, theta: Sequence[float], rows: Sequence[Sequence[float]], labels: Sequence[bool]) -> float:
        total = (
            0.5 * self._strength * sum((value - anchor) ** 2 for value, anchor in zip(theta, self._prior, strict=True))
        )
        for row, label in zip(rows, labels, strict=True):
            margin = sum(weight * value for weight, value in zip(theta, row, strict=True))
            total += _softplus(margin) - (margin if label else 0.0)
        return total

    def _derivatives(
        self, theta: Sequence[float], rows: Sequence[Sequence[float]], labels: Sequence[bool]
    ) -> tuple[list[float], list[list[float]]]:
        size = len(theta)
        gradient = [self._strength * (value - anchor) for value, anchor in zip(theta, self._prior, strict=True)]
        hessian = [[self._strength if row == column else 0.0 for column in range(size)] for row in range(size)]
        for row, label in zip(rows, labels, strict=True):
            predicted = _sigmoid(sum(weight * value for weight, value in zip(theta, row, strict=True)))
            error = predicted - float(label)
            curvature = predicted * (1.0 - predicted)
            for first in range(size):
                gradient[first] += error * row[first]
                for second in range(size):
                    hessian[first][second] += curvature * row[first] * row[second]
        return gradient, hessian


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    augmented = [[*row, vector[index]] for index, row in enumerate(matrix)]
    for pivot in range(size):
        best = max(range(pivot, size), key=lambda row: abs(augmented[row][pivot]))
        augmented[pivot], augmented[best] = augmented[best], augmented[pivot]
        divisor = augmented[pivot][pivot] or 1e-12
        augmented[pivot] = [value / divisor for value in augmented[pivot]]
        for row in range(size):
            if row != pivot:
                factor = augmented[row][pivot]
                augmented[row] = [
                    value - factor * base for value, base in zip(augmented[row], augmented[pivot], strict=True)
                ]
    return [augmented[row][size] for row in range(size)]


def _softplus(value: float) -> float:
    return value + math.log1p(math.exp(-value)) if value > 0 else math.log1p(math.exp(value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)
