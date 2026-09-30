from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from immune.config.spec import QuestionSpec, Spec
from immune.config.yaml_io import load_yaml
from immune.errors import ConfigError
from immune.evidence.collect import SignalRow
from immune.evidence.metrics import Metrics, Scored, finite
from immune.sensing.priority import RequestPrioritizer
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading

VARIANT_MARK = "~"


class QuestionVariants:
    def __init__(self, variants: Mapping[str, Sequence[str]]) -> None:
        self._variants = {key: tuple(texts) for key, texts in variants.items()}

    @classmethod
    def load(cls, path: Path) -> QuestionVariants:
        try:
            payload = load_yaml(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"cannot read question variants {path}: {error}") from error
        if not isinstance(payload, dict):
            raise ConfigError(f"{path} must map question keys to lists of alternative texts")
        return cls({str(key): [str(text) for text in texts] for key, texts in payload.items()})

    def expand(self, questions: Sequence[QuestionSpec]) -> list[QuestionSpec]:
        expanded = list(questions)
        for question in questions:
            prefix, _, base = question.key.rpartition("__")
            for index, text in enumerate(self._variants.get(base.removesuffix("__defanged"), ()), start=1):
                if base.endswith("__defanged"):
                    continue
                key = f"{prefix}__{base}{VARIANT_MARK}{index}" if prefix else f"{base}{VARIANT_MARK}{index}"
                expanded.append(question.model_copy(update={"key": key, "text": text}))
        return expanded


class VariantSensor(Sensor):
    name: ClassVar[str] = "variants"

    def __init__(self, inner: Sensor, variants: QuestionVariants) -> None:
        self._inner = inner
        self._variants = variants

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        return await self._inner.read(state, self._variants.expand(questions))

    async def close(self) -> None:
        await self._inner.close()


@dataclass(frozen=True, slots=True)
class VariantResult:
    key: str
    variant: int
    samples: int
    roc_auc: float | None
    ece: float | None


class QuestionSearch:
    def __init__(self, spec: Spec) -> None:
        self._threats: dict[str, set[str]] = defaultdict(set)
        for threat, head in spec.heads.items():
            for feature in head.features:
                if feature.signal:
                    self._threats[feature.signal].add(threat)

    def rank(self, rows: Sequence[SignalRow]) -> dict[str, list[VariantResult]]:
        grouped: dict[tuple[str, int], list[Scored]] = defaultdict(list)
        for row in rows:
            base, _, number = RequestPrioritizer.base_key(row.key).partition(VARIANT_MARK)
            labels = [row.labels[threat] for threat in self._threats.get(base, ()) if threat in row.labels]
            if labels:
                grouped[(base, int(number or 0))].append(Scored(row.probability, any(labels)))
        ranked: dict[str, list[VariantResult]] = defaultdict(list)
        for (base, variant), items in grouped.items():
            ranked[base].append(
                VariantResult(base, variant, len(items), finite(Metrics.roc_auc(items)), finite(Metrics.ece(items)))
            )
        return {key: sorted(results, key=self._order) for key, results in ranked.items()}

    @staticmethod
    def _order(result: VariantResult) -> tuple[float, float]:
        auc = result.roc_auc if result.roc_auc is not None else -1.0
        ece = result.ece if result.ece is not None else 1.0
        return (-auc, ece)
