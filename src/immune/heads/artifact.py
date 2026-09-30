from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from immune.config.spec import Spec
from immune.errors import ConfigError
from immune.heads.calibration import Calibrator

_LOGGER = logging.getLogger("immune")
FORMAT = 1


@dataclass(frozen=True, slots=True)
class TrainedHead:
    weights: tuple[float, ...]
    bias: float
    calibration: Calibrator
    threshold: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights),
            "bias": self.bias,
            "calibration": self.calibration.to_dict(),
            "threshold": self.threshold,
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainedHead:
        return cls(
            weights=tuple(float(value) for value in data["weights"]),
            bias=float(data["bias"]),
            calibration=Calibrator.from_dict(data.get("calibration") or {}),
            threshold=None if data.get("threshold") is None else float(data["threshold"]),
            metrics={key: float(value) for key, value in (data.get("metrics") or {}).items()},
        )


@dataclass(frozen=True, slots=True)
class HeadsArtifact:
    spec_version: str
    jev_model: str
    heads: Mapping[str, TrainedHead]
    data: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "spec_version": self.spec_version,
            "jev_model": self.jev_model,
            "created_at": self.created_at,
            "data": dict(self.data),
            "heads": {threat: head.to_dict() for threat, head in sorted(self.heads.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HeadsArtifact:
        if data.get("format") != FORMAT:
            raise ConfigError(f"unsupported heads artifact format {data.get('format')!r}")
        return cls(
            spec_version=str(data["spec_version"]),
            jev_model=str(data["jev_model"]),
            heads={threat: TrainedHead.from_dict(entry) for threat, entry in data["heads"].items()},
            data=dict(data.get("data") or {}),
            created_at=str(data.get("created_at", "")),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> HeadsArtifact:
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise ConfigError(f"cannot read heads artifact {path}: {error}") from error

    def compatible(self, spec: Spec, jev_model: str) -> str | None:
        if self.spec_version != spec.version:
            return f"trained for spec {self.spec_version}, running spec {spec.version}"
        if self.jev_model != jev_model:
            return f"trained on {self.jev_model}, running {jev_model}"
        return None

    def apply(self, spec: Spec) -> Spec:
        heads = dict(spec.heads)
        threats = dict(spec.threats)
        for threat_id, trained in self.heads.items():
            head = heads.get(threat_id)
            if head is None or len(head.features) != len(trained.weights):
                _LOGGER.warning("immune: skipping trained head %s; its features no longer match the spec", threat_id)
                continue
            features = tuple(
                feature.model_copy(update={"weight": weight})
                for feature, weight in zip(head.features, trained.weights, strict=True)
            )
            heads[threat_id] = head.model_copy(update={"features": features, "bias": trained.bias})
            threat = threats.get(threat_id)
            if threat is not None and trained.threshold is not None:
                update: dict[str, Any] = {"threshold": trained.threshold}
                if threat.floor is not None:
                    update["floor"] = threat.floor.model_copy(update={"threshold": trained.threshold})
                threats[threat_id] = threat.model_copy(update=update)
        return spec.with_heads(heads, threats, f"heads-{self.created_at or 'trained'}")

    def calibrators(self) -> dict[str, Calibrator]:
        return {threat: head.calibration for threat, head in self.heads.items()}
