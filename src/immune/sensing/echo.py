from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from immune.config.spec import QuestionSpec
from immune.reflexes.findings import Finding
from immune.sensing.signals import SensorReading

_PREFIX = "echo__"
_UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EchoField:
    name: str
    kind: Literal["noul", "choice", "score"]
    description: str = ""
    options: tuple[str, ...] = ()
    minimum: int = 0

    @property
    def key(self) -> str:
        return f"{_PREFIX}{self.name}"


@dataclass(frozen=True, slots=True)
class EchoResult:
    distributions: Mapping[str, Mapping[str, float]]
    findings: tuple[Finding, ...]


class SchemaEcho:
    MAX_OPTIONS = 12
    MAX_LEVELS = 10

    def __init__(self, min_agreement: float = 0.2) -> None:
        self._min_agreement = min_agreement

    def fields(self, schema: Mapping[str, Any] | None) -> tuple[EchoField, ...]:
        properties = (schema or {}).get("properties")
        if not isinstance(properties, Mapping):
            return ()
        fields: list[EchoField] = []
        for name, definition in properties.items():
            field = self._field(str(name), definition) if isinstance(definition, Mapping) else None
            if field is not None:
                fields.append(field)
        return tuple(fields)

    def _field(self, name: str, definition: Mapping[str, Any]) -> EchoField | None:
        description = str(definition.get("description") or "")
        enum = definition.get("enum")
        if (
            isinstance(enum, list)
            and 2 <= len(enum) <= self.MAX_OPTIONS
            and all(isinstance(item, str) for item in enum)
        ):
            return EchoField(name, "choice", description, tuple(enum))
        if definition.get("type") == "boolean":
            return EchoField(name, "noul", description)
        if definition.get("type") == "integer":
            low, high = definition.get("minimum"), definition.get("maximum")
            if isinstance(low, int) and isinstance(high, int) and 1 <= high - low < self.MAX_LEVELS:
                return EchoField(name, "score", description, tuple(str(level) for level in range(low, high + 1)), low)
        return None

    def questions(self, fields: tuple[EchoField, ...]) -> list[QuestionSpec]:
        questions: list[QuestionSpec] = []
        for field in fields:
            hint = f" ({field.description})" if field.description else ""
            if field.kind == "choice":
                options = {option: option for option in field.options}
                options[_UNKNOWN] = "Cannot tell from the provided input"
                questions.append(
                    QuestionSpec(
                        key=field.key,
                        kind="choice",
                        options=options,
                        text=f"Based only on the provided input, which value is correct for '{field.name}'{hint}?",
                    )
                )
            elif field.kind == "noul":
                questions.append(
                    QuestionSpec(
                        key=field.key,
                        kind="noul",
                        text=f"Based only on the provided input, '{field.name}'{hint} is true.",
                    )
                )
            else:
                questions.append(
                    QuestionSpec(
                        key=field.key,
                        kind="score",
                        levels=field.options,
                        text=f"Based only on the provided input, what is the correct value of '{field.name}'{hint}?",
                    )
                )
        return questions

    def compare(
        self,
        fields: tuple[EchoField, ...],
        structured: Mapping[str, Any] | None,
        reading: SensorReading,
        min_agreement: float | None = None,
    ) -> EchoResult:
        threshold = self._min_agreement if min_agreement is None else min_agreement
        distributions: dict[str, dict[str, float]] = {}
        findings: list[Finding] = []
        for field in fields:
            signal = reading.get(field.key)
            if signal is None:
                continue
            if field.kind == "noul":
                distribution = {"true": signal.probability, "false": 1 - signal.probability}
            else:
                distribution = {key: value for key, value in signal.distribution.items() if key != _UNKNOWN}
            distributions[field.name] = distribution
            chosen = (structured or {}).get(field.name)
            agreement = self._agreement(field, distribution, chosen)
            if agreement is not None and agreement < threshold:
                findings.append(
                    Finding(
                        "output.echo_disagreement",
                        f"{field.name}={chosen!r} has independent probability {agreement:.2f}",
                        subject=field.name,
                    )
                )
        return EchoResult(distributions=distributions, findings=tuple(findings))

    @staticmethod
    def _agreement(field: EchoField, distribution: Mapping[str, float], chosen: Any) -> float | None:
        if chosen is None:
            return None
        total = sum(distribution.values())
        if total <= 0:
            return None
        if field.kind == "noul" and isinstance(chosen, bool):
            return distribution["true" if chosen else "false"] / total
        return distribution.get(str(chosen), 0.0) / total
