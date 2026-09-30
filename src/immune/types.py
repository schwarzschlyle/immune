from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from typing import Any


class Mode(StrEnum):
    AUTO = "auto"
    OBSERVE = "observe"
    STRICT = "strict"
    OFF = "off"


class Stage(StrEnum):
    OPERATOR = "operator"
    INPUT = "input"
    DATA = "data"
    TOOL = "tool"
    OUTPUT = "output"


class Sink(StrEnum):
    TEXT = "text"
    SOFTWARE = "software"
    TOOL = "tool"
    DATA = "data"


class Taint(StrEnum):
    CLEAN = "clean"
    EXTERNAL = "external"
    SUSPICIOUS = "suspicious"

    @property
    def rank(self) -> int:
        return _TAINT_RANK[self]

    def escalate(self, other: Taint) -> Taint:
        return other if other.rank > self.rank else self


_TAINT_RANK = {Taint.CLEAN: 0, Taint.EXTERNAL: 1, Taint.SUSPICIOUS: 2}


class Action(StrEnum):
    ALLOW = "allow"
    ANNOTATE = "annotate"
    AUGMENT = "augment"
    CONFIRM = "confirm"
    HOLD = "hold"
    DENY = "deny"
    NEUTRALIZE = "neutralize"
    REFUSE = "refuse"
    REDIRECT = "redirect"
    REWRITE = "rewrite"
    HANDOFF = "handoff"
    END_SESSION = "end_session"

    @property
    def severity(self) -> int:
        return _ACTION_SEVERITY[self]

    @property
    def intervenes(self) -> bool:
        return self.severity > _ACTION_SEVERITY[Action.ANNOTATE]

    @classmethod
    def most_severe(cls, actions: list[Action] | tuple[Action, ...]) -> Action:
        return max(actions, key=lambda action: action.severity, default=cls.ALLOW)


_ACTION_SEVERITY = {
    Action.ALLOW: 0,
    Action.ANNOTATE: 1,
    Action.AUGMENT: 2,
    Action.CONFIRM: 3,
    Action.HOLD: 4,
    Action.DENY: 5,
    Action.NEUTRALIZE: 5,
    Action.REFUSE: 6,
    Action.REDIRECT: 6,
    Action.REWRITE: 7,
    Action.HANDOFF: 8,
    Action.END_SESSION: 9,
}


@dataclass(frozen=True, slots=True)
class Hit:
    threat: str
    invariant: str
    stage: Stage
    probability: float
    action: Action
    enforced: bool
    evidence: tuple[str, ...] = ()
    frameworks: tuple[str, ...] = ()
    organ: str | None = None
    floor: str | None = None


@dataclass(frozen=True, slots=True)
class SensorInfo:
    name: str
    model: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    calls: int = 0

    @classmethod
    def deterministic_only(cls) -> SensorInfo:
        return cls(name="tier0_only")


@dataclass(frozen=True, slots=True)
class Verdict:
    trace_id: str
    site: str
    session_id: str | None
    action: Action
    would_action: Action
    hits: tuple[Hit, ...] = ()
    echo: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    taint: Taint = Taint.CLEAN
    session_risk: float = 0.0
    sensor: SensorInfo = field(default_factory=SensorInfo.deterministic_only)
    config_hash: str = ""
    spec_version: str = ""
    explanation: str = ""

    @property
    def blocked(self) -> bool:
        return self.action.intervenes

    @property
    def enforced_hits(self) -> tuple[Hit, ...]:
        return tuple(hit for hit in self.hits if hit.enforced)

    def threats(self) -> set[str]:
        return {hit.threat for hit in self.hits}

    def to_dict(self) -> dict[str, Any]:
        plain: dict[str, Any] = _plain(asdict(self))
        return plain

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Verdict:
        hits = tuple(
            Hit(
                threat=hit["threat"],
                invariant=hit["invariant"],
                stage=Stage(hit["stage"]),
                probability=float(hit["probability"]),
                action=Action(hit["action"]),
                enforced=bool(hit["enforced"]),
                evidence=tuple(hit.get("evidence", ())),
                frameworks=tuple(hit.get("frameworks", ())),
                organ=hit.get("organ"),
                floor=hit.get("floor"),
            )
            for hit in data.get("hits", ())
        )
        sensor = data.get("sensor") or {}
        return cls(
            trace_id=data["trace_id"],
            site=data["site"],
            session_id=data.get("session_id"),
            action=Action(data["action"]),
            would_action=Action(data["would_action"]),
            hits=hits,
            echo={field: dict(values) for field, values in (data.get("echo") or {}).items()},
            taint=Taint(data.get("taint", Taint.CLEAN.value)),
            session_risk=float(data.get("session_risk", 0.0)),
            sensor=SensorInfo(**sensor) if sensor else SensorInfo.deterministic_only(),
            config_hash=data.get("config_hash", ""),
            spec_version=data.get("spec_version", ""),
            explanation=data.get("explanation", ""),
        )


@dataclass(frozen=True, slots=True)
class SiteStatus:
    site: str
    archetype: str
    organs: tuple[str, ...]
    calls: int
    enforced: tuple[str, ...]
    observed: tuple[str, ...]
    posture: tuple[str, ...]
    anonymous_calls: int = 0


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
