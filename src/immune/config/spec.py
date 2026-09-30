from __future__ import annotations

import fnmatch
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from importlib import resources
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from immune.config.yaml_io import load_yaml
from immune.errors import SpecError
from immune.types import Action, Sink, Stage, Taint

if TYPE_CHECKING:
    from importlib.abc import Traversable

_DEFAULT_ACTIONS = {
    Sink.TEXT: Action.REWRITE,
    Sink.SOFTWARE: Action.REFUSE,
    Sink.TOOL: Action.DENY,
    Sink.DATA: Action.NEUTRALIZE,
}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True, slots=True)
class PanelFacts:
    has_operator: bool = False
    has_context: bool = False
    has_tools: bool = False
    user_facing: bool = False
    minor: bool = False
    organs: frozenset[str] = field(default_factory=frozenset)
    site: str | None = None


class Condition(_Frozen):
    has_operator: bool | None = None
    has_context: bool | None = None
    has_tools: bool | None = None
    user_facing: bool | None = None
    minor: bool | None = None
    organs: tuple[str, ...] = ()
    sites: tuple[str, ...] = ()

    def holds(self, facts: PanelFacts) -> bool:
        flags = {
            "has_operator": facts.has_operator,
            "has_context": facts.has_context,
            "has_tools": facts.has_tools,
            "user_facing": facts.user_facing,
            "minor": facts.minor,
        }
        for name, actual in flags.items():
            expected = getattr(self, name)
            if expected is not None and expected != actual:
                return False
        if self.sites and (
            facts.site is None or not any(fnmatch.fnmatch(facts.site, pattern) for pattern in self.sites)
        ):
            return False
        return not self.organs or bool(set(self.organs) & facts.organs)


class QuestionSpec(_Frozen):
    key: str
    kind: Literal["noul", "choice", "score"]
    text: str
    options: dict[str, str] = Field(default_factory=dict)
    levels: tuple[str, ...] = ()
    views: bool = False
    when: Condition = Field(default_factory=Condition)

    def bound(self, key: str, **placeholders: str) -> QuestionSpec:
        return self.model_copy(update={"key": key, "text": self.text.format(**placeholders)})


class PanelSpec(_Frozen):
    panel: str
    questions: tuple[QuestionSpec, ...]

    def applicable(self, facts: PanelFacts) -> tuple[QuestionSpec, ...]:
        return tuple(question for question in self.questions if question.when.holds(facts))


class FloorSpec(_Frozen):
    id: str
    threshold: float = 0.0
    requires: Literal["user_facing", "rendered", "view_agreement"] | None = None


class ThreatSpec(_Frozen):
    id: str
    invariant: str
    stage: Stage
    detector: Literal["jev", "candidate", "deterministic"]
    severity: Literal["low", "medium", "high", "critical"]
    threshold: float = 1.0
    frameworks: tuple[str, ...] = ()
    actions: dict[Sink, Action] = Field(default_factory=dict)
    floor: FloorSpec | None = None
    organ: str | None = None
    taints: Taint | None = None

    def action_for(self, sink: Sink) -> Action:
        return self.actions.get(sink, _DEFAULT_ACTIONS[sink])


class FeatureSpec(_Frozen):
    signal: str | None = None
    finding: str | None = None
    options: tuple[str, ...] = ()
    reduce: Literal["raw", "max_view", "divergence"] = "raw"
    transform: Literal["logit", "positive", "invert"] = "logit"
    weight: float = 1.0


class HeadSpec(_Frozen):
    bias: float = 0.0
    features: tuple[FeatureSpec, ...]


class OrganRule(_Frozen):
    has_tools: bool | None = None
    archetypes: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    sinks: tuple[Sink, ...] = ()
    attributes: tuple[str, ...] = ()


class OrganSpec(_Frozen):
    description: str
    when: OrganRule


class EndpointRoute(_Frozen):
    codec: str
    path: str


class EndpointSpec(_Frozen):
    exclude_hosts: tuple[str, ...] = ()
    known_hosts: tuple[str, ...] = ()
    routes: tuple[EndpointRoute, ...]


class InvariantSpec(_Frozen):
    name: str
    statement: str


class Templates(_Frozen):
    redirect: str
    refuse: str
    rewrite: str
    handoff: str
    end_session: str
    crisis_augment: str
    confirm_action: str
    hold_action: str
    deny_action: str
    data_placeholder: str
    redaction: str
    link_removed: str
    reasons: dict[str, str]

    def reason(self, threat: str) -> str:
        return self.reasons.get(threat, self.reasons["default"])


class ReflexPatterns(_Frozen):
    secrets: dict[str, str]
    template_tokens: dict[str, str]
    guard_addressed: dict[str, str]
    sql: dict[str, str]
    paths: dict[str, str]


class Spec:
    def __init__(
        self,
        version: str,
        invariants: Mapping[str, InvariantSpec],
        threats: Mapping[str, ThreatSpec],
        panels: Mapping[str, PanelSpec],
        heads: Mapping[str, HeadSpec],
        organs: Mapping[str, OrganSpec],
        endpoints: EndpointSpec,
        templates: Templates,
        reflexes: ReflexPatterns,
        digest: str,
    ) -> None:
        self.version = version
        self.invariants = dict(invariants)
        self.threats = dict(threats)
        self.panels = dict(panels)
        self.heads = dict(heads)
        self.organs = dict(organs)
        self.endpoints = endpoints
        self.templates = templates
        self.reflexes = reflexes
        self.digest = digest
        self._validate()

    def with_heads(self, heads: Mapping[str, HeadSpec], threats: Mapping[str, ThreatSpec], label: str) -> Spec:
        return Spec(
            version=self.version,
            invariants=self.invariants,
            threats=threats,
            panels=self.panels,
            heads=heads,
            organs=self.organs,
            endpoints=self.endpoints,
            templates=self.templates,
            reflexes=self.reflexes,
            digest=f"{self.digest}+{label}",
        )

    def with_vaccines(
        self,
        threats: Mapping[str, ThreatSpec],
        heads: Mapping[str, HeadSpec],
        questions: Mapping[str, Sequence[QuestionSpec]],
        invariants: Mapping[str, InvariantSpec],
        label: str,
    ) -> Spec:
        clashes = sorted(set(threats) & set(self.threats))
        if clashes:
            raise SpecError(f"vaccines cannot reuse built-in threat ids: {', '.join(clashes)}")
        panels = dict(self.panels)
        for name, added in questions.items():
            panel = self.panel(name)
            known = {question.key for question in panel.questions}
            duplicates = sorted(known & {question.key for question in added})
            if duplicates:
                raise SpecError(f"vaccine questions reuse keys in the {name} panel: {', '.join(duplicates)}")
            panels[name] = panel.model_copy(update={"questions": (*panel.questions, *added)})
        return Spec(
            version=self.version,
            invariants={**self.invariants, **invariants},
            threats={**self.threats, **threats},
            panels=panels,
            heads={**self.heads, **heads},
            organs=self.organs,
            endpoints=self.endpoints,
            templates=self.templates,
            reflexes=self.reflexes,
            digest=f"{self.digest}+{label}",
        )

    def threat(self, threat_id: str) -> ThreatSpec:
        try:
            return self.threats[threat_id]
        except KeyError as error:
            raise SpecError(f"unknown threat {threat_id!r}") from error

    def panel(self, name: str) -> PanelSpec:
        try:
            return self.panels[name]
        except KeyError as error:
            raise SpecError(f"unknown panel {name!r}") from error

    def threats_for(self, stage: Stage) -> tuple[ThreatSpec, ...]:
        return tuple(threat for threat in self.threats.values() if threat.stage is stage)

    def _validate(self) -> None:
        for threat in self.threats.values():
            if threat.invariant not in self.invariants:
                raise SpecError(f"threat {threat.id} references unknown invariant {threat.invariant}")
            if threat.detector in ("jev", "candidate") and threat.id not in self.heads:
                raise SpecError(f"{threat.detector} threat {threat.id} has no head")
            if threat.organ is not None and threat.organ not in self.organs:
                raise SpecError(f"threat {threat.id} references unknown organ {threat.organ}")
        for head_id, head in self.heads.items():
            if head_id not in self.threats:
                raise SpecError(f"head {head_id} has no threat")
            for feature in head.features:
                if (feature.signal is None) == (feature.finding is None):
                    raise SpecError(f"head {head_id} feature needs exactly one of signal or finding")

    @classmethod
    def load(cls, root: Traversable | None = None) -> Spec:
        return _load(root or resources.files("immune").joinpath("spec"))

    @classmethod
    def default(cls) -> Spec:
        return _default()


@cache
def _default() -> Spec:
    return _load(resources.files("immune").joinpath("spec"))


def _load(root: Traversable) -> Spec:
    digest = hashlib.sha256()

    def read(*parts: str) -> Any:
        node = root
        for part in parts:
            node = node.joinpath(part)
        raw = node.read_text(encoding="utf-8")
        digest.update(raw.encode("utf-8"))
        return load_yaml(raw)

    try:
        version = str(root.joinpath("version.txt").read_text(encoding="utf-8")).strip()
        threats = {item["id"]: ThreatSpec.model_validate(item) for item in read("threats.yaml")}
        panels = {
            name: PanelSpec.model_validate(read("questions", f"{name}.yaml"))
            for name in ("operator", "input", "data", "tool", "output", "candidates")
        }
        return Spec(
            version=version,
            invariants={key: InvariantSpec.model_validate(value) for key, value in read("invariants.yaml").items()},
            threats=threats,
            panels=panels,
            heads={key: HeadSpec.model_validate(value) for key, value in read("heads.yaml").items()},
            organs={key: OrganSpec.model_validate(value) for key, value in read("organs.yaml").items()},
            endpoints=EndpointSpec.model_validate(read("endpoints.yaml")),
            templates=Templates.model_validate(read("templates.yaml")),
            reflexes=ReflexPatterns.model_validate(read("reflexes.yaml")),
            digest=digest.hexdigest()[:16],
        )
    except (OSError, ValidationError, yaml.YAMLError, KeyError, TypeError) as error:
        raise SpecError(f"invalid immune spec: {error}") from error
