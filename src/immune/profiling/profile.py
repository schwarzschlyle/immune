from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from immune.config.settings import SiteSettings
from immune.config.spec import OrganRule, PanelFacts, QuestionSpec, Spec
from immune.core.conversation import Conversation
from immune.profiling.capabilities import CapabilityInferrer, ToolCapabilities
from immune.sensing.signals import SensorReading
from immune.types import Sink

_ATTRIBUTES = (
    "talks_to_end_users",
    "may_talk_to_minors",
    "handles_personal_data",
    "processes_external_content",
    "represents_business",
)
_ALLERGIES = ("grants_unlimited_discretion", "demands_unconditional_compliance", "hides_ai_identity")
_UNRENDERED_ARCHETYPES = frozenset({"code_assistant", "content_generation"})
_ATTRIBUTE_THRESHOLD = 0.6
_ARCHETYPE_CONFIDENCE = 0.5

ProfileSource = Literal["provisional", "inferred", "pinned"]


@dataclass(frozen=True, slots=True)
class SiteProfile:
    archetype: str = "unknown"
    attributes: frozenset[str] = field(default_factory=frozenset)
    regulated_domain: str = "none"
    tools: Mapping[str, ToolCapabilities] = field(default_factory=dict)
    sink: Sink = Sink.TEXT
    organs: frozenset[str] = field(default_factory=frozenset)
    allergies: tuple[str, ...] = ()
    confidence: float = 0.0
    source: ProfileSource = "provisional"
    user_facing_override: bool | None = None

    @property
    def user_facing(self) -> bool:
        if self.user_facing_override is not None:
            return self.user_facing_override
        return self.sink is Sink.TEXT and "talks_to_end_users" in self.attributes

    @property
    def rendered(self) -> bool:
        return self.sink is Sink.TEXT and self.archetype not in _UNRENDERED_ARCHETYPES

    @property
    def interactive(self) -> bool:
        return self.user_facing

    def capabilities(self, tool: str) -> ToolCapabilities:
        return self.tools.get(tool, ToolCapabilities())

    def any_capability(self, capability: str) -> bool:
        return any(capabilities.has(capability) for capabilities in self.tools.values())

    def facts(self, conversation: Conversation, minor: bool = False) -> PanelFacts:
        return PanelFacts(
            has_operator=bool(conversation.operator_text.strip()),
            has_context=conversation.has_context,
            has_tools=conversation.has_tools,
            user_facing=self.user_facing,
            minor=minor,
            organs=self.organs,
        )

    def merged(self, other: SiteProfile) -> SiteProfile:
        if self.source == "pinned":
            return replace(self, organs=self.organs | other.organs)
        return replace(
            other,
            attributes=self.attributes | other.attributes,
            organs=self.organs | other.organs,
            tools={**other.tools, **{name: caps for name, caps in self.tools.items() if caps.configured}},
            user_facing_override=self.user_facing_override,
        )


class OrganSelector:
    def __init__(self, spec: Spec) -> None:
        self._rules = {name: organ.when for name, organ in spec.organs.items()}

    def select(self, profile: SiteProfile, has_tools: bool) -> frozenset[str]:
        return frozenset(name for name, rule in self._rules.items() if self._matches(rule, profile, has_tools))

    @staticmethod
    def _matches(rule: OrganRule, profile: SiteProfile, has_tools: bool) -> bool:
        capabilities = {name for caps in profile.tools.values() for name in caps.names}
        return any(
            (
                bool(rule.has_tools) and has_tools,
                profile.archetype in rule.archetypes,
                bool(capabilities & set(rule.capabilities)),
                profile.sink in rule.sinks,
                bool(profile.attributes & set(rule.attributes)),
            )
        )


class ProfileInferrer:
    def __init__(self, spec: Spec) -> None:
        self._spec = spec
        self._capabilities = CapabilityInferrer()
        self._organs = OrganSelector(spec)

    def questions(self) -> tuple[QuestionSpec, ...]:
        return self._spec.panel("operator").questions

    def provisional(self, conversation: Conversation, settings: SiteSettings | None) -> SiteProfile:
        tools = self._tools(conversation, settings)
        profile = SiteProfile(
            tools=tools,
            sink=Sink.SOFTWARE if conversation.expects_structure else Sink.TEXT,
            user_facing_override=settings.user_facing if settings else None,
        )
        if settings and settings.archetype:
            profile = replace(profile, archetype=settings.archetype, source="pinned", confidence=1.0)
        return self._with_organs(profile, conversation, settings)

    def inferred(
        self,
        provisional: SiteProfile,
        reading: SensorReading,
        conversation: Conversation,
        settings: SiteSettings | None,
    ) -> SiteProfile:
        archetype_signal = reading.get("archetype")
        archetype, confidence = provisional.archetype, provisional.confidence
        if archetype_signal is not None and provisional.source != "pinned":
            candidate = str(archetype_signal.value or "unknown")
            if archetype_signal.probability >= _ARCHETYPE_CONFIDENCE:
                archetype, confidence = candidate, archetype_signal.probability
        domain_signal = reading.get("regulated_domain")
        attributes = frozenset(name for name in _ATTRIBUTES if self._holds(reading, name))
        allergies = tuple(name for name in _ALLERGIES if self._holds(reading, name))
        profile = replace(
            provisional,
            archetype=archetype,
            confidence=confidence,
            attributes=provisional.attributes | attributes,
            regulated_domain=str(domain_signal.value) if domain_signal and domain_signal.value else "none",
            allergies=allergies,
            source="pinned" if provisional.source == "pinned" else "inferred",
        )
        return self._with_organs(profile, conversation, settings)

    def refreshed(self, profile: SiteProfile, conversation: Conversation, settings: SiteSettings | None) -> SiteProfile:
        tools = {**self._tools(conversation, settings), **profile.tools}
        attributes = profile.attributes | ({"processes_external_content"} if conversation.data else set())
        refreshed = replace(profile, tools=tools, attributes=frozenset(attributes))
        return self._with_organs(self._pinned(refreshed, settings), conversation, settings)

    @staticmethod
    def _pinned(profile: SiteProfile, settings: SiteSettings | None) -> SiteProfile:
        """Apply what the site's settings pin, so configuration wins over a profile stored before it changed."""
        if settings is None:
            return profile
        if settings.user_facing is not None and profile.user_facing_override != settings.user_facing:
            profile = replace(profile, user_facing_override=settings.user_facing)
        if settings.archetype and profile.archetype != settings.archetype:
            profile = replace(profile, archetype=settings.archetype, source="pinned", confidence=1.0)
        return profile

    def _with_organs(
        self, profile: SiteProfile, conversation: Conversation, settings: SiteSettings | None
    ) -> SiteProfile:
        organs = self._organs.select(profile, conversation.has_tools or bool(profile.tools))
        pinned = frozenset(settings.organs) if settings else frozenset()
        return replace(profile, organs=profile.organs | organs | pinned)

    def _tools(self, conversation: Conversation, settings: SiteSettings | None) -> dict[str, ToolCapabilities]:
        configured = settings.tools if settings else {}
        tools: dict[str, ToolCapabilities] = {}
        for tool in conversation.tools:
            capabilities = self._capabilities.infer(tool)
            override = configured.get(tool.name)
            if override is not None:
                capabilities = capabilities.overridden(
                    egress=override.egress,
                    writes_state=override.writes_state,
                    executes=override.executes,
                    reads_private=override.reads_private,
                    irreversible=override.irreversible,
                )
            tools[tool.name] = capabilities
        return tools

    @staticmethod
    def _holds(reading: SensorReading, key: str) -> bool:
        signal = reading.get(key)
        return signal is not None and signal.probability >= _ATTRIBUTE_THRESHOLD
