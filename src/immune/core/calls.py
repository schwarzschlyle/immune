from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from immune.codecs.base import Codec, RequestContext
from immune.config.settings import SiteSettings
from immune.config.spec import QuestionSpec
from immune.core.candidates import Candidate, CandidateNumbers
from immune.core.conversation import Conversation, Locator, Reply, Segment, ToolCall
from immune.core.decide import Decision
from immune.core.documents import JsonDocument, TextEdit
from immune.profiling.profile import SiteProfile
from immune.profiling.registry import Site
from immune.reflexes.findings import Finding
from immune.sensing.echo import EchoField
from immune.sensing.panels import ViewedText
from immune.sensing.quota import RequestPriority
from immune.sensing.signals import SensorReading
from immune.sessions.resolver import SessionIdentity
from immune.sessions.state import SessionState
from immune.types import Verdict


@dataclass(frozen=True, slots=True)
class DataItem:
    item_id: str
    segment: Segment
    view: ViewedText
    findings: tuple[Finding, ...]
    digest: str
    cached: SensorReading | None = None


SensorRequest = tuple[dict[str, Any], Sequence[QuestionSpec]]
SensorResults = Mapping[str, SensorReading | BaseException]


@dataclass(frozen=True, slots=True)
class SensingPlan:
    requests: Mapping[str, SensorRequest] = field(default_factory=dict)
    batches: Mapping[str, tuple[DataItem, ...]] = field(default_factory=dict)
    priorities: Mapping[str, RequestPriority] = field(default_factory=dict)
    aliases: Mapping[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.requests

    def unavailable(self, error: BaseException) -> dict[str, SensorReading | BaseException]:
        return dict.fromkeys(self.requests, error)

    def priority(self, name: str) -> RequestPriority:
        return self.priorities.get(name, RequestPriority.FLOOR)

    def result(self, results: SensorResults, name: str) -> SensorReading | BaseException | None:
        return results.get(self.aliases.get(name, name))

    def names(self) -> list[str]:
        return [*self.requests, *self.aliases]


@dataclass(frozen=True, slots=True)
class CallCount:
    repeats: int
    total: int


@dataclass(slots=True)
class PreparedCall:
    trace_id: str
    codec: Codec
    context: RequestContext
    conversation: Conversation
    document: JsonDocument
    request_changed: bool
    site: Site
    site_settings: SiteSettings | None
    identity: SessionIdentity
    session: SessionState
    user_view: ViewedText | None
    user_findings: tuple[Finding, ...]
    data_items: tuple[DataItem, ...]
    tool_findings: tuple[Finding, ...]
    echo_fields: tuple[EchoField, ...]
    streaming: bool
    reused_input: SensorReading | None
    input_digest: str | None
    started_at: float = field(default_factory=time.perf_counter)
    candidates: tuple[Candidate, ...] = ()
    sanitation: Mapping[Locator, TextEdit] = field(default_factory=dict)
    numbering: CandidateNumbers = field(default_factory=CandidateNumbers)
    outbound_plan: SensingPlan | None = None
    outbound_candidates_failed: bool = False
    warnings: list[str] = field(default_factory=list)
    vaccines_off: frozenset[str] = frozenset()

    @property
    def profile(self) -> SiteProfile:
        return self.site.profile

    @property
    def latest_user_text(self) -> str:
        latest = self.conversation.latest_user
        return latest.text if latest else ""

    @property
    def user_turn(self) -> int:
        latest = self.conversation.latest_user
        return latest.turn if latest else 0


@dataclass(slots=True)
class InboundResult:
    input_reading: SensorReading
    item_readings: Mapping[str, SensorReading]
    profile_reading: SensorReading | None = None
    degraded: tuple[str, ...] = ()
    candidate_reading: SensorReading | None = None
    candidates_failed: bool = False


@dataclass(slots=True)
class InboundDecision:
    decisions: list[Decision]
    block_text: str | None = None
    refusal: bool = False
    neutralized: list[TextEdit] = field(default_factory=list)
    augment: bool = False

    @property
    def blocks(self) -> bool:
        return self.block_text is not None

    @property
    def rewrites(self) -> bool:
        return bool(self.neutralized)


@dataclass(slots=True)
class ParsedReply:
    body: dict[str, Any]
    reply: Reply
    structured: Mapping[str, Any] | None
    output_findings: list[Finding]
    call_findings: dict[str, list[Finding]]
    calls: dict[str, ToolCall]
    confirmed: set[str]
    candidates: list[Candidate] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Outcome:
    verdict: Verdict
    content: bytes | None
    content_type: str = "application/json"
    replaced: bool = False

    @property
    def passthrough(self) -> bool:
        return self.content is None
