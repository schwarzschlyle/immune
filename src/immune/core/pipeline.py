from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from immune.codecs.base import Codec, ReplyEdit, RequestContext, Structured
from immune.config.settings import Settings, SiteSettings
from immune.config.spec import PanelFacts, QuestionSpec, Spec, ThreatSpec
from immune.core.assess import Assessment, Assessor, strongest
from immune.core.calls import (
    CallCount,
    DataItem,
    InboundDecision,
    InboundResult,
    Outcome,
    ParsedReply,
    PreparedCall,
    SensingPlan,
    SensorRequest,
    SensorResults,
)
from immune.core.candidates import Candidate, CandidateNumbers, Nominator
from immune.core.compose import ResponseComposer
from immune.core.conversation import Channel, Conversation, Locator, Reply, Segment, ToolCall
from immune.core.decide import Decider, Decision, applied, would
from immune.core.documents import JsonDocument, TextEdit
from immune.core.segment import EmbeddedDataSegmenter
from immune.core.sse import ServerSentEvent, ServerSentEvents
from immune.core.streaming import ProgressiveScreen
from immune.organs.base import OrganSet
from immune.profiling.posture import PostureAssessor
from immune.profiling.profile import ProfileInferrer
from immune.profiling.registry import Site, SiteRegistry
from immune.reflexes import ReflexSuite
from immune.reflexes.findings import Finding
from immune.reflexes.memo import InboundScan, TextMemo
from immune.reflexes.outbound import Canary, LinkInspector
from immune.reflexes.tools import DestinationProvenance
from immune.sensing.echo import SchemaEcho
from immune.sensing.guard import ReadingCache
from immune.sensing.panels import PanelCompiler, StateBuilder, ViewedText
from immune.sensing.priority import RequestPrioritizer
from immune.sensing.quota import RequestPriority, SensingContext
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading
from immune.sessions.confirm import ConfirmationSeal
from immune.sessions.provenance import Provenance
from immune.sessions.resolver import CallContext, SessionIdentity, SessionResolver
from immune.sessions.state import SessionState
from immune.sessions.store import SessionStore
from immune.sessions.transcript import TranscriptCache
from immune.telemetry.observers import CallStart, CallSummary, Observers, SensorExchange, ToolSummary
from immune.telemetry.stats import PromotionLedger
from immune.telemetry.throttle import ThrottledLog
from immune.telemetry.verdicts import VerdictIndex, VerdictRecorder
from immune.types import Action, Hit, Mode, SensorInfo, Sink, Stage, Taint, Verdict
from immune.vaccines import VaccineReflexes
from immune.vaccines.switchboard import Activation

_LOGGER = logging.getLogger("immune")
_THROTTLED = ThrottledLog(_LOGGER)
_ATTACK_THREATS = (
    "input.override",
    "input.framing_bypass",
    "input.extraction",
    "input.obfuscation",
    "input.judge_manipulation",
    "input.severe_harm",
)
_BLOCKING = frozenset({Action.REDIRECT, Action.REFUSE, Action.REWRITE, Action.HANDOFF, Action.END_SESSION})
_ITEMS_PER_CALL = 8
_SPAN_THREATS = frozenset(
    {"output.exfil_link", "output.unsafe_markup", "output.secret_leak", "output.canary_leak", "output.prompt_copy"}
)
_WHOLE_REPLY = frozenset({Action.REWRITE, Action.REDIRECT, Action.HANDOFF, Action.END_SESSION, Action.REFUSE})


def _replaces(threat: ThreatSpec, sink: Sink) -> bool:
    return threat.action_for(sink) in _WHOLE_REPLY


_SECRET_KINDS = frozenset({"card", "ssn", "iban"})
_REFUSED_AFTER_ERROR = "refused after an internal error (on_internal_error: block)"
_CANDIDATES_IN = "candidates_in"
_CANDIDATES_OUT = "candidates_out"


@dataclass(frozen=True, slots=True)
class PipelineParts:
    spec: Spec
    settings: Settings
    reflexes: ReflexSuite
    segmenter: EmbeddedDataSegmenter
    provenance: Provenance
    sites: SiteRegistry
    inferrer: ProfileInferrer
    posture: PostureAssessor
    sessions: SessionStore
    resolver: SessionResolver
    states: StateBuilder
    compiler: PanelCompiler
    echo: SchemaEcho
    assessor: Assessor
    decider: Decider
    composer: ResponseComposer
    organs: OrganSet
    sensor: Sensor
    canary: Canary | None
    ledger: PromotionLedger
    recorder: VerdictRecorder
    index: VerdictIndex
    item_cache: ReadingCache
    seal: ConfirmationSeal
    observers: Observers
    transcripts: TranscriptCache
    vaccines: VaccineReflexes = field(default_factory=VaccineReflexes)
    activation: Activation = field(default_factory=Activation)


class CallPipeline:
    def __init__(self, parts: PipelineParts) -> None:
        self._parts = parts
        reflexes = parts.reflexes
        self._prioritizer = RequestPrioritizer(parts.spec, parts.decider.policy)
        self._nominator = Nominator(parts.spec)
        self._memo = TextMemo(
            reveal=reflexes.unicode.reveal,
            clean=reflexes.hidden_markup.clean,
            destinations=reflexes.destinations.from_text,
            inbound=self._scan_inbound,
        )

    @property
    def parts(self) -> PipelineParts:
        return self._parts

    def reconfigure(self, parts: PipelineParts) -> None:
        self._prioritizer = RequestPrioritizer(parts.spec, parts.decider.policy)
        self._nominator = Nominator(parts.spec)
        self._parts = parts

    def prepare(self, codec: Codec, context: RequestContext) -> PreparedCall:
        parsed = codec.parse_request(context)
        if parsed.previous_response_id:
            parsed = parsed.with_history(self._parts.transcripts.load(parsed.previous_response_id))
        document = JsonDocument(dict(context.body))
        edits, cleaned, sanitation = self._sanitize(parsed)
        conversation = self._parts.segmenter.split(parsed.with_segments(cleaned))
        site, settings = self._site(conversation)
        off = self._parts.activation.off(settings)
        trace_id = uuid.uuid4().hex[:16]
        identity = self._parts.resolver.resolve(conversation, trace_id, codec.server_side_history)
        session = self._session(identity)
        if identity.anonymous:
            self._parts.sites.note_anonymous(site)
        if not identity.continued:
            session.taint_with(Taint.EXTERNAL)
        with session.lock:
            session.begin_turn(conversation.latest_turn)
            self._learn(conversation, session)
        request_changed = False
        canary = self._parts.canary
        if canary is not None and not context.signed and codec.append_operator_note(document, canary.note):
            request_changed = True
        user_view, user_findings = self._user(conversation, sanitation, site, off)
        data_items = self._items(conversation, sanitation, site, off)
        tool_findings = tuple(self._parts.organs.inspect_request(site.site_id, conversation, site.profile))
        numbering = CandidateNumbers()
        digest = self._input_digest(conversation)
        self._parts.observers.started(CallStart(trace_id, codec.name, conversation.model))
        return PreparedCall(
            trace_id=trace_id,
            codec=codec,
            context=context,
            conversation=conversation,
            document=document,
            request_changed=request_changed,
            site=site,
            site_settings=settings,
            identity=identity,
            session=session,
            user_view=user_view,
            user_findings=user_findings,
            data_items=data_items,
            tool_findings=tool_findings,
            echo_fields=self._parts.echo.fields(conversation.response_schema),
            streaming=codec.is_streaming(context),
            reused_input=session.last_input_reading if digest and digest == session.last_input_digest else None,
            input_digest=digest,
            candidates=self._inbound_candidates(conversation, user_findings, data_items, tool_findings, numbering),
            sanitation={edit.locator: edit for edit in edits},
            numbering=numbering,
            vaccines_off=off,
        )

    def request_body(self, prepared: PreparedCall) -> bytes | None:
        return json.dumps(prepared.document.body).encode("utf-8") if prepared.request_changed else None

    def immediate(self, prepared: PreparedCall) -> Outcome | None:
        if not prepared.session.locked:
            return None
        text = self._parts.composer.template(Action.END_SESSION)
        return self._synthesized(
            prepared,
            InboundDecision(decisions=[], block_text=text),
            InboundResult(input_reading=SensorReading.empty(), item_readings={}),
        )

    def plan_inbound(self, prepared: PreparedCall) -> SensingPlan:
        facts = self._facts(prepared)
        requests: dict[str, SensorRequest] = {}
        batches: dict[str, tuple[DataItem, ...]] = {}
        if prepared.reused_input is None and (prepared.user_view is not None or prepared.echo_fields):
            view = prepared.user_view or ViewedText("", "")
            questions = (
                self._parts.compiler.questions("input", facts, split_view=view.split) if prepared.user_view else []
            )
            questions.extend(self._parts.echo.questions(prepared.echo_fields))
            provided = "\n\n".join(item.segment.text for item in prepared.data_items) or None
            requests["input"] = (self._parts.states.input(prepared.conversation, view, provided), questions)
        fresh = [item for item in prepared.data_items if item.cached is None]
        for index in range(0, len(fresh), _ITEMS_PER_CALL):
            batch = tuple(fresh[index : index + _ITEMS_PER_CALL])
            name = f"data{index // _ITEMS_PER_CALL}"
            batches[name] = batch
            questions = self._parts.compiler.item_questions(
                "data",
                facts,
                "item",
                [item.item_id for item in batch],
                [item.item_id for item in batch if item.view.split],
            )
            state = self._parts.states.data(
                prepared.conversation, [(item.item_id, item.segment, item.view) for item in batch]
            )
            requests[name] = (state, questions)
        if self._claims_profile(prepared):
            state = self._parts.states.operator(prepared.site.template, prepared.conversation)
            requests["profile"] = (state, self._parts.inferrer.questions())
        candidates = self._candidate_request(prepared.conversation, None, prepared.candidates)
        if candidates is not None:
            requests[_CANDIDATES_IN] = candidates
        aliases = self._coalesced(requests) if self._parts.settings.sensor.coalesce else {}
        return SensingPlan(
            requests=requests, batches=batches, priorities=self._priorities(prepared, requests), aliases=aliases
        )

    @staticmethod
    def _coalesced(requests: dict[str, SensorRequest]) -> dict[str, str]:
        if "input" not in requests or "data0" not in requests:
            return {}
        input_state, input_questions = requests.pop("input")
        data_state, data_questions = requests.pop("data0")
        merged = {**data_state, **{key: value for key, value in input_state.items() if key != "provided_content"}}
        requests["input+data0"] = (merged, [*input_questions, *data_questions])
        return {"input": "input+data0", "data0": "input+data0"}

    def _priorities(self, prepared: PreparedCall, requests: Mapping[str, SensorRequest]) -> dict[str, RequestPriority]:
        return {
            name: self._prioritizer.priority(questions, prepared.site.site_id, prepared.site_settings)
            for name, (_, questions) in requests.items()
        }

    async def sense(self, plan: SensingPlan) -> dict[str, SensorReading | BaseException]:
        if plan.is_empty:
            return {}
        results = await asyncio.gather(
            *(self._read(state, questions, plan.priority(name)) for name, (state, questions) in plan.requests.items()),
            return_exceptions=True,
        )
        return dict(zip(plan.requests, results, strict=True))

    async def _read(
        self, state: dict[str, Any], questions: Sequence[QuestionSpec], priority: RequestPriority
    ) -> SensorReading:
        with SensingContext.priority(priority):
            return await self._parts.sensor.read(state, questions)

    def collect_inbound(self, prepared: PreparedCall, plan: SensingPlan, results: SensorResults) -> InboundResult:
        self._report(prepared, plan, results)
        degraded = tuple(name for name in plan.names() if isinstance(plan.result(results, name), BaseException))
        for name in degraded:
            _THROTTLED.warning(
                f"inbound:{name}",
                "immune: sensor unavailable for %s (%s); %s",
                name,
                plan.result(results, name),
                self._outage_note(),
            )
            prepared.warnings.append(f"Jev was unavailable for {name} screening; {self._outage_note()}")
        item_readings = self._item_readings(prepared, plan, results)
        profile_reading = self._reading(plan.result(results, "profile"))
        self._update_profile(prepared, profile_reading, "profile" in degraded)
        input_reading = prepared.reused_input or self._reading(plan.result(results, "input")) or SensorReading.empty()
        candidate_reading = self._reading(plan.result(results, _CANDIDATES_IN))
        return InboundResult(
            input_reading=input_reading,
            item_readings=item_readings,
            profile_reading=profile_reading,
            degraded=degraded,
            candidate_reading=candidate_reading,
            candidates_failed=_CANDIDATES_IN in plan.requests and candidate_reading is None,
        )

    def gate(self, prepared: PreparedCall, inbound: InboundResult) -> InboundDecision:
        parts = self._parts
        profile = prepared.profile
        session = prepared.session
        with session.lock:
            staged = self._staged(prepared.candidates)
            user = self._assessed(
                Stage.INPUT,
                inbound.input_reading,
                prepared.user_findings,
                self._decided(staged[Stage.INPUT], inbound.candidate_reading, inbound.candidates_failed, prepared),
                prepared,
            )
            if prepared.reused_input is None and not inbound.input_reading.is_empty:
                session.risk.update(strongest(user, _ATTACK_THREATS))
                session.last_input_digest = prepared.input_digest
                session.last_input_reading = inbound.input_reading
            user.append(parts.assessor.session(session.risk.probability))
            items: list[Assessment] = []
            for item in prepared.data_items:
                decided = self._decided(
                    [candidate for candidate in staged[Stage.DATA] if candidate.subject == item.item_id],
                    inbound.candidate_reading,
                    inbound.candidates_failed,
                    prepared,
                )
                reading = inbound.item_readings.get(item.item_id, SensorReading.empty())
                items.extend(self._assessed(Stage.DATA, reading, item.findings, decided, prepared, item.item_id))
            input_decisions = parts.decider.decide(
                user, profile.sink, profile, prepared.site.site_id, prepared.site_settings
            )
            data_decisions = parts.decider.decide(
                items, Sink.DATA, profile, prepared.site.site_id, prepared.site_settings
            )
            self._absorb_taints(session, [*input_decisions, *data_decisions])
            if any(decision.hit.threat == "care.minor_signals" for decision in input_decisions):
                session.minor = True
            decision = InboundDecision(decisions=[*input_decisions, *data_decisions])
            blocking = [d.enforced_action for d in input_decisions if d.enforced_action in _BLOCKING]
            if blocking:
                action = Action.most_severe(blocking)
                decision.refusal = profile.sink is Sink.SOFTWARE
                decision.block_text = parts.composer.blocking_text(
                    input_decisions, Action.REFUSE if decision.refusal else action
                )
                session.locked = session.locked or action is Action.END_SESSION
            decision.augment = any(d.enforced_action is Action.AUGMENT for d in input_decisions)
            if prepared.context.signed:
                decision.decisions = [self._unrewritable(item) for item in decision.decisions]
            else:
                decision.neutralized = self._neutralizations(prepared, [*input_decisions, *data_decisions])
        return decision

    def neutralized_body(self, prepared: PreparedCall, decision: InboundDecision) -> bytes:
        prepared.codec.apply_edits(prepared.document, decision.neutralized)
        prepared.request_changed = True
        return json.dumps(prepared.document.body).encode("utf-8")

    def blocked(self, prepared: PreparedCall, inbound: InboundResult, decision: InboundDecision) -> Outcome:
        return self._synthesized(prepared, decision, inbound)

    def refused(self, prepared: PreparedCall, inbound: InboundResult) -> Outcome:
        text = self._parts.composer.template(Action.REFUSE)
        decision = InboundDecision(decisions=[], block_text=text, refusal=True)
        return self._synthesized(prepared, decision, inbound, forced=Action.REFUSE, reason=_REFUSED_AFTER_ERROR)

    def parse(self, prepared: PreparedCall, raw: bytes) -> ParsedReply:
        codec = prepared.codec
        if prepared.streaming:
            body = codec.assemble_stream(ServerSentEvents.parse(raw))
        else:
            loaded = json.loads(raw)
            body = loaded if isinstance(loaded, dict) else {"value": loaded}
        reply = codec.parse_response(body)
        conversation = prepared.conversation
        calls = {f"call_{index + 1}": call for index, call in enumerate(reply.tool_calls)}
        parsed = ParsedReply(
            body=body,
            reply=reply,
            structured=Structured.parse(reply.text, conversation.expects_structure),
            output_findings=self._output_findings(prepared, reply.text),
            call_findings={},
            calls=calls,
            confirmed=set(),
        )
        earlier = Counter(call.signature for call in conversation.calls_in_turn(conversation.latest_turn))
        current: Counter[str] = Counter()
        with prepared.session.lock:
            for position, (call_id, call) in enumerate(calls.items(), start=1):
                current[call.signature] += 1
                counted = CallCount(earlier[call.signature] + current[call.signature], earlier.total() + position)
                parsed.call_findings[call_id] = self._call_findings(prepared, call_id, call, parsed, counted)
        parsed.candidates = self._outbound_candidates(prepared, parsed)
        return parsed

    def plan_outbound(self, prepared: PreparedCall, parsed: ParsedReply) -> SensingPlan:
        facts = self._facts(prepared)
        requests: dict[str, SensorRequest] = {}
        if parsed.reply.text.strip() and not parsed.reply.refused:
            state = self._parts.states.output(prepared.conversation, parsed.reply)
            requests["output"] = (state, self._parts.compiler.questions("output", facts))
        screened = [(call_id, call) for call_id, call in parsed.calls.items() if self._worth_screening(prepared, call)]
        if screened:
            questions = self._parts.compiler.item_questions("tool", facts, "call", [call_id for call_id, _ in screened])
            requests["tool"] = (self._parts.states.tool(prepared.conversation, screened), questions)
        candidates = self._candidate_request(prepared.conversation, parsed.reply, parsed.candidates)
        if candidates is not None:
            requests[_CANDIDATES_OUT] = candidates
        plan = SensingPlan(requests=requests, priorities=self._priorities(prepared, requests))
        prepared.outbound_plan = plan
        return plan

    def collect_outbound(self, prepared: PreparedCall, results: SensorResults) -> tuple[SensorReading, bool]:
        if prepared.outbound_plan is not None:
            self._report(prepared, prepared.outbound_plan, results)
        readings = [result for result in results.values() if isinstance(result, SensorReading)]
        degraded = len(readings) < len(results)
        planned = prepared.outbound_plan is not None and _CANDIDATES_OUT in prepared.outbound_plan.requests
        prepared.outbound_candidates_failed = planned and not isinstance(results.get(_CANDIDATES_OUT), SensorReading)
        if degraded:
            note = self._outage_note()
            _THROTTLED.warning("outbound", "immune: sensor unavailable for outbound screening; %s", note)
            prepared.warnings.append(f"Jev was unavailable for outbound screening; {note}")
        merged = readings[0].merged(*readings[1:]) if readings else SensorReading.empty()
        return merged, degraded

    def finalize(
        self,
        prepared: PreparedCall,
        inbound: InboundResult,
        inbound_decision: InboundDecision,
        parsed: ParsedReply,
        outbound: SensorReading,
    ) -> Outcome:
        verdict, edit, _ = self._conclude(prepared, inbound, inbound_decision, parsed, outbound, streamed=False)
        content: bytes | None = None
        content_type = "text/event-stream" if prepared.streaming else "application/json"
        if not edit.is_noop:
            content = self._encode(prepared, prepared.codec.render_response(parsed.body, edit))
        return Outcome(verdict=verdict, content=content, content_type=content_type, replaced=edit.replaced)

    def finalize_stream(
        self,
        prepared: PreparedCall,
        inbound: InboundResult,
        inbound_decision: InboundDecision,
        parsed: ParsedReply,
        outbound: SensorReading,
        screen: ProgressiveScreen,
    ) -> tuple[Verdict, list[ServerSentEvent]]:
        verdict, edit, output_decisions = self._conclude(
            prepared, inbound, inbound_decision, parsed, outbound, streamed=True
        )
        appendix = f"\n\n{edit.appendix}" if edit.appendix and parsed.reply.text else edit.appendix
        replacements = self._parts.composer.span_replacements(
            [decision for decision in output_decisions if decision.hit.enforced]
        )
        remainder = screen.complete(parsed.reply.text, replacements) + appendix
        final = ReplyEdit(text=screen.emitted_text + appendix, removed_calls=edit.removed_calls, keep_reasoning=True)
        return verdict, screen.continuation(prepared.codec.render_response(parsed.body, final), remainder)

    def progressive(self, prepared: PreparedCall) -> bool:
        settings = self._parts.settings
        site = prepared.site_settings
        choice = site.streaming if site is not None and site.streaming is not None else settings.streaming
        if not prepared.streaming or choice != "progressive" or prepared.conversation.expects_structure:
            return False
        if settings.mode is Mode.STRICT:
            return False
        return not any(
            self._parts.decider.policy.could_enforce(threat, prepared.site.site_id, site)
            for threat in self._parts.spec.threats_for(Stage.OUTPUT)
            if threat.floor is None
            and threat.id not in _SPAN_THREATS
            and threat.id not in prepared.vaccines_off
            and _replaces(threat, prepared.profile.sink)
        )

    def stream_hold(self, prepared: PreparedCall, text: str) -> int | None:
        """Where a streamed reply must stop until Jev decides: the start of the first enforceable candidate."""
        parts = self._parts
        starts = [
            finding.span.start
            for finding in self._output_findings(prepared, text)
            if finding.span is not None
            and finding.threat in _SPAN_THREATS
            and parts.decider.policy.could_enforce(
                parts.spec.threat(finding.threat), prepared.site.site_id, prepared.site_settings
            )
        ]
        return min(starts, default=None)

    def _conclude(
        self,
        prepared: PreparedCall,
        inbound: InboundResult,
        inbound_decision: InboundDecision,
        parsed: ParsedReply,
        outbound: SensorReading,
        streamed: bool,
    ) -> tuple[Verdict, ReplyEdit, list[Decision]]:
        parts = self._parts
        profile = prepared.profile
        session = prepared.session
        site_echo = prepared.site_settings.echo if prepared.site_settings is not None else None
        echo = parts.echo.compare(
            prepared.echo_fields,
            parsed.structured,
            inbound.input_reading,
            site_echo.min_agreement if site_echo is not None else None,
        )
        findings = [*parsed.output_findings, *echo.findings]
        staged = self._staged(parsed.candidates)
        failed = prepared.outbound_candidates_failed
        output = self._assessed(
            Stage.OUTPUT, outbound, findings, self._decided(staged[Stage.OUTPUT], outbound, failed, prepared), prepared
        )
        tools: list[Assessment] = []
        for call_id, call_findings in parsed.call_findings.items():
            call_candidates = [candidate for candidate in staged[Stage.TOOL] if candidate.subject == call_id]
            decided = self._decided(call_candidates, outbound, failed, prepared)
            reading = outbound.scoped(call_id)
            tools.extend(self._assessed(Stage.TOOL, reading, call_findings, decided, prepared, call_id))
        requested = [candidate for candidate in prepared.candidates if candidate.stage is Stage.TOOL]
        tools.extend(self._decided(requested, inbound.candidate_reading, inbound.candidates_failed, prepared))
        tools.extend(parts.assessor.deterministic(self._nominator.deterministic(prepared.tool_findings)))
        site_id, settings = prepared.site.site_id, prepared.site_settings
        output_decisions = parts.decider.decide(output, profile.sink, profile, site_id, settings)
        if streamed:
            output_decisions = [self._delivered(decision) for decision in output_decisions]
        tool_decisions = [
            decision
            for decision in parts.decider.decide(tools, Sink.TOOL, profile, site_id, settings)
            if decision.assessment.subject not in parsed.confirmed
        ]
        edit = parts.composer.compose(
            parsed.reply,
            output_decisions,
            tool_decisions,
            profile.sink,
            parsed.calls,
            profile.interactive,
            inbound_decision.augment,
            prepared.user_turn,
        )
        with session.lock:
            self._absorb_taints(session, output_decisions)
            self._request_confirmations(prepared, tool_decisions, parsed)
            if session.taint is not Taint.CLEAN and parsed.reply.text:
                parts.provenance.absorb(parsed.reply.text)
        decisions = [*inbound_decision.decisions, *output_decisions, *tool_decisions]
        verdict = self._verdict(prepared, decisions, inbound, outbound, echo.distributions)
        self._record(prepared, verdict, decisions, inbound, parsed, parsed.reply.response_id)
        return verdict, edit, output_decisions

    def _inbound_candidates(
        self,
        conversation: Conversation,
        user_findings: Sequence[Finding],
        data_items: Sequence[DataItem],
        tool_findings: Sequence[Finding],
        numbering: CandidateNumbers,
    ) -> tuple[Candidate, ...]:
        nominate = self._nominator.nominate
        latest = conversation.latest_user
        candidates = nominate(user_findings, numbering, Stage.INPUT, "user message", latest.text if latest else "")
        for item in data_items:
            where = f"data from {item.segment.origin}"
            candidates.extend(nominate(item.findings, numbering, Stage.DATA, where, item.segment.text, item.item_id))
        candidates.extend(nominate(tool_findings, numbering, Stage.TOOL, "tool description"))
        return tuple(candidates)

    def _outbound_candidates(self, prepared: PreparedCall, parsed: ParsedReply) -> list[Candidate]:
        nominate = self._nominator.nominate
        numbering = prepared.numbering
        candidates = nominate(parsed.output_findings, numbering, Stage.OUTPUT, "assistant reply", parsed.reply.text)
        for call_id, findings in parsed.call_findings.items():
            call = parsed.calls[call_id]
            text = f"{call.name}({json.dumps(call.arguments, ensure_ascii=False, default=str)})"
            candidates.extend(nominate(findings, numbering, Stage.TOOL, f"tool call {call.name}", text, call_id))
        return candidates

    def _candidate_request(
        self, conversation: Conversation, reply: Reply | None, candidates: Sequence[Candidate]
    ) -> SensorRequest | None:
        questions = self._parts.compiler.candidate_questions(candidates)
        if not questions:
            return None
        return self._parts.states.candidates(conversation, reply, candidates), questions

    @staticmethod
    def _staged(candidates: Iterable[Candidate]) -> dict[Stage, list[Candidate]]:
        staged: dict[Stage, list[Candidate]] = defaultdict(list)
        for candidate in candidates:
            staged[candidate.stage].append(candidate)
        return staged

    def _decided(
        self,
        candidates: Sequence[Candidate],
        reading: SensorReading | None,
        failed: bool,
        prepared: PreparedCall,
    ) -> list[Assessment]:
        if not candidates:
            return []
        act = self._parts.settings.sensor.on_outage == "act_on_candidates"
        return self._parts.assessor.candidates(candidates, reading, failed, act, prepared.site.site_id)

    def _assessed(
        self,
        stage: Stage,
        reading: SensorReading,
        findings: Sequence[Finding],
        decided: Sequence[Assessment],
        prepared: PreparedCall,
        subject: str | None = None,
    ) -> list[Assessment]:
        """Deterministic findings (vaccines), Jev's candidate decisions, then the judged threats.

        A candidate counts as evidence for other threats' heads only once Jev has confirmed it.
        """
        assessor = self._parts.assessor
        confirmed = {assessment.threat.id for assessment in decided if assessment.triggered}
        evidence = [
            finding for finding in findings if not self._nominator.is_candidate(finding) or finding.threat in confirmed
        ]
        return [
            *assessor.deterministic(self._nominator.deterministic(findings), subject=subject),
            *decided,
            *assessor.judged(
                stage,
                reading,
                evidence,
                prepared.profile.organs,
                subject,
                prepared.site.site_id,
                prepared.vaccines_off,
            ),
        ]

    @staticmethod
    def _unrewritable(decision: Decision) -> Decision:
        hit = decision.hit
        if not hit.enforced or hit.action is not Action.NEUTRALIZE:
            return decision
        evidence = (*hit.evidence, "not enforced: the request is signed and cannot be rewritten")
        return Decision(assessment=decision.assessment, hit=replace(hit, enforced=False, evidence=evidence))

    @staticmethod
    def _delivered(decision: Decision) -> Decision:
        hit = decision.hit
        if not hit.enforced or hit.threat in _SPAN_THREATS or hit.action not in _WHOLE_REPLY:
            return decision
        evidence = (*hit.evidence, "not enforced: the reply was already streamed")
        return Decision(assessment=decision.assessment, hit=replace(hit, enforced=False, evidence=evidence))

    def screens(self, prepared: PreparedCall) -> bool:
        settings = prepared.site_settings
        return self._parts.settings.mode is not Mode.OFF and (settings is None or settings.enabled)

    def degraded_verdict(
        self, prepared: PreparedCall, reason: str = "passed through after an internal error"
    ) -> Verdict:
        verdict = Verdict(
            trace_id=prepared.trace_id,
            site=prepared.site.site_id,
            session_id=self._session_id(prepared),
            action=Action.ALLOW,
            would_action=Action.ALLOW,
            config_hash=self._parts.settings.digest,
            spec_version=self._parts.spec.version,
            explanation=reason,
        )
        self._parts.index.add(verdict)
        self._parts.observers.finish(verdict)
        return verdict

    def _outage_note(self) -> str:
        if self._parts.settings.sensor.on_outage == "act_on_candidates":
            return "floor candidates act without Jev"
        return "candidates pass without Jev (sensor.on_outage: pass)"

    def _report(self, prepared: PreparedCall, plan: SensingPlan, results: SensorResults) -> None:
        observers = self._parts.observers
        if not observers.wants_details():
            return
        for name, (state, questions) in plan.requests.items():
            result = results.get(name)
            reading = result if isinstance(result, SensorReading) else None
            error = None
            if result is None:
                error = "no answer before the deadline"
            elif isinstance(result, BaseException):
                error = f"{type(result).__name__}: {result}"
            observers.sensed(SensorExchange(prepared.trace_id, name, state, tuple(questions), reading, error))

    def _summary(self, prepared: PreparedCall, parsed: ParsedReply | None) -> CallSummary:
        conversation = prepared.conversation
        calls = parsed.calls.values() if parsed is not None else ()
        return CallSummary(
            trace_id=prepared.trace_id,
            site=prepared.site.site_id,
            session=self._session_id(prepared),
            provider=prepared.codec.name,
            model=conversation.model,
            user=prepared.latest_user_text,
            data=tuple(item.segment.text for item in prepared.data_items),
            reply=parsed.reply.text if parsed is not None else "",
            tools=tuple(
                ToolSummary(call.name, json.dumps(call.arguments, ensure_ascii=False, default=str)) for call in calls
            ),
            warnings=tuple(prepared.warnings),
        )

    def _session(self, identity: SessionIdentity) -> SessionState:
        if identity.anonymous:
            return SessionState(session_id=identity.session_id, anonymous=True)
        return self._parts.sessions.get(identity.session_id)

    @staticmethod
    def _session_id(prepared: PreparedCall) -> str | None:
        return None if prepared.identity.anonymous else prepared.identity.session_id

    @staticmethod
    def _facts(prepared: PreparedCall) -> PanelFacts:
        facts = replace(
            prepared.profile.facts(prepared.conversation, prepared.session.minor),
            site=prepared.site.site_id,
            off=prepared.vaccines_off,
        )
        if prepared.site.profiled or prepared.profile.sink is not Sink.TEXT:
            return facts
        return replace(facts, user_facing=True)

    def _sanitize(
        self, conversation: Conversation
    ) -> tuple[list[TextEdit], tuple[Segment, ...], dict[Locator, list[Finding]]]:
        memo = self._memo
        edits: list[TextEdit] = []
        cleaned: list[Segment] = []
        findings: dict[Locator, list[Finding]] = {}
        for segment in conversation.segments:
            if segment.channel not in (Channel.USER, Channel.DATA) or segment.historical:
                cleaned.append(segment)
                continue
            revealed = memo.reveal(segment.text)
            text, found = revealed.text, list(revealed.findings)
            if segment.channel is Channel.DATA:
                stripped = memo.clean(text)
                text, found = stripped.text, [*found, *stripped.findings]
            if found:
                edits.append(TextEdit(segment.locator, text))
                findings.setdefault(segment.locator, []).extend(found)
            cleaned.append(segment.with_text(text))
        return edits, tuple(cleaned), findings

    def _site(self, conversation: Conversation) -> tuple[Site, SiteSettings | None]:
        parts = self._parts
        explicit = CallContext.current_site()
        settings = parts.settings.site(explicit) if explicit else None
        provisional = parts.inferrer.provisional(conversation, settings)
        site, created = parts.sites.resolve(parts.sites.key(conversation), provisional, explicit)
        settings = settings or parts.settings.site(site.name)
        refreshed = parts.inferrer.refreshed(site.profile, conversation, settings)
        if refreshed != site.profile:
            parts.sites.update(site, refreshed)
        if created:
            _LOGGER.info(
                "immune: new call site %s (%s, %d tools)", site.name, conversation.provider, len(conversation.tools)
            )
        return site, settings

    def _learn(self, conversation: Conversation, session: SessionState) -> None:
        destinations = self._memo.destinations
        for segment in conversation.segments:
            if segment.channel in (Channel.OPERATOR, Channel.USER):
                session.trusted_destinations.update(destinations(segment.text))
            elif segment.channel is Channel.DATA:
                session.data_destinations.update(destinations(segment.text))
                self._parts.provenance.absorb(segment.text)
        if conversation.data:
            session.taint_with(Taint.EXTERNAL)

    def _scan_inbound(self, text: str) -> InboundScan:
        reflexes = self._parts.reflexes
        decoded = tuple(reflexes.encoded.findings(text))
        findings = (*reflexes.template_tokens.findings(text), *reflexes.guard_addressed.findings(text), *decoded)
        hidden = "\n".join(finding.payload for finding in decoded if finding.payload)
        return findings, hidden

    def _user(
        self, conversation: Conversation, sanitation: dict[Locator, list[Finding]], site: Site, off: frozenset[str]
    ) -> tuple[ViewedText | None, tuple[Finding, ...]]:
        latest = conversation.latest_user
        if latest is None:
            return None, ()
        findings, decoded = self._memo.inbound(latest.text)
        findings = (
            *findings,
            *self._parts.vaccines.text_findings(Stage.INPUT, latest.text, site.site_id, site.profile.organs, off),
        )
        prior = sanitation.get(latest.locator, [])
        hidden = "\n".join([decoded, *(finding.payload for finding in prior if finding.payload)]).strip()
        raw = f"{latest.text}\n\n[decoded hidden content]\n{hidden}" if hidden else latest.text
        view = ViewedText(raw=raw, defanged=self._parts.reflexes.guard_addressed.defang(latest.text))
        return view, (*prior, *findings)

    def _items(
        self, conversation: Conversation, sanitation: dict[Locator, list[Finding]], site: Site, off: frozenset[str]
    ) -> tuple[DataItem, ...]:
        items: list[DataItem] = []
        for index, segment in enumerate(conversation.current_data):
            findings, decoded = self._memo.inbound(segment.text)
            findings = (
                *findings,
                *self._parts.vaccines.text_findings(Stage.DATA, segment.text, site.site_id, site.profile.organs, off),
            )
            prior = [] if segment.is_embedded else sanitation.get(segment.locator, [])
            hidden = "\n".join([decoded, *(finding.payload for finding in prior if finding.payload)]).strip()
            raw = f"{segment.text}\n\n[decoded hidden content]\n{hidden}" if hidden else segment.text
            digest = hashlib.sha256(f"{segment.origin}\n{raw}".encode()).hexdigest()
            items.append(
                DataItem(
                    item_id=f"item_{index + 1}",
                    segment=segment,
                    view=ViewedText(raw=raw, defanged=self._parts.reflexes.guard_addressed.defang(segment.text)),
                    findings=(*prior, *findings),
                    digest=digest,
                    cached=self._parts.item_cache.get(digest),
                )
            )
        return tuple(items)

    @staticmethod
    def _input_digest(conversation: Conversation) -> str | None:
        latest = conversation.latest_user
        if latest is None:
            return None
        return hashlib.sha256(f"{latest.turn}\n{latest.text}".encode()).hexdigest()

    def _needs_profile(self, prepared: PreparedCall) -> bool:
        return (
            not prepared.site.profiled
            and self._parts.settings.privacy.profiling == "jev"
            and bool(prepared.conversation.operator_text.strip())
            and prepared.profile.source != "pinned"
        )

    def _claims_profile(self, prepared: PreparedCall) -> bool:
        return self._needs_profile(prepared) and self._parts.sites.claim_profiling(prepared.site)

    @staticmethod
    def _reading(result: SensorReading | BaseException | None) -> SensorReading | None:
        return result if isinstance(result, SensorReading) else None

    def _item_readings(
        self, prepared: PreparedCall, plan: SensingPlan, results: SensorResults
    ) -> dict[str, SensorReading]:
        readings = {item.item_id: item.cached for item in prepared.data_items if item.cached is not None}
        for name, batch in plan.batches.items():
            result = self._reading(plan.result(results, name))
            if result is None:
                continue
            for item in batch:
                scoped = result.scoped(item.item_id)
                self._parts.item_cache.put(item.digest, scoped)
                readings[item.item_id] = scoped
        return readings

    def _update_profile(self, prepared: PreparedCall, reading: SensorReading | None, failed: bool) -> None:
        parts = self._parts
        site = prepared.site
        if site.profiled or failed:
            return
        if reading is None and not self._needs_profile(prepared):
            parts.sites.update(site, site.profile, profiled=True)
            return
        if reading is None:
            return
        inferred = parts.inferrer.inferred(site.profile, reading, prepared.conversation, prepared.site_settings)
        parts.sites.update(site, site.profile.merged(inferred), profiled=True)
        issues = parts.posture.assess(site.profile, prepared.conversation)
        _LOGGER.info(
            "immune: profiled %s as %s (organs: %s)%s",
            site.name,
            site.profile.archetype,
            ", ".join(sorted(site.profile.organs)) or "none",
            "".join(f"; posture: {issue.message}" for issue in issues),
        )

    def _neutralizations(self, prepared: PreparedCall, decisions: list[Decision]) -> list[TextEdit]:
        items = {item.item_id: item for item in prepared.data_items}
        latest = prepared.conversation.latest_user
        stripped: set[Locator] = set()
        placeholders: dict[str, TextEdit] = {}
        for decision in decisions:
            floor = decision.assessment.threat.floor
            subject = decision.assessment.subject
            item = items.get(subject or "")
            if decision.enforced_action is not Action.NEUTRALIZE:
                continue
            if floor is not None and floor.id == "F1":
                locator = item.segment.locator if item else latest.locator if latest else None
                if locator is not None and locator in prepared.sanitation:
                    stripped.add(locator)
            elif item is not None and item.item_id not in placeholders:
                placeholder = self._parts.spec.templates.data_placeholder.format(origin=item.segment.origin)
                placeholders[item.item_id] = TextEdit(item.segment.locator, placeholder, item.segment.span)
        return self._composed(prepared.sanitation, stripped, placeholders.values())

    @staticmethod
    def _composed(
        sanitation: Mapping[Locator, TextEdit], stripped: set[Locator], placeholders: Iterable[TextEdit]
    ) -> list[TextEdit]:
        """Placeholder spans point into the cleaned text, so a message they edit is cleaned along with them."""
        grouped: dict[Locator, list[TextEdit]] = defaultdict(list)
        for edit in placeholders:
            grouped[edit.locator].append(edit)
        edits: list[TextEdit] = []
        for locator, group in grouped.items():
            cleaned = sanitation.get(locator)
            if cleaned is None:
                edits.extend(group)
                continue
            edits.append(TextEdit(locator, JsonDocument.rewrite(cleaned.replacement, group)))
            stripped.discard(locator)
        edits.extend(sanitation[locator] for locator in sorted(stripped, key=repr))
        return edits

    def _output_findings(self, prepared: PreparedCall, text: str) -> list[Finding]:
        if not text:
            return []
        parts = self._parts
        reflexes = parts.reflexes
        conversation = prepared.conversation
        trusted = set(self._trusted_hosts(prepared))
        private = "\n".join([conversation.operator_text, *(segment.text for segment in conversation.data)])
        findings = list(reflexes.exfiltration.findings(text, trusted, private))
        if prepared.profile.rendered:
            findings.extend(reflexes.markup.findings(text))
        findings.extend(Finding("output.secret_leak", match.kind, match.span) for match in reflexes.secrets.find(text))
        if parts.canary is not None:
            findings.extend(parts.canary.findings(text))
        findings.extend(reflexes.prompt_copy.findings(text, conversation.operator_text))
        findings.extend(parts.organs.inspect_reply(conversation, Reply(text=text), prepared.profile))
        findings.extend(
            parts.vaccines.text_findings(
                Stage.OUTPUT, text, prepared.site.site_id, prepared.profile.organs, prepared.vaccines_off
            )
        )
        return findings

    def _trusted_hosts(self, prepared: PreparedCall) -> Iterable[str]:
        conversation = prepared.conversation
        for segment in conversation.segments:
            if segment.channel in (Channel.OPERATOR, Channel.USER):
                yield from LinkInspector.hosts(segment.text)
        for destination in prepared.session.trusted_destinations:
            if destination.domain:
                yield destination.domain
        if prepared.site_settings is not None:
            yield from prepared.site_settings.allowed_destinations

    def _call_findings(
        self, prepared: PreparedCall, call_id: str, call: ToolCall, parsed: ParsedReply, counted: CallCount
    ) -> list[Finding]:
        parts = self._parts
        session = prepared.session
        capabilities = prepared.profile.capabilities(call.name)
        confirmed = session.confirmations.redeem(call, prepared.latest_user_text) or parts.seal.confirmed(
            prepared.conversation, call
        )
        if confirmed:
            parsed.confirmed.add(call_id)
        findings: list[Finding] = []
        repeats = max(session.record_call(call), counted.repeats)
        total = max(session.calls_this_turn, counted.total)
        limits = parts.settings.limits
        if repeats > limits.max_identical_tool_calls or total > limits.max_tool_calls_per_turn:
            findings.append(Finding("tool.loop", f"{call.name} called {repeats} times this turn", subject=call_id))
        if not confirmed and (capabilities.egress or capabilities.writes_state):
            allowed = self._allowed_destinations(prepared, call.name)
            provenance = DestinationProvenance(session.trusted_destinations, allowed)
            destinations = parts.reflexes.destinations.from_arguments(call.arguments)
            untrusted = provenance.untrusted(destinations, session.data_destinations)
            findings.extend(
                Finding(
                    "tool.destination_provenance", f"{item.value} appeared only in untrusted content", subject=call_id
                )
                for item in sorted(untrusted, key=lambda destination: destination.value)
            )
        if capabilities.egress:
            findings.extend(self._egress_secrets(call, call_id))
        if not confirmed and session.taint is not Taint.CLEAN and capabilities.irreversible:
            findings.append(
                Finding("tool.tainted_irreversible", f"{call.name} after reading untrusted content", subject=call_id)
            )
        findings.extend(
            Finding(finding.threat, finding.evidence, finding.span, finding.payload, call_id)
            for finding in parts.organs.inspect_call(call, capabilities, prepared.profile)
        )
        findings.extend(
            Finding(finding.threat, finding.evidence, finding.span, finding.payload, call_id)
            for finding in parts.vaccines.call_findings(
                call,
                prepared.site.site_id,
                prepared.profile.organs,
                session.taint is not Taint.CLEAN,
                prepared.vaccines_off,
            )
        )
        return findings

    def _allowed_destinations(self, prepared: PreparedCall, tool: str) -> list[str]:
        settings = prepared.site_settings
        if settings is None:
            return []
        tool_settings = settings.tools.get(tool)
        return [*settings.allowed_destinations, *(tool_settings.allowed_destinations if tool_settings else ())]

    def _egress_secrets(self, call: ToolCall, call_id: str) -> Iterable[Finding]:
        reflexes = self._parts.reflexes
        payload = json.dumps(call.arguments, ensure_ascii=False, default=str)
        for match in reflexes.secrets.find(payload):
            yield Finding("tool.secret_egress", match.kind, subject=call_id)
        for personal in reflexes.personal.find(payload):
            if personal.kind in _SECRET_KINDS:
                yield Finding("tool.secret_egress", personal.kind, subject=call_id)
        if self._parts.canary is not None and any(True for _ in self._parts.canary.findings(payload)):
            yield Finding("tool.secret_egress", "deployment canary", subject=call_id)

    def _worth_screening(self, prepared: PreparedCall, call: ToolCall) -> bool:
        capabilities = prepared.profile.capabilities(call.name)
        return prepared.session.taint is not Taint.CLEAN or bool(capabilities.names)

    @staticmethod
    def _absorb_taints(session: SessionState, decisions: Iterable[Decision]) -> None:
        for decision in decisions:
            taint = decision.assessment.threat.taints
            if taint is not None:
                session.taint_with(taint)

    def _request_confirmations(
        self, prepared: PreparedCall, decisions: Iterable[Decision], parsed: ParsedReply
    ) -> None:
        if not prepared.profile.interactive:
            return
        for decision in decisions:
            call = parsed.calls.get(decision.assessment.subject or "")
            if call is not None and decision.hit.enforced and decision.hit.action in (Action.CONFIRM, Action.HOLD):
                prepared.session.confirmations.request(call)

    def _synthesized(
        self,
        prepared: PreparedCall,
        decision: InboundDecision,
        inbound: InboundResult,
        forced: Action | None = None,
        reason: str | None = None,
    ) -> Outcome:
        response_id = f"{prepared.codec.response_prefix}{prepared.trace_id}"
        body = prepared.codec.synthesize_response(
            prepared.conversation, decision.block_text or "", response_id, decision.refusal
        )
        verdict = self._verdict(
            prepared,
            decision.decisions,
            inbound,
            SensorReading.empty(),
            {},
            forced=forced or (Action.END_SESSION if prepared.session.locked and not decision.decisions else None),
            reason=reason,
        )
        self._record(prepared, verdict, decision.decisions, inbound, None, response_id)
        return Outcome(
            verdict=verdict,
            content=self._encode(prepared, body),
            content_type="text/event-stream" if prepared.streaming else "application/json",
            replaced=True,
        )

    def _encode(self, prepared: PreparedCall, body: dict[str, Any]) -> bytes:
        if prepared.streaming:
            return ServerSentEvents.serialize(prepared.codec.stream_from_body(body))
        return json.dumps(body).encode("utf-8")

    def _verdict(
        self,
        prepared: PreparedCall,
        decisions: Sequence[Decision],
        inbound: InboundResult,
        outbound: SensorReading,
        echo: Mapping[str, Mapping[str, float]],
        forced: Action | None = None,
        reason: str | None = None,
    ) -> Verdict:
        readings = [inbound.input_reading, *inbound.item_readings.values(), outbound]
        active = [reading for reading in readings if reading.calls or not reading.is_empty]  # cached answers count
        merged = active[0].merged(*active[1:]) if active else None
        sensor = (
            SensorInfo(
                name=merged.source,
                model=merged.model,
                latency_ms=round(merged.latency_ms, 1),
                input_tokens=merged.input_tokens,
                calls=merged.calls,
            )
            if merged
            else SensorInfo.deterministic_only()
        )
        action = forced or applied(decisions)
        would_action = forced or would(decisions)
        return Verdict(
            trace_id=prepared.trace_id,
            site=prepared.site.site_id,
            session_id=self._session_id(prepared),
            action=action,
            would_action=would_action,
            hits=tuple(self._disclosed(decision.hit) for decision in decisions),
            echo=dict(echo),
            taint=prepared.session.taint,
            session_risk=round(prepared.session.risk.probability, 4),
            sensor=sensor,
            config_hash=self._parts.settings.digest,
            spec_version=self._parts.spec.version,
            explanation=reason or self._explain(action, would_action, decisions),
        )

    def _disclosed(self, hit: Hit) -> Hit:
        if self._parts.settings.privacy.raw_evidence or not hit.evidence:
            return hit
        mask = self._parts.reflexes.redactor.mask
        return replace(hit, evidence=tuple(mask(item) for item in hit.evidence))

    @staticmethod
    def _explain(action: Action, would_action: Action, decisions: Sequence[Decision]) -> str:
        if not decisions:
            return "no threats detected" if action is Action.ALLOW else f"{action.value}: session is closed"
        enforced = sorted({d.hit.threat for d in decisions if d.hit.enforced})
        observed = sorted({d.hit.threat for d in decisions if not d.hit.enforced})
        parts = []
        if enforced:
            parts.append(f"{action.value}: {', '.join(enforced)}")
        if observed:
            parts.append(f"observed (would {would_action.value}): {', '.join(observed)}")
        return "; ".join(parts)

    def _record(
        self,
        prepared: PreparedCall,
        verdict: Verdict,
        decisions: Sequence[Decision],
        inbound: InboundResult,
        parsed: ParsedReply | None,
        response_id: str | None,
    ) -> None:
        parts = self._parts
        organs, off = prepared.profile.organs, prepared.vaccines_off
        evaluated: list[str] = []
        if not inbound.input_reading.is_empty or prepared.user_findings:
            evaluated.extend(parts.assessor.evaluated_threats(Stage.INPUT, organs, off))
        if prepared.data_items:
            evaluated.extend(parts.assessor.evaluated_threats(Stage.DATA, organs, off))
        if parsed is not None and parsed.reply.text:
            evaluated.extend(parts.assessor.evaluated_threats(Stage.OUTPUT, organs, off))
        if parsed is not None and parsed.calls:
            evaluated.extend(parts.assessor.evaluated_threats(Stage.TOOL, organs, off))
        if not prepared.identity.anonymous:
            parts.sessions.save(prepared.session)
        parts.ledger.observe(prepared.site.site_id, evaluated, {decision.hit.threat for decision in decisions})
        parts.recorder.record(verdict)
        parts.index.add(verdict, response_id)
        parts.resolver.remember(response_id, prepared.identity)
        if parts.observers.wants_details():
            parts.observers.summarized(self._summary(prepared, parsed))
        parts.observers.finish(verdict)
        if response_id and prepared.codec.server_side_history and parsed is not None:
            parts.transcripts.save(response_id, prepared.conversation, parsed.reply.text)
