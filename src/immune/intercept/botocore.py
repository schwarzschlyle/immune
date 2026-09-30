from __future__ import annotations

import json
import logging
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from immune.codecs import CODECS
from immune.codecs.base import RequestContext
from immune.core.calls import InboundDecision, InboundResult, Outcome, PreparedCall
from immune.core.sse import ServerSentEvent, ServerSentEvents
from immune.core.streaming import Holdback, ProgressiveScreen
from immune.errors import Blocked
from immune.intercept.blocking import BlockedErrors
from immune.intercept.flow import InterceptionFlow
from immune.intercept.hooks import PostImportHooks
from immune.telemetry.verdicts import VerdictIndex
from immune.types import Mode

_LOGGER = logging.getLogger("immune")
_SERVICE = "bedrock-runtime"
_OPERATIONS = ("Converse", "ConverseStream")
_PENDING = "immune"
_CODEC = CODECS["bedrock_converse"]


@dataclass(frozen=True, slots=True)
class _Pending:
    prepared: PreparedCall
    inbound: InboundResult
    decision: InboundDecision


class BedrockScreen:
    def __init__(self, flow: InterceptionFlow) -> None:
        self._flow = flow
        self._pipeline = flow.pipeline

    def before(self, params: dict[str, Any], context: dict[str, Any]) -> tuple[Any, dict[str, Any]] | None:
        if self._pipeline.parts.settings.mode is Mode.OFF:
            return None
        try:
            prepared = self._prepare(params)
        except Exception:
            _LOGGER.exception("immune: failed to prepare a Bedrock call; passing it through unscreened")
            return None
        if prepared is None:
            return None
        try:
            return self._screen_inbound(prepared, params, context)
        except Blocked:
            raise
        except Exception:
            _LOGGER.exception("immune: Bedrock inbound screening failed; passing the call through")
            return None

    def after(self, http_response: Any, parsed: dict[str, Any], context: dict[str, Any]) -> None:
        pending = context.pop(_PENDING, None)
        if not isinstance(pending, _Pending):
            return
        prepared = pending.prepared
        if http_response.status_code >= 300:
            reason = f"passed through: Bedrock returned HTTP {http_response.status_code}"
            self._pipeline.degraded_verdict(prepared, reason)
            return
        if prepared.streaming and "stream" in parsed:
            parsed["stream"] = ScreenedEventStream(parsed["stream"], self, pending)
            return
        try:
            outcome = self._screen_reply(
                pending, {key: value for key, value in parsed.items() if key != "ResponseMetadata"}
            )
        except Exception:
            _LOGGER.exception("immune: failed while screening a Bedrock reply")
            self._pipeline.degraded_verdict(prepared)
            return
        self._raise_if_blocked(outcome)
        if outcome.content is not None:
            metadata = parsed.get("ResponseMetadata")
            parsed.clear()
            parsed.update(json.loads(outcome.content))
            if metadata is not None:
                parsed["ResponseMetadata"] = metadata
        VerdictIndex.remember_last(outcome.verdict)

    def finish_stream(self, pending: _Pending, screen: ProgressiveScreen) -> list[ServerSentEvent]:
        pipeline = self._pipeline
        parsed = pipeline.parse(pending.prepared, screen.unscreened())
        outbound, _ = pipeline.collect_outbound(
            pending.prepared, self._flow.dispatch.collect(pipeline.plan_outbound(pending.prepared, parsed))
        )
        verdict, tail = pipeline.finalize_stream(
            pending.prepared, pending.inbound, pending.decision, parsed, outbound, screen
        )
        VerdictIndex.remember_last(verdict)
        return tail

    def finish_buffered(self, pending: _Pending, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pipeline, prepared = self._pipeline, pending.prepared
        try:
            parsed = pipeline.parse(prepared, ServerSentEvents.serialize(ServerSentEvent.of(event) for event in events))
            results = self._flow.dispatch.collect(pipeline.plan_outbound(prepared, parsed))
            outbound, _ = pipeline.collect_outbound(prepared, results)
            outcome = pipeline.finalize(prepared, pending.inbound, pending.decision, parsed, outbound)
        except Exception:
            _LOGGER.exception("immune: failed while screening a Bedrock stream; passing it through")
            return events
        self._raise_if_blocked(outcome)
        VerdictIndex.remember_last(outcome.verdict)
        if outcome.content is None:
            return events
        return [event.json() for event in ServerSentEvents.parse(outcome.content)]

    def progressive(self, pending: _Pending) -> ProgressiveScreen | None:
        prepared = pending.prepared
        if not self._pipeline.progressive(prepared):
            return None
        return ProgressiveScreen(_CODEC, Holdback(lambda text: self._pipeline.stream_hold(prepared, text)))

    def _prepare(self, params: Mapping[str, Any]) -> PreparedCall | None:
        body = json.loads(params.get("body") or b"{}")
        context = RequestContext(path=str(params.get("url_path", "")), body=body)
        if not isinstance(body, dict) or not _CODEC.accepts(context):
            return None
        prepared = self._pipeline.prepare(_CODEC, context)
        return prepared if self._pipeline.screens(prepared) else None

    def _screen_inbound(
        self, prepared: PreparedCall, params: dict[str, Any], context: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]] | None:
        pipeline = self._pipeline
        immediate = pipeline.immediate(prepared)
        if immediate is not None:
            return self._synthesized(prepared, immediate)
        plan = pipeline.plan_inbound(prepared)
        inbound = pipeline.collect_inbound(prepared, plan, self._flow.dispatch.collect(plan))
        decision = pipeline.gate(prepared, inbound)
        if decision.blocks:
            return self._synthesized(prepared, pipeline.blocked(prepared, inbound, decision))
        body = pipeline.neutralized_body(prepared, decision) if decision.rewrites else pipeline.request_body(prepared)
        if body is not None:
            params["body"] = body
        context[_PENDING] = _Pending(prepared, inbound, decision)
        return None

    def _screen_reply(self, pending: _Pending, body: Mapping[str, Any]) -> Outcome:
        pipeline, prepared = self._pipeline, pending.prepared
        parsed = pipeline.parse(prepared, json.dumps(body, default=str).encode("utf-8"))
        results = self._flow.dispatch.collect(pipeline.plan_outbound(prepared, parsed))
        outbound, _ = pipeline.collect_outbound(prepared, results)
        return pipeline.finalize(prepared, pending.inbound, pending.decision, parsed, outbound)

    def _synthesized(self, prepared: PreparedCall, outcome: Outcome) -> tuple[Any, dict[str, Any]]:
        self._raise_if_blocked(outcome)
        VerdictIndex.remember_last(outcome.verdict)
        assert outcome.content is not None
        metadata = {"HTTPStatusCode": 200, "HTTPHeaders": {"x-immune-trace": prepared.trace_id}, "RetryAttempts": 0}
        if prepared.streaming:
            events = [event.json() for event in ServerSentEvents.parse(outcome.content)]
            parsed: dict[str, Any] = {"stream": ListedEventStream(events), "ResponseMetadata": metadata}
        else:
            parsed = {**json.loads(outcome.content), "ResponseMetadata": metadata}
        return SyntheticResponse(200), parsed

    def _raise_if_blocked(self, outcome: Outcome) -> None:
        if self._flow.raise_on_block and outcome.replaced:
            BlockedErrors.raise_for(_CODEC.name, outcome.verdict)


class SyntheticResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.headers: dict[str, str] = {}
        self.content = b""


class ListedEventStream:
    def __init__(self, events: Iterable[dict[str, Any]]) -> None:
        self._events = list(events)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._events)

    def close(self) -> None:
        return None


class ScreenedEventStream:
    def __init__(self, inner: Any, screen: BedrockScreen, pending: _Pending) -> None:
        self._inner = inner
        self._screen = screen
        self._pending = pending

    def __iter__(self) -> Iterator[dict[str, Any]]:
        progressive = self._screen.progressive(self._pending)
        if progressive is None:
            yield from self._buffered()
            return
        screening = True
        for event in self._inner:
            if screening:
                try:
                    yield from self._dicts(progressive.feed_events([ServerSentEvent.of(event)]))
                    continue
                except Exception:
                    screening = False
                    _LOGGER.exception("immune: failed while screening a Bedrock stream; passing the rest through")
                    yield from self._dicts(progressive.fallback_events())
                    continue
            yield event
        if screening:
            try:
                yield from self._dicts(self._screen.finish_stream(self._pending, progressive))
            except Exception:
                _LOGGER.exception("immune: failed while finishing a Bedrock stream")
                yield from self._dicts(progressive.fallback_events())

    def close(self) -> None:
        self._inner.close()

    def _buffered(self) -> Iterator[dict[str, Any]]:
        events = list(self._inner)
        yield from self._screen.finish_buffered(self._pending, events)

    @staticmethod
    def _dicts(events: Iterable[ServerSentEvent]) -> Iterator[dict[str, Any]]:
        for event in events:
            payload = event.json()
            if isinstance(payload, dict):
                yield payload


class BotocoreHooks:
    def __init__(self, flow_source: Callable[[], InterceptionFlow | None]) -> None:
        self._flow_source = flow_source
        self._attached: weakref.WeakSet[Any] = weakref.WeakSet()
        self._original: Callable[..., Any] | None = None
        self._owner: type[Any] | None = None

    def install(self, hooks: PostImportHooks) -> None:
        hooks.when_imported("botocore.client", self._patch)

    def uninstall(self) -> None:
        if self._owner is not None and self._original is not None:
            self._owner.create_client = self._original
        self._owner = self._original = None

    def attach(self, client: Any) -> bool:
        meta = getattr(client, "meta", None)
        if meta is None or meta.service_model.service_name != _SERVICE:
            return False
        if client in self._attached:
            return True
        for operation in _OPERATIONS:
            meta.events.register(f"before-call.{_SERVICE}.{operation}", self._before)
            meta.events.register(f"after-call.{_SERVICE}.{operation}", self._after)
        self._attached.add(client)
        return True

    def _patch(self, module: ModuleType) -> None:
        owner = module.ClientCreator
        if self._owner is not None:
            return
        original = owner.create_client
        hooks = self

        def create_client(creator: Any, *args: Any, **kwargs: Any) -> Any:
            client = original(creator, *args, **kwargs)
            hooks.attach(client)
            return client

        self._owner, self._original = owner, original
        owner.create_client = create_client

    def _before(self, params: dict[str, Any], context: dict[str, Any], **_: Any) -> Any:
        flow = self._flow_source()
        return BedrockScreen(flow).before(params, context) if flow is not None else None

    def _after(self, http_response: Any, parsed: dict[str, Any], context: dict[str, Any], **_: Any) -> None:
        flow = self._flow_source()
        if flow is not None:
            BedrockScreen(flow).after(http_response, parsed, context)
