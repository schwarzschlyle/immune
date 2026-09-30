from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

from immune.core.calls import InboundDecision, InboundResult, Outcome, PreparedCall
from immune.core.pipeline import CallPipeline
from immune.core.sse import ServerSentEvents
from immune.core.streaming import Holdback, ProgressiveScreen
from immune.errors import Blocked
from immune.intercept.blocking import BlockedErrors
from immune.intercept.router import EndpointRouter, RoutedRequest
from immune.intercept.sensing import SensingDispatch
from immune.intercept.streams import ScreenedStreams
from immune.intercept.upstream import AsyncUpstream, SyncUpstream, Upstream
from immune.sensing.signals import SensorReading
from immune.telemetry.verdicts import VerdictIndex
from immune.types import Mode, Verdict

_LOGGER = logging.getLogger("immune")
_SCREENING: contextvars.ContextVar[bool] = contextvars.ContextVar("immune_screening", default=False)
_ERROR_STATUS = 400
Screened = tuple[Any, Verdict | None]


class InterceptionFlow:
    def __init__(
        self,
        pipeline: CallPipeline,
        router: EndpointRouter,
        dispatch: SensingDispatch,
        fail_open: bool = True,
        raise_on_block: bool = False,
    ) -> None:
        self._pipeline = pipeline
        self._router = router
        self._dispatch = dispatch
        self._fail_open = fail_open
        self._raise_on_block = raise_on_block

    @property
    def pipeline(self) -> CallPipeline:
        return self._pipeline

    def handle_sync(self, request: Any, send: Callable[[Any], Any], response_type: type[Any]) -> Any:
        prepared = self._prepare(request)
        if prepared is None:
            return send(request)
        with self._screening():
            response, verdict = SyncCall(self, prepared, request, SyncUpstream(send, response_type)).run()
        if verdict is not None:
            VerdictIndex.remember_last(verdict)
        return response

    async def handle_async(self, request: Any, send: Callable[[Any], Awaitable[Any]], response_type: type[Any]) -> Any:
        prepared = self._prepare(request)
        if prepared is None:
            return await send(request)
        with self._screening():
            response, verdict = await AsyncCall(self, prepared, request, AsyncUpstream(send, response_type)).run()
        if verdict is not None:
            VerdictIndex.remember_last(verdict)
        return response

    @property
    def dispatch(self) -> SensingDispatch:
        return self._dispatch

    @property
    def fail_open(self) -> bool:
        return self._fail_open

    @property
    def raise_on_block(self) -> bool:
        return self._raise_on_block

    def reconfigure(self, router: EndpointRouter, fail_open: bool, raise_on_block: bool) -> None:
        self._router = router
        self._fail_open = fail_open
        self._raise_on_block = raise_on_block

    def _prepare(self, request: Any) -> PreparedCall | None:
        if _SCREENING.get():
            return None
        if self._pipeline.parts.settings.mode is Mode.OFF:
            VerdictIndex.forget_last()
            return None
        try:
            routed = self._route(request)
            prepared = self._pipeline.prepare(routed.codec, routed.context) if routed else None
            if prepared is not None and self._pipeline.screens(prepared):
                return prepared
            if prepared is not None:
                VerdictIndex.forget_last()
            return None
        except Exception:
            _LOGGER.exception("immune: failed to prepare call; passing through unscreened")
            return None

    def _route(self, request: Any) -> RoutedRequest | None:
        try:
            content = bytes(request.content)
        except Exception:
            return None
        return self._router.route(request.method, request.url.host, request.url.path, content, request.headers)

    @staticmethod
    @contextlib.contextmanager
    def _screening() -> Iterator[None]:
        token = _SCREENING.set(True)
        try:
            yield
        finally:
            _SCREENING.reset(token)


class _Call:
    def __init__(self, flow: InterceptionFlow, prepared: PreparedCall, request: Any, upstream: Upstream) -> None:
        self._flow = flow
        self._pipeline = flow.pipeline
        self._dispatch = flow.dispatch
        self._prepared = prepared
        self._request = request
        self._upstream = upstream
        self._response: Any = None
        self._raw: bytes | None = None
        self._upstream_failed = False

    def _outgoing(self) -> Any:
        body = self._pipeline.request_body(self._prepared)
        return Upstream.rebuild(self._request, body) if body is not None else self._request

    def _neutralized(self, decision: InboundDecision) -> Any:
        return Upstream.rebuild(self._request, self._pipeline.neutralized_body(self._prepared, decision))

    def _raise_if_blocked(self, outcome: Outcome) -> None:
        if self._flow.raise_on_block and outcome.replaced:
            BlockedErrors.raise_for(self._prepared.codec.name, outcome.verdict)

    def _synthesized(self, outcome: Outcome) -> Screened:
        assert outcome.content is not None
        self._raise_if_blocked(outcome)
        response = self._upstream.respond(
            self._request, None, 200, outcome.content, outcome.content_type, outcome.verdict.trace_id
        )
        return response, outcome.verdict

    def _replied(self, response: Any, raw: bytes, outcome: Outcome) -> Screened:
        self._raise_if_blocked(outcome)
        content_type = outcome.content_type if outcome.content else response.headers.get("content-type", "")
        replied = self._upstream.respond(
            self._request,
            response,
            response.status_code,
            outcome.content or raw,
            content_type,
            outcome.verdict.trace_id,
        )
        return replied, outcome.verdict

    @property
    def _on_error(self) -> str:
        return "passing the call through" if self._flow.fail_open else "refusing the call (on_internal_error: block)"

    def _refused_or_none(self, inbound: InboundResult | None) -> Screened | None:
        if self._flow.fail_open:
            return None
        empty = InboundResult(input_reading=SensorReading.empty(), item_readings={})
        return self._synthesized(self._pipeline.refused(self._prepared, inbound or empty))

    def _passed_through(self, response: Any) -> Screened:
        verdict = self._pipeline.degraded_verdict(self._prepared)
        if self._raw is not None:
            content_type = response.headers.get("content-type", "application/json")
            replied = self._upstream.respond(
                self._request, response, response.status_code, self._raw, content_type, verdict.trace_id
            )
            return replied, verdict
        return self._upstream.tagged(response, verdict.trace_id), verdict

    def _screen(self) -> ProgressiveScreen:
        prepared = self._prepared
        return ProgressiveScreen(prepared.codec, Holdback(lambda text: self._pipeline.stream_hold(prepared, text)))

    def _streamed(self, response: Any, stream: Any) -> Screened:
        replied = self._upstream.respond_stream(
            self._request, response, stream, "text/event-stream", self._prepared.trace_id
        )
        return replied, None

    @staticmethod
    def _failed_stream(screen: ProgressiveScreen) -> bytes:
        _LOGGER.exception("immune: failed while screening a streamed reply; passing the rest through")
        return screen.fallback()

    @staticmethod
    def _degraded_inbound() -> InboundResult:
        _LOGGER.exception("immune: inbound screening failed; continuing without Jev")
        return InboundResult(
            input_reading=SensorReading.empty(), item_readings={}, degraded=("inbound",), candidates_failed=True
        )


class SyncCall(_Call):
    _upstream: SyncUpstream

    def run(self) -> Screened:
        try:
            return self._run()
        except Blocked:
            raise
        except Exception:
            if self._upstream_failed:
                raise
            _LOGGER.exception("immune: internal error; %s", self._on_error)
            refused = self._refused_or_none(None)
            if refused is not None:
                return refused
            return self._passed_through(self._response or self._send(self._request))

    def _send(self, request: Any) -> Any:
        try:
            self._response = self._upstream.send(request)
        except BaseException:
            self._upstream_failed = True
            raise
        return self._response

    def _run(self) -> Screened:
        pipeline = self._pipeline
        prepared = self._prepared
        immediate = pipeline.immediate(prepared)
        if immediate is not None:
            return self._synthesized(immediate)
        plan = pipeline.plan_inbound(prepared)
        pending = self._dispatch.start(plan)
        response = self._send(self._outgoing())
        try:
            inbound = pipeline.collect_inbound(prepared, plan, self._dispatch.wait(plan, pending))
        except Exception:
            inbound = self._degraded_inbound()
        decision = pipeline.gate(prepared, inbound)
        if decision.blocks:
            self._upstream.close(response)
            return self._synthesized(pipeline.blocked(prepared, inbound, decision))
        if decision.rewrites:
            self._upstream.close(response)
            self._response = None
            response = self._send(self._neutralized(decision))
        if response.status_code >= _ERROR_STATUS:
            reason = f"passed through: the provider returned HTTP {response.status_code}"
            return self._upstream.tagged(response, prepared.trace_id), pipeline.degraded_verdict(prepared, reason)
        if pipeline.progressive(prepared):
            return self._progressive(response, inbound, decision)
        raw = self._raw = self._upstream.read(response)
        try:
            parsed = pipeline.parse(prepared, raw)
            results = self._dispatch.collect(pipeline.plan_outbound(prepared, parsed))
            outbound, _ = pipeline.collect_outbound(prepared, results)
            outcome = pipeline.finalize(prepared, inbound, decision, parsed, outbound)
        except Exception:
            _LOGGER.exception("immune: failed while screening the reply; %s", self._on_error)
            return self._refused_or_none(inbound) or self._passed_through(response)
        return self._replied(response, raw, outcome)

    def _progressive(self, response: Any, inbound: InboundResult, decision: InboundDecision) -> Screened:
        screen = self._screen()
        pipeline, prepared = self._pipeline, self._prepared

        def finish() -> bytes:
            parsed = pipeline.parse(prepared, screen.unscreened())
            results = self._dispatch.collect(pipeline.plan_outbound(prepared, parsed))
            outbound, _ = pipeline.collect_outbound(prepared, results)
            verdict, tail = pipeline.finalize_stream(prepared, inbound, decision, parsed, outbound, screen)
            VerdictIndex.remember_last(verdict)
            return ServerSentEvents.serialize(tail)

        def chunks() -> Iterator[bytes]:
            screening = True
            for chunk in response.iter_bytes():
                if screening:
                    try:
                        yield screen.feed(chunk)
                        continue
                    except Exception:
                        screening = False
                        yield self._failed_stream(screen)
                        continue
                yield chunk
            if screening:
                try:
                    yield screen.drain() + finish()
                except Exception:
                    yield self._failed_stream(screen)

        return self._streamed(response, ScreenedStreams.sync(self._upstream.response_type, chunks, response.close))


class AsyncCall(_Call):
    _upstream: AsyncUpstream

    async def run(self) -> Screened:
        try:
            return await self._run()
        except Blocked:
            raise
        except Exception:
            if self._upstream_failed:
                raise
            _LOGGER.exception("immune: internal error; %s", self._on_error)
            refused = self._refused_or_none(None)
            if refused is not None:
                return refused
            return self._passed_through(self._response or await self._send(self._request))

    async def _send(self, request: Any) -> Any:
        try:
            self._response = await self._upstream.send(request)
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._upstream_failed = True
            raise
        return self._response

    async def _run(self) -> Screened:
        pipeline = self._pipeline
        prepared = self._prepared
        immediate = pipeline.immediate(prepared)
        if immediate is not None:
            return self._synthesized(immediate)
        plan = pipeline.plan_inbound(prepared)
        sensing = asyncio.ensure_future(self._dispatch.gather(plan))
        pending = asyncio.ensure_future(self._send(self._outgoing()))
        try:
            inbound = pipeline.collect_inbound(prepared, plan, await sensing)
        except Exception:
            inbound = self._degraded_inbound()
        decision = pipeline.gate(prepared, inbound)
        if decision.blocks:
            self._upstream.discard(pending)
            return self._synthesized(pipeline.blocked(prepared, inbound, decision))
        if decision.rewrites:
            self._upstream.discard(pending)
            self._response = None
            pending = asyncio.ensure_future(self._send(self._neutralized(decision)))
        response = await pending
        if response.status_code >= _ERROR_STATUS:
            reason = f"passed through: the provider returned HTTP {response.status_code}"
            return self._upstream.tagged(response, prepared.trace_id), pipeline.degraded_verdict(prepared, reason)
        if pipeline.progressive(prepared):
            return self._progressive(response, inbound, decision)
        raw = self._raw = await self._upstream.read(response)
        try:
            parsed = pipeline.parse(prepared, raw)
            results = await self._dispatch.gather(pipeline.plan_outbound(prepared, parsed))
            outbound, _ = pipeline.collect_outbound(prepared, results)
            outcome = pipeline.finalize(prepared, inbound, decision, parsed, outbound)
        except Exception:
            _LOGGER.exception("immune: failed while screening the reply; %s", self._on_error)
            return self._refused_or_none(inbound) or self._passed_through(response)
        return self._replied(response, raw, outcome)

    def _progressive(self, response: Any, inbound: InboundResult, decision: InboundDecision) -> Screened:
        screen = self._screen()
        pipeline, prepared = self._pipeline, self._prepared

        async def finish() -> bytes:
            parsed = pipeline.parse(prepared, screen.unscreened())
            results = await self._dispatch.gather(pipeline.plan_outbound(prepared, parsed))
            outbound, _ = pipeline.collect_outbound(prepared, results)
            verdict, tail = pipeline.finalize_stream(prepared, inbound, decision, parsed, outbound, screen)
            VerdictIndex.remember_last(verdict)
            return ServerSentEvents.serialize(tail)

        async def chunks() -> AsyncIterator[bytes]:
            screening = True
            async for chunk in response.aiter_bytes():
                if screening:
                    try:
                        yield screen.feed(chunk)
                        continue
                    except Exception:
                        screening = False
                        yield self._failed_stream(screen)
                        continue
                yield chunk
            if screening:
                try:
                    yield screen.drain() + await finish()
                except Exception:
                    yield self._failed_stream(screen)

        stream = ScreenedStreams.asynchronous(self._upstream.response_type, chunks, response.aclose)
        return self._streamed(response, stream)
