from __future__ import annotations

import importlib
import threading
from collections import OrderedDict
from typing import Any

from immune.telemetry.observers import CallObserver, CallStart
from immune.types import Verdict

_TRACER = "immune"
_PROVIDERS = {"openai": "openai", "anthropic": "anthropic", "gemini": "gcp.gemini"}
_CONTEXTS = 10_000
SEMCONV_VERSION = "1.37.0"


class OpenTelemetryObserver(CallObserver):
    def __init__(self, trace: Any, metrics: Any, context: Any, version: str) -> None:
        self._context = context
        self._tracer = trace.get_tracer(
            _TRACER, version, schema_url=f"https://opentelemetry.io/schemas/{SEMCONV_VERSION}"
        )
        meter = metrics.get_meter(_TRACER, version)
        self._calls = meter.create_counter("immune.calls", unit="{call}", description="LLM calls screened by immune")
        self._interventions = meter.create_counter(
            "immune.interventions", unit="{hit}", description="Threats that changed a call"
        )
        self._observed = meter.create_counter("immune.observations", unit="{hit}", description="Threats observed only")
        self._latency = meter.create_histogram(
            "immune.sensor.latency", unit="ms", description="Latency of Jev sensing per call"
        )
        self._tokens = meter.create_counter("immune.sensor.tokens", unit="{token}", description="Jev input tokens")
        self._parents: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    @classmethod
    def create(cls, version: str) -> OpenTelemetryObserver | None:
        try:
            trace = importlib.import_module("opentelemetry.trace")
            metrics = importlib.import_module("opentelemetry.metrics")
            context = importlib.import_module("opentelemetry.context")
        except ImportError:
            return None
        return cls(trace, metrics, context, version)

    def started(self, call: CallStart) -> None:
        with self._lock:
            self._parents[call.trace_id] = self._context.get_current()
            while len(self._parents) > _CONTEXTS:
                self._parents.popitem(last=False)

    def finished(self, call: CallStart, verdict: Verdict) -> None:
        with self._lock:
            parent = self._parents.pop(call.trace_id, None)
        attributes = self._attributes(call, verdict)
        span = self._tracer.start_span(
            "immune.screen", context=parent, start_time=call.started_ns, attributes=attributes
        )
        for hit in verdict.hits:
            span.add_event(
                "immune.hit",
                attributes={
                    "immune.threat": hit.threat,
                    "immune.action": hit.action.value,
                    "immune.enforced": hit.enforced,
                    "immune.probability": hit.probability,
                },
            )
        span.end()
        site = {"immune.site": verdict.site, "gen_ai.provider.name": _provider(call.provider)}
        self._calls.add(1, site)
        for hit in verdict.hits:
            labels = {**site, "immune.threat": hit.threat, "immune.action": hit.action.value}
            (self._interventions if hit.enforced else self._observed).add(1, labels)
        if verdict.sensor.calls:
            self._latency.record(verdict.sensor.latency_ms, site)
            self._tokens.add(verdict.sensor.input_tokens, site)

    @staticmethod
    def _attributes(call: CallStart, verdict: Verdict) -> dict[str, Any]:
        return {
            "gen_ai.provider.name": _provider(call.provider),
            "gen_ai.request.model": call.model,
            "immune.trace_id": verdict.trace_id,
            "immune.site": verdict.site,
            "immune.action": verdict.action.value,
            "immune.would_action": verdict.would_action.value,
            "immune.threats": sorted(verdict.threats()),
            "immune.taint": verdict.taint.value,
            "immune.session_risk": verdict.session_risk,
            "immune.sensor": verdict.sensor.name,
            "immune.sensor.calls": verdict.sensor.calls,
            "immune.spec_version": verdict.spec_version,
        }


def _provider(codec: str) -> str:
    return _PROVIDERS.get(codec.split("_", maxsplit=1)[0], codec)
