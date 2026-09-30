from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import httpx2
import openai
import pytest
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import immune
from immune.errors import ConfigError
from immune.telemetry.alerts import Alert, SpikeDetector, SpikePolicy
from immune.telemetry.observers import CallStart
from immune.testing import FakeProvider, FakeReply, MockSensor
from immune.types import Action, Hit, Stage, Verdict
from tests.integration.test_floor import EXFIL

CHART = FakeReply(text=f"Chart {EXFIL}")
MESSAGES = [{"role": "user", "content": "chart"}]
SPANS = InMemorySpanExporter()
METRICS = InMemoryMetricReader()


@pytest.fixture(scope="module", autouse=True)
def providers() -> None:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(SPANS))
    trace.set_tracer_provider(tracer_provider)
    metrics.set_meter_provider(MeterProvider(metric_readers=[METRICS]))


@pytest.fixture
def client(tmp_path: Path) -> Iterator[openai.OpenAI]:
    immune.init(sensor=MockSensor(), state_dir=tmp_path)
    provider = FakeProvider(CHART)
    yield openai.OpenAI(
        api_key="sk-test", http_client=httpx2.Client(transport=provider.transport(httpx2)), max_retries=0
    )
    immune.shutdown()


def chart(client: openai.OpenAI) -> str:
    return client.chat.completions.create(model="m", messages=MESSAGES).choices[0].message.content or ""


class TestOpenTelemetry:
    def test_screening_spans_nest_under_the_callers_span(self, client: openai.OpenAI) -> None:
        SPANS.clear()
        with trace.get_tracer("app").start_as_current_span("llm call") as parent:
            chart(client)
        spans = {span.name: span for span in SPANS.get_finished_spans()}
        screen = spans["immune.screen"]
        assert screen.parent is not None
        assert screen.parent.span_id == parent.get_span_context().span_id
        assert screen.attributes["immune.action"] == "rewrite"
        assert "output.exfil_link" in screen.attributes["immune.threats"]
        assert screen.attributes["gen_ai.provider.name"] == "openai"
        assert [event.name for event in screen.events] == ["immune.hit"]

    def test_metrics_count_calls_and_interventions(self, client: openai.OpenAI) -> None:
        chart(client)
        names = {
            metric.name
            for resource in METRICS.get_metrics_data().resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        assert {"immune.calls", "immune.interventions"} <= names


class TestCallbacksAndAlerts:
    def test_verdict_callbacks_run_off_the_request_thread(self, client: openai.OpenAI) -> None:
        seen: list[tuple[str, str]] = []
        remove = immune.on_verdict(lambda verdict: seen.append((verdict.trace_id, threading.current_thread().name)))
        chart(client)
        runtime = immune.runtime()
        assert runtime is not None
        runtime.callbacks.flush()
        remove()
        assert seen
        assert seen[0][1] == "immune-callbacks"

    def test_failing_callbacks_do_not_break_calls(self, client: openai.OpenAI) -> None:
        def explode(_: Verdict) -> None:
            raise RuntimeError("bad callback")

        immune.on_verdict(explode)
        assert chart(client) == "Chart [link removed]"

    def test_a_replayed_campaign_raises_exactly_one_alert(self) -> None:
        clock = [0.0]
        detector = SpikeDetector(SpikePolicy(min_baseline_calls=50), clock=lambda: clock[0])
        raised: list[Alert] = []
        detector.on_alert(raised.append)
        calm = self._verdict(threats=())
        attack = self._verdict(threats=("input.override",))
        for _ in range(500):
            clock[0] += 60
            detector.finished(CallStart(calm.trace_id, "openai_chat", "m"), calm)
        for _ in range(60):
            clock[0] += 5
            detector.finished(CallStart(attack.trace_id, "openai_chat", "m"), attack)
        assert [(alert.site, alert.threat) for alert in raised] == [("support", "input.override")]

    @staticmethod
    def _verdict(threats: tuple[str, ...]) -> Verdict:
        hits = tuple(Hit(threat, "U1", Stage.INPUT, 0.9, Action.REDIRECT, enforced=False) for threat in threats)
        return Verdict(
            trace_id="t", site="support", session_id=None, action=Action.ALLOW, would_action=Action.ALLOW, hits=hits
        )


class TestRuntimeControls:
    def test_configure_changes_the_mode_without_a_restart(self, client: openai.OpenAI) -> None:
        immune.configure(mode="observe")
        assert EXFIL in chart(client)
        immune.configure(mode="auto")
        assert chart(client) == "Chart [link removed]"

    def test_off_mode_passes_everything_through(self, client: openai.OpenAI) -> None:
        immune.configure(mode="off")
        response = client.chat.completions.with_raw_response.create(model="m", messages=MESSAGES)
        assert "x-immune-trace" not in response.headers
        assert EXFIL in (response.parse().choices[0].message.content or "")

    def test_unscreened_calls_report_no_verdict(self, client: openai.OpenAI) -> None:
        chart(client)
        assert immune.verdict() is not None
        immune.configure(mode="off")
        chart(client)
        assert immune.verdict() is None
        immune.configure(mode="auto", sites={"reports": {"enabled": False}})
        chart(client)
        assert immune.verdict() is not None
        with immune.site("reports"):
            chart(client)
        assert immune.verdict() is None

    def test_sites_can_be_switched_off(self, client: openai.OpenAI) -> None:
        immune.configure(sites={"reports": {"enabled": False}})
        with immune.site("reports"):
            assert EXFIL in chart(client)
        with immune.site("chat"):
            assert chart(client) == "Chart [link removed]"

    def test_fixed_settings_cannot_change_at_runtime(self, client: openai.OpenAI) -> None:
        with pytest.raises(ConfigError):
            immune.configure(state={"backend": "sqlite"})
