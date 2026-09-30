from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from immune.config.loader import SettingsLoader
from immune.telemetry.alerts import Alert
from immune.telemetry.langsmith import LangSmithObserver
from immune.testing import FakeReply, ImmuneHarness, LangSmithRecorder, MockSensor
from immune.types import Mode
from tests.conftest import OPERATOR

langsmith = pytest.importorskip("langsmith")

Factory = Callable[..., ImmuneHarness]
LEAK = "Use key AKIAABCDEFGHIJKLMNOP to connect."


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Iterator[LangSmithRecorder]:
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with LangSmithRecorder() as installed:
        yield installed


def chat(harness: ImmuneHarness, user: str = "How do I connect?") -> Any:
    return harness.openai().chat.completions.create(
        model="gpt-5.5", messages=[{"role": "system", "content": OPERATOR}, {"role": "user", "content": user}]
    )


def only_run(recorder: LangSmithRecorder, harness: ImmuneHarness) -> dict[str, Any]:
    harness.close()
    runs = recorder.immune_runs()
    assert len(runs) == 1, [run["name"] for run in runs]
    return runs[0]


class TestWhatIsSent:
    def test_clean_calls_send_nothing(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness(script=FakeReply(text="Open the app and sign in."))
        chat(harness)
        harness.close()
        assert recorder.runs == []
        assert recorder.feedback == []

    def test_enforced_threats_become_tagged_runs(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness(script=FakeReply(text=LEAK))
        chat(harness, "How do I connect? I'm bob@acme.test")
        run = only_run(recorder, harness)
        verdict = harness.verdict()
        assert verdict is not None
        hit = verdict.hits[0]
        assert run["name"] == f"immune · {verdict.action.value} · output.secret_leak"
        assert run["run_type"] == "chain"
        assert {"immune", "immune:enforced", "threat:output.secret_leak", f"floor:{hit.floor}"} <= set(run["tags"])
        assert f"site:{verdict.site}" in run["tags"]
        metadata = run["extra"]["metadata"]
        assert metadata["immune_trace_id"] == verdict.trace_id
        assert metadata["spec_version"] == verdict.spec_version
        assert metadata["mode"] == "auto"
        threat = run["outputs"]["threats"][0]
        assert (threat["id"], threat["enforced"], threat["floor"]) == ("output.secret_leak", True, hit.floor)
        assert not run.get("error")

    def test_inputs_are_masked_by_default(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness(script=FakeReply(text=LEAK))
        chat(harness, "How do I connect? I'm bob@acme.test")
        run = only_run(recorder, harness)
        assert run["inputs"]["user"] == "How do I connect? I'm [EMAIL]"
        assert "AKIAABCDEFGHIJKLMNOP" not in str(run)
        assert "bob@acme.test" not in str(run)

    def test_inputs_can_be_left_out(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness(script=FakeReply(text=LEAK), config={"telemetry": {"langsmith": {"inputs": "none"}}})
        chat(harness)
        run = only_run(recorder, harness)
        assert set(run["inputs"]) == {"provider", "model"}
        assert all("state" not in child["inputs"] for child in recorder.children(run))

    def test_raw_inputs_also_need_full_privacy_logs(
        self, immune_harness: Factory, recorder: LangSmithRecorder, caplog: pytest.LogCaptureFixture
    ) -> None:
        masked = immune_harness(script=FakeReply(text=LEAK), config={"telemetry": {"langsmith": {"inputs": "raw"}}})
        assert "inputs: raw also needs privacy.log: full" in caplog.text
        chat(masked, "I'm bob@acme.test")
        assert only_run(recorder, masked)["inputs"]["user"] == "I'm [EMAIL]"
        recorder.runs.clear()
        raw = immune_harness(
            script=FakeReply(text=LEAK),
            config={"telemetry": {"langsmith": {"inputs": "raw"}}, "privacy": {"log": "full"}},
        )
        chat(raw, "I'm bob@acme.test")
        assert only_run(recorder, raw)["inputs"]["user"] == "I'm bob@acme.test"

    def test_observed_threats_are_logged_too(
        self, immune_harness: Factory, recorder: LangSmithRecorder, tmp_path: Path
    ) -> None:
        vaccine = tmp_path / "competitors.yaml"
        document = {"id": "acme.competitors", "version": "1.2.0", "title": "Competitors", "stage": "output"}
        vaccine.write_text(yaml.safe_dump({**document, "detect": {"keywords": ["Burger Palace"]}}), "utf-8")
        harness = immune_harness(
            script=FakeReply(text="Try Burger Palace."), config={"vaccines": {"paths": [str(vaccine)]}}
        )
        chat(harness, "Ideas?")
        run = only_run(recorder, harness)
        assert run["name"] == "immune · observed · acme.competitors"
        assert "immune:observed" in run["tags"]
        assert run["outputs"]["threats"][0]["vaccine"] == {"id": "acme.competitors", "version": "1.2.0"}
        assert run["extra"]["metadata"]["vaccines"] == {"acme.competitors": "1.2.0"}
        assert recorder.feedback_on(run["id"], "immune.blocked")[0]["score"] == 0

    def test_blocks_can_be_marked_as_errors(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness(
            script=FakeReply(text=LEAK), config={"telemetry": {"langsmith": {"mark_blocked_as_error": True}}}
        )
        chat(harness)
        verdict = harness.verdict()
        assert verdict is not None
        run = only_run(recorder, harness)
        assert verdict.blocked
        assert run["error"].startswith("blocked by Immune:")


class TestTraceShape:
    def test_jev_requests_become_child_runs(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness(script=FakeReply(text=LEAK))
        chat(harness)
        run = only_run(recorder, harness)
        children = recorder.children(run)
        assert children
        assert all(child["run_type"] == "llm" and child["name"].startswith("jev · ") for child in children)
        assert all(child["trace_id"] == run["trace_id"] for child in children)
        first = children[0]
        assert first["inputs"]["questions"]
        assert first["outputs"]["answers"]

    def test_jev_child_runs_can_be_turned_off(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness(script=FakeReply(text=LEAK), config={"telemetry": {"langsmith": {"jev_runs": False}}})
        chat(harness)
        assert recorder.children(only_run(recorder, harness)) == []

    def test_runs_nest_under_the_apps_trace(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness(script=FakeReply(text=LEAK))

        @langsmith.traceable(client=recorder, name="support-agent")
        def app() -> None:
            chat(harness)

        with langsmith.tracing_context(enabled=True):
            app()
        run = only_run(recorder, harness)
        parent = next(item for item in recorder.runs if item["name"] == "support-agent")
        assert run["parent_run_id"] == parent["id"]
        assert run["trace_id"] == parent["trace_id"]
        blocked = recorder.feedback_on(parent["id"], "immune.blocked")
        assert blocked[0]["score"] == 1
        threats = recorder.feedback_on(parent["id"], "immune.threat")
        assert [(item["value"], item["trace_id"]) for item in threats] == [("output.secret_leak", run["trace_id"])]

    def test_feedback_goes_on_immunes_run_without_a_parent(
        self, immune_harness: Factory, recorder: LangSmithRecorder
    ) -> None:
        harness = immune_harness(script=FakeReply(text=LEAK))
        chat(harness)
        run = only_run(recorder, harness)
        assert "parent_run_id" not in run or run["parent_run_id"] is None
        assert {item["key"] for item in recorder.feedback_on(run["id"])} == {"immune.blocked", "immune.threat"}

    def test_sensor_outages_are_attached_as_events(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness(sensor=MockSensor(fail=True), script=FakeReply(text=LEAK))
        chat(harness)
        run = only_run(recorder, harness)
        warnings = [event["message"] for event in run["events"] if event["name"] == "immune.warning"]
        assert any("Jev was unavailable" in message for message in warnings)
        assert run["extra"]["metadata"]["warnings"] == warnings
        errors = [child["error"] for child in recorder.children(run)]
        assert errors
        assert all(error and "SensorUnavailable" in error for error in errors)

    def test_alerts_become_runs(self, immune_harness: Factory, recorder: LangSmithRecorder) -> None:
        harness = immune_harness()
        observer = harness.runtime.langsmith
        assert observer is not None
        observer.alert(Alert("support", "input.override", 9, 40, 0.225, 0.01, 1_790_000_000.0))
        harness.close()
        run = next(item for item in recorder.runs if item["name"] == "immune · alert · input.override")
        assert "immune:alert" in run["tags"]
        assert run["outputs"]["fired"] == 9


class TestActivation:
    @pytest.mark.parametrize(
        ("environ", "expected"),
        [
            ({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "lsv2-x"}, True),
            ({"LANGCHAIN_TRACING_V2": "1", "LANGCHAIN_API_KEY": "lsv2-x"}, True),
            ({"LANGSMITH_TRACING": "true"}, False),
            ({"LANGSMITH_TRACING": "false", "LANGSMITH_API_KEY": "lsv2-x"}, False),
            ({}, False),
        ],
    )
    def test_auto_follows_the_langsmith_environment(self, environ: dict[str, str], expected: bool) -> None:
        assert LangSmithObserver.configured(environ) is expected
        settings = SettingsLoader(environ={}).load()
        observer = LangSmithObserver.create(settings, str, lambda: Mode.AUTO, environ=environ)
        assert (observer is not None) is expected

    def test_false_turns_it_off_even_with_a_recorder(self, recorder: LangSmithRecorder) -> None:
        settings = SettingsLoader(environ={}).load({"telemetry": {"langsmith": {"enabled": False}}})
        assert LangSmithObserver.create(settings, str, lambda: Mode.AUTO, environ={}) is None

    def test_missing_package_is_reported_when_asked_for(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
        settings = SettingsLoader(environ={}).load({"telemetry": {"langsmith": {"enabled": True}}})
        with caplog.at_level(logging.WARNING, logger="immune"):
            assert LangSmithObserver.create(settings, str, lambda: Mode.AUTO, environ={}) is None
        assert "pip install 'immune-ai[langsmith]'" in caplog.text
