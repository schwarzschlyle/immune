from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import queue
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from immune.config.settings import LangSmithSettings, Settings
from immune.telemetry.alerts import Alert
from immune.telemetry.observers import CallObserver, CallStart, CallSummary, SensorExchange
from immune.telemetry.throttle import ThrottledLog
from immune.types import Hit, Mode, Verdict

_LOGGER = logging.getLogger("immune")
_THROTTLED = ThrottledLog(_LOGGER, interval_s=60.0)
_TRUE = frozenset({"1", "true", "yes", "on"})
_TEXT_LIMIT = 4_000
_PENDING = 10_000
_QUEUE = 1_000


class _ClientOverride:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._client: Any = None

    def get(self) -> Any:
        with self._lock:
            return self._client

    def swap(self, client: Any) -> Any:
        with self._lock:
            previous, self._client = self._client, client
        return previous


_OVERRIDE = _ClientOverride()


def use_client(client: Any) -> Callable[[], None]:
    """Send LangSmith runs to ``client`` instead of the real API, until the returned function is called.

    Runtimes created while an override is installed trace with it even when the LangSmith environment variables are
    unset. ``immune.testing.LangSmithRecorder`` uses this.
    """
    previous = _OVERRIDE.swap(client)

    def restore() -> None:
        _OVERRIDE.swap(previous)

    return restore


@dataclass(slots=True)
class _Trace:
    call: CallStart
    parent: Any
    exchanges: list[tuple[SensorExchange, float]] = field(default_factory=list)
    summary: CallSummary | None = None


class LangSmithObserver(CallObserver):
    """Logs calls with possible threats to LangSmith, nested in the app's own trace.

    Only calls whose verdict has at least one hit (enforced or observed) are sent, plus spike alerts. Clean calls
    never leave the process. Runs are built and posted on a background thread so a call never waits for LangSmith.
    """

    detailed = True

    def __init__(
        self,
        settings: LangSmithSettings,
        mask: Callable[[str], str],
        raw_inputs: bool = False,
        mode: Callable[[], Mode] = lambda: Mode.AUTO,
        vaccines: Mapping[str, str] | None = None,
        client: Any = None,
    ) -> None:
        self._settings = settings
        self._mask = mask
        self._raw = raw_inputs and settings.inputs == "raw"
        self._mode = mode
        self._vaccines = dict(vaccines or {})
        self._client = client
        self._run_trees = importlib.import_module("langsmith.run_trees")
        self._current: Callable[[], Any] = importlib.import_module("langsmith").get_current_run_tree
        self._pending: OrderedDict[str, _Trace] = OrderedDict()
        self._lock = threading.Lock()
        self._jobs: queue.Queue[Callable[[], None] | None] = queue.Queue(maxsize=_QUEUE)
        self._worker: threading.Thread | None = None
        self._clients: dict[int, Any] = {}
        self.dropped = 0

    @classmethod
    def create(
        cls,
        settings: Settings,
        mask: Callable[[str], str],
        mode: Callable[[], Mode],
        vaccines: Mapping[str, str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> LangSmithObserver | None:
        config = settings.telemetry.langsmith
        if config.enabled is False:
            return None
        client = _OVERRIDE.get()
        if importlib.util.find_spec("langsmith") is None:
            if config.enabled is True or client is not None:
                _LOGGER.warning(
                    "immune: telemetry.langsmith is on but langsmith is not installed; "
                    "install it with pip install 'immune-ai[langsmith]'"
                )
            return None
        if config.enabled == "auto" and client is None and not cls.configured(environ or os.environ):
            return None
        if config.inputs == "raw" and not settings.privacy.raw_evidence:
            _LOGGER.warning("immune: telemetry.langsmith.inputs: raw also needs privacy.log: full; sending masked")
        _LOGGER.info("immune: tracing possible threats to LangSmith")
        return cls(config, mask, settings.privacy.raw_evidence, mode, vaccines, client)

    @staticmethod
    def configured(environ: Mapping[str, str]) -> bool:
        tracing = environ.get("LANGSMITH_TRACING") or environ.get("LANGCHAIN_TRACING_V2") or ""
        key = environ.get("LANGSMITH_API_KEY") or environ.get("LANGCHAIN_API_KEY")
        return tracing.strip().lower() in _TRUE and bool(key)

    def started(self, call: CallStart) -> None:
        try:
            parent = self._current()
        except Exception:
            parent = None
        with self._lock:
            self._pending[call.trace_id] = _Trace(call, parent)
            while len(self._pending) > _PENDING:
                self._pending.popitem(last=False)

    def sensed(self, exchange: SensorExchange) -> None:
        with self._lock:
            trace = self._pending.get(exchange.trace_id)
            if trace is not None:
                trace.exchanges.append((exchange, time.time()))

    def summarized(self, summary: CallSummary) -> None:
        with self._lock:
            trace = self._pending.get(summary.trace_id)
            if trace is not None:
                trace.summary = summary

    def finished(self, call: CallStart, verdict: Verdict) -> None:
        with self._lock:
            trace = self._pending.pop(call.trace_id, None)
        if trace is None or not verdict.hits:
            return
        ended = time.time()
        mode = self._mode()
        self._submit(lambda: self._post_call(trace, verdict, mode, ended))

    def alert(self, alert: Alert) -> None:
        self._submit(lambda: self._post_alert(alert))

    def flush(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while self._jobs.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        for client in list(self._clients.values()):
            flush = getattr(client, "flush", None)
            if callable(flush):
                try:
                    flush(timeout=max(deadline - time.monotonic(), 0.0))
                except Exception:
                    _LOGGER.exception("immune: flushing LangSmith failed")

    def close(self) -> None:
        self.flush()
        if self._worker is not None:
            self._jobs.put(None)
            self._worker.join(timeout=2)
            self._worker = None

    def _submit(self, job: Callable[[], None]) -> None:
        with self._lock:
            if self._worker is None:
                self._worker = threading.Thread(target=self._drain, name="immune-langsmith", daemon=True)
                self._worker.start()
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            self.dropped += 1
            _THROTTLED.warning("langsmith-full", "immune: LangSmith queue is full; dropping runs")

    def _drain(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                job()
            except Exception as error:
                _THROTTLED.warning("langsmith-post", "immune: sending a run to LangSmith failed: %r", error)
            finally:
                self._jobs.task_done()

    def _post_call(self, trace: _Trace, verdict: Verdict, mode: Mode, ended: float) -> None:
        parent = trace.parent
        client = self._client_for(parent)
        options: dict[str, Any] = {}
        if parent is not None:
            options.update(parent_run=parent, project_name=parent.session_name)
        elif self._settings.project:
            options["project_name"] = self._settings.project
        run = self._run_trees.RunTree(
            name=self._name(verdict),
            run_type="chain",
            inputs=self._inputs(trace.summary),
            start_time=_time(trace.call.started_ns / 1e9),
            extra={"metadata": self._metadata(trace, verdict, mode)},
            tags=self._tags(verdict),
            ls_client=client,
            **options,
        )
        if self._settings.jev_runs:
            for exchange, at in trace.exchanges:
                self._jev_run(run, exchange, at)
        for warning in trace.summary.warnings if trace.summary is not None else ():
            run.add_event({"name": "immune.warning", "time": _time(ended).isoformat(), "message": warning})
        blocked = verdict.action.intervenes
        error = None
        if blocked and self._settings.mark_blocked_as_error:
            error = f"blocked by Immune: {verdict.explanation}"
        run.end(outputs=self._outputs(verdict), error=error, end_time=_time(ended))
        run.post(exclude_child_runs=False)
        if self._settings.feedback:
            self._feedback(client, parent.id if parent is not None else run.id, run.trace_id, verdict)

    def _post_alert(self, alert: Alert) -> None:
        client = self._client_for(None)
        options = {"project_name": self._settings.project} if self._settings.project else {}
        run = self._run_trees.RunTree(
            name=f"immune · alert · {alert.threat}",
            run_type="chain",
            inputs={"site": alert.site, "threat": alert.threat},
            start_time=_time(alert.at),
            tags=["immune", "immune:alert", f"threat:{alert.threat}", f"site:{alert.site}"],
            extra={"metadata": {"site": alert.site, "threat": alert.threat}},
            ls_client=client,
            **options,
        )
        outputs = {
            "message": alert.message,
            "fired": alert.fired,
            "screened": alert.screened,
            "rate": round(alert.rate, 4),
            "baseline_rate": None if alert.baseline_rate is None else round(alert.baseline_rate, 4),
        }
        run.end(outputs=outputs, end_time=_time(alert.at))
        run.post()

    def _jev_run(self, run: Any, exchange: SensorExchange, at: float) -> None:
        inputs: dict[str, Any] = {"questions": [{"key": q.key, "text": q.text} for q in exchange.questions]}
        if self._settings.inputs != "none":
            inputs["state"] = self._scrub(exchange.state)
        reading = exchange.reading
        answers = {}
        metadata: dict[str, Any] = {}
        if reading is not None:
            answers = {
                key: {"probability": round(signal.probability, 4), "value": signal.value}
                for key, signal in reading.signals.items()
            }
            metadata = {"jev_model": reading.model, "input_tokens": reading.input_tokens}
            metadata["latency_ms"] = round(reading.latency_ms, 1)
        child = run.create_child(
            name=f"jev · {exchange.name}",
            run_type="llm",
            inputs=inputs,
            start_time=run.start_time,
            extra={"metadata": metadata},
            tags=["immune", "jev"],
        )
        child.end(outputs={"answers": answers}, error=exchange.error, end_time=_time(at))

    def _feedback(self, client: Any, target: Any, trace_id: Any, verdict: Verdict) -> None:
        client.create_feedback(
            target,
            key="immune.blocked",
            score=1 if verdict.action.intervenes else 0,
            value=verdict.action.value,
            comment=verdict.explanation,
            trace_id=trace_id,
        )
        for hit in verdict.hits:
            client.create_feedback(
                target,
                key="immune.threat",
                score=round(hit.probability, 4),
                value=hit.threat,
                comment=self._text("; ".join(hit.evidence)) or None,
                trace_id=trace_id,
            )

    def _client_for(self, parent: Any) -> Any:
        client = self._client
        if client is None:
            client = parent.client if parent is not None else self._run_trees.get_cached_client()
        self._clients.setdefault(id(client), client)
        return client

    @staticmethod
    def _name(verdict: Verdict) -> str:
        top = _top(verdict)
        state = verdict.action.value if verdict.enforced_hits else "observed"
        return f"immune · {state} · {top.threat}"

    @staticmethod
    def _tags(verdict: Verdict) -> list[str]:
        tags = ["immune", f"immune:{verdict.action.value}"]
        tags.append("immune:enforced" if verdict.enforced_hits else "immune:observed")
        tags += [f"threat:{hit.threat}" for hit in verdict.hits]
        tags += [f"floor:{hit.floor}" for hit in verdict.enforced_hits if hit.floor]
        tags.append(f"site:{verdict.site}")
        return list(dict.fromkeys(tags))

    def _metadata(self, trace: _Trace, verdict: Verdict, mode: Mode) -> dict[str, Any]:
        summary = trace.summary
        metadata: dict[str, Any] = {
            "immune_trace_id": verdict.trace_id,
            "site": verdict.site,
            "mode": mode.value,
            "action": verdict.action.value,
            "would_action": verdict.would_action.value,
            "taint": verdict.taint.value,
            "session_risk": verdict.session_risk,
            "spec_version": verdict.spec_version,
            "config_hash": verdict.config_hash,
            "provider": trace.call.provider,
            "model": trace.call.model,
            "jev_model": verdict.sensor.model,
            "jev_calls": verdict.sensor.calls,
            "jev_latency_ms": round(verdict.sensor.latency_ms, 1),
            "jev_input_tokens": verdict.sensor.input_tokens,
        }
        if verdict.session_id:
            metadata["session_id"] = verdict.session_id
        fired = {hit.threat: self._vaccines[hit.threat] for hit in verdict.hits if hit.threat in self._vaccines}
        if fired:
            metadata["vaccines"] = fired
        if summary is not None and summary.warnings:
            metadata["warnings"] = list(summary.warnings)
        return metadata

    def _inputs(self, summary: CallSummary | None) -> dict[str, Any]:
        if summary is None:
            return {}
        inputs: dict[str, Any] = {"provider": summary.provider, "model": summary.model}
        if self._settings.inputs == "none":
            return inputs
        inputs["user"] = self._text(summary.user)
        if summary.data:
            inputs["data"] = [self._text(item) for item in summary.data]
        if summary.reply:
            inputs["reply"] = self._text(summary.reply)
        if summary.tools:
            inputs["tool_calls"] = [
                {"name": tool.name, "arguments": self._text(tool.arguments)} for tool in summary.tools
            ]
        return inputs

    def _outputs(self, verdict: Verdict) -> dict[str, Any]:
        return {
            "action": verdict.action.value,
            "would_action": verdict.would_action.value,
            "explanation": verdict.explanation,
            "threats": [self._threat(hit) for hit in verdict.hits],
        }

    def _threat(self, hit: Hit) -> dict[str, Any]:
        threat: dict[str, Any] = {
            "id": hit.threat,
            "stage": hit.stage.value,
            "probability": round(hit.probability, 4),
            "enforced": hit.enforced,
            "action": hit.action.value,
            "invariant": hit.invariant,
            "floor": hit.floor,
            "frameworks": list(hit.frameworks),
            "evidence": list(hit.evidence),
        }
        if hit.threat in self._vaccines:
            threat["vaccine"] = {"id": hit.threat, "version": self._vaccines[hit.threat]}
        return threat

    def _text(self, text: str) -> str:
        shown = text if self._raw else self._mask(text)
        return shown if len(shown) <= _TEXT_LIMIT else f"{shown[:_TEXT_LIMIT]}… [{len(shown) - _TEXT_LIMIT} more]"

    def _scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._text(value)
        if isinstance(value, Mapping):
            return {str(key): self._scrub(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._scrub(item) for item in value]
        return value


def _top(verdict: Verdict) -> Hit:
    candidates = verdict.enforced_hits or verdict.hits
    return max(candidates, key=lambda hit: hit.probability)


def _time(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, UTC)
