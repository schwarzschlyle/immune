from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from immune.config.loader import SettingsLoader, parse_mode
from immune.config.settings import Settings
from immune.core.runtime import Runtime
from immune.core.segment import UntrustedMarker
from immune.errors import ConfigError
from immune.intercept.adapters import AdapterSet, Coverage, DspyTransport, GoogleGenAIRouting
from immune.intercept.botocore import BotocoreHooks
from immune.intercept.hooks import PostImportHooks
from immune.intercept.transport import ClientProtector, FlowHolder, TransportPatcher
from immune.sensing.sensor import Sensor
from immune.sessions.resolver import CallContext
from immune.state.backend import StateBackend
from immune.telemetry.alerts import AlertCallback
from immune.telemetry.observers import VerdictCallback
from immune.telemetry.verdicts import VerdictIndex
from immune.types import Mode, SiteStatus, Verdict

_LOGGER = logging.getLogger("immune")
_DISABLED_VALUES = frozenset({"1", "true", "yes", "on"})
Client = TypeVar("Client")


class Immune:
    def __init__(self) -> None:
        self._holder = FlowHolder()
        self._hooks = PostImportHooks()
        self._patcher = TransportPatcher(self._holder, self._hooks)
        self._botocore = BotocoreHooks(lambda: self._holder.flow)
        self._protector = ClientProtector(self._holder, self._botocore)
        self._adapters = AdapterSet([GoogleGenAIRouting(), DspyTransport(lambda: self._holder.flow)])
        self._coverage = Coverage()
        self._runtime: Runtime | None = None
        self._lock = threading.RLock()

    @property
    def runtime(self) -> Runtime | None:
        return self._runtime

    @property
    def active(self) -> bool:
        return self._runtime is not None

    @property
    def coverage(self) -> Coverage:
        return Coverage(
            points=self._patcher.labels(), adapters=self._coverage.adapters, warnings=self._coverage.warnings
        )

    def init(
        self,
        *,
        mode: str | Mode | None = None,
        config: str | Path | dict[str, Any] | Settings | None = None,
        sensor: Sensor | None = None,
        state_dir: str | Path | None = None,
        patch: bool = True,
        backend: StateBackend | None = None,
        vaccines: Sequence[str | Path] | None = None,
    ) -> Runtime | None:
        if os.environ.get("IMMUNE_DISABLED", "").strip().lower() in _DISABLED_VALUES:
            _LOGGER.warning("immune: disabled by IMMUNE_DISABLED")
            return None
        settings = SettingsLoader().load(
            config, mode=parse_mode(mode), state_dir=Path(state_dir) if state_dir is not None else None
        )
        if vaccines:
            paths = (*settings.vaccines.paths, *(Path(path) for path in vaccines))
            settings = settings.model_copy(update={"vaccines": settings.vaccines.model_copy(update={"paths": paths})})
        runtime = Runtime(settings, sensor=sensor, backend=backend)
        with self._lock:
            previous, self._runtime = self._runtime, runtime
            self._holder.flow = runtime.flow
            if patch:
                points = self._patcher.install()
                adapters = self._adapters.install(self._hooks)
                self._botocore.install(self._hooks)
                adapters.append("boto3 bedrock-runtime Converse and ConverseStream")
                self._coverage = Coverage(points=points, adapters=adapters)
                _LOGGER.info(
                    "immune: protecting LLM calls made through %s (mode=%s)", "; ".join(points), settings.mode.value
                )
                self._coverage.report()
        if previous is not None:
            previous.close()
        return runtime

    def shutdown(self) -> None:
        with self._lock:
            runtime, self._runtime = self._runtime, None
            self._holder.flow = None
            self._hooks.clear()
            self._patcher.uninstall()
            self._adapters.uninstall()
            self._botocore.uninstall()
            self._coverage = Coverage()
        if runtime is not None:
            runtime.close()

    def protect(self, client: Client, **options: Any) -> Client:
        with self._lock:
            if self._runtime is None:
                self.init(patch=False, **options)
        protected: Client = self._protector.protect(client)
        return protected

    def configure(self, changes: Mapping[str, Any]) -> Settings:
        runtime = self._require()
        merged = _deep_merge(runtime.settings.model_dump(), dict(changes))
        settings = SettingsLoader(environ={}).load(merged)
        runtime.reconfigure(settings)
        _LOGGER.info("immune: reconfigured (mode=%s)", settings.mode.value)
        return settings

    def on_verdict(self, callback: VerdictCallback) -> Callable[[], None]:
        return self._require().callbacks.add(callback)

    def on_alert(self, callback: AlertCallback) -> None:
        self._require().alerts.on_alert(callback)

    def verdict(self, subject: Any = None) -> Verdict | None:
        runtime = self._runtime
        found = runtime.index.lookup(subject) if runtime is not None and subject is not None else None
        if found is not None:
            return found
        last = VerdictIndex.last()
        if last is None or runtime is None or runtime.index.owns(last):
            return last
        return None

    def feedback(self, trace_id: str, label: str, note: str | None = None, threat: str | None = None) -> None:
        runtime = self._require()
        verdict = runtime.index.lookup(trace_id)
        if verdict is None:
            raise ConfigError(f"no recent verdict with trace id {trace_id!r}")
        runtime.labels.add(verdict, label, note, threat)

    def status(self) -> list[SiteStatus]:
        runtime = self._runtime
        return runtime.status() if runtime is not None else []

    def _require(self) -> Runtime:
        if self._runtime is None:
            raise ConfigError("immune is not initialized; call immune.init() first")
        return self._runtime


_IMMUNE = Immune()


def init(
    *,
    mode: str | Mode | None = None,
    config: str | Path | dict[str, Any] | Settings | None = None,
    sensor: Sensor | None = None,
    state_dir: str | Path | None = None,
    backend: StateBackend | None = None,
    vaccines: Sequence[str | Path] | None = None,
) -> Runtime | None:
    return _IMMUNE.init(
        mode=mode, config=config, sensor=sensor, state_dir=state_dir, backend=backend, vaccines=vaccines
    )


def shutdown() -> None:
    _IMMUNE.shutdown()


def protect(client: Client, **options: Any) -> Client:
    return _IMMUNE.protect(client, **options)


def session(session_id: str) -> contextlib.AbstractContextManager[None]:
    return CallContext.session(session_id)


def site(name: str) -> contextlib.AbstractContextManager[None]:
    return CallContext.site(name)


def untrusted(text: str, *, source: str = "data") -> str:
    return UntrustedMarker.wrap(text, source)


def verdict(subject: Any = None) -> Verdict | None:
    return _IMMUNE.verdict(subject)


def feedback(trace_id: str, label: str, note: str | None = None, threat: str | None = None) -> None:
    _IMMUNE.feedback(trace_id, label, note, threat)


def status() -> list[SiteStatus]:
    return _IMMUNE.status()


def runtime() -> Runtime | None:
    return _IMMUNE.runtime


def configure(**changes: Any) -> Settings:
    return _IMMUNE.configure(changes)


def on_verdict(callback: VerdictCallback) -> Callable[[], None]:
    return _IMMUNE.on_verdict(callback)


def on_alert(callback: AlertCallback) -> None:
    _IMMUNE.on_alert(callback)


def _deep_merge(base: dict[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in changes.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def coverage() -> Coverage:
    return _IMMUNE.coverage


@contextlib.contextmanager
def protected(**options: Any) -> Iterator[Runtime | None]:
    active = _IMMUNE.init(**options)
    try:
        yield active
    finally:
        _IMMUNE.shutdown()
