from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from immune.config.loader import SettingsLoader
from immune.config.settings import Settings
from immune.core.runtime import Runtime
from immune.intercept.transport import FlowHolder, HttpLibrary, ImmuneAsyncTransport, ImmuneTransport
from immune.sensing.offline import MockSensor
from immune.sensing.sensor import Sensor
from immune.state.backend import StateBackend
from immune.telemetry.verdicts import VerdictIndex
from immune.testing.fake_provider import FakeProvider, FakeReply, Script
from immune.types import Mode, Verdict


class ImmuneHarness:
    def __init__(
        self,
        state_dir: Path,
        sensor: Sensor | None = None,
        script: Script | list[FakeReply] | FakeReply | None = None,
        mode: Mode | str | None = None,
        config: dict[str, Any] | str | Path | None = None,
        status: int = 200,
        backend: StateBackend | None = None,
    ) -> None:
        self.sensor = sensor or MockSensor()
        self.provider = FakeProvider(script, status)
        source = config if isinstance(config, (str, Path)) else dict(config or {})
        settings = SettingsLoader(environ={}).load(source, mode=Mode(mode) if mode else None, state_dir=state_dir)
        self.runtime = Runtime(settings, sensor=self.sensor, backend=backend)
        self._holder = FlowHolder(self.runtime.flow)
        self._library = HttpLibrary.named("httpx2")

    @property
    def settings(self) -> Settings:
        return self.runtime.settings

    def http_client(self) -> Any:
        transport = ImmuneTransport(self._holder, self.provider.transport(self._library.module), self._library)
        return self._library.module.Client(transport=transport)

    def async_http_client(self) -> Any:
        transport = ImmuneAsyncTransport(
            self._holder, self.provider.async_transport(self._library.module), self._library
        )
        return self._library.module.AsyncClient(transport=transport)

    def openai(self, asynchronous: bool = False, base_url: str = "https://api.openai.com/v1") -> Any:
        module = importlib.import_module("openai")
        if asynchronous:
            return module.AsyncOpenAI(
                api_key="sk-test", base_url=base_url, http_client=self.async_http_client(), max_retries=0
            )
        return module.OpenAI(api_key="sk-test", base_url=base_url, http_client=self.http_client(), max_retries=0)

    def anthropic(self, asynchronous: bool = False) -> Any:
        module = importlib.import_module("anthropic")
        if asynchronous:
            return module.AsyncAnthropic(api_key="sk-ant-test", http_client=self.async_http_client(), max_retries=0)
        return module.Anthropic(api_key="sk-ant-test", http_client=self.http_client(), max_retries=0)

    def gemini(self) -> Any:
        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")
        legacy = HttpLibrary.named("httpx")
        sync_transport = ImmuneTransport(self._holder, self.provider.transport(legacy.module), legacy)
        async_transport = ImmuneAsyncTransport(self._holder, self.provider.async_transport(legacy.module), legacy)
        options = types.HttpOptions(
            httpx_client=legacy.module.Client(transport=sync_transport),
            httpx_async_client=legacy.module.AsyncClient(transport=async_transport),
        )
        return genai.Client(api_key="gemini-test", http_options=options)

    def verdict(self, subject: Any = None) -> Verdict | None:
        index = self.runtime.index
        found = index.lookup(subject) if subject is not None else None
        if found is not None:
            return found
        last = VerdictIndex.last()
        return last if last is not None and index.owns(last) else None

    def close(self) -> None:
        self.runtime.close()

    def __enter__(self) -> ImmuneHarness:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
