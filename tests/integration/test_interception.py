from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import httpx
import httpx2
import openai
import pytest

import immune
from immune.config.spec import QuestionSpec
from immune.core.pipeline import CallPipeline
from immune.intercept.adapters import GoogleGenAIRouting
from immune.intercept.hooks import PostImportHooks
from immune.intercept.transport import HttpLibrary
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading
from immune.testing import FakeProvider, FakeReply, MockSensor
from tests.integration.test_floor import EXFIL

CHART = FakeReply(text=f"Chart {EXFIL}")
MESSAGES = [{"role": "user", "content": "chart"}]


class ProviderTransport(httpx2.AsyncBaseTransport):
    def __init__(self, provider: FakeProvider) -> None:
        self._provider = provider

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        return self._provider.handle(request, httpx2.Response)


class HangingSensor(Sensor):
    name: ClassVar[str] = "hanging"

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        await asyncio.sleep(3600)
        return SensorReading.empty()


# Deadline tests prove a call does not wait for a stuck sensor. The sensor stays stuck far longer than the allowed
# time, which itself leaves room for slow CI runners (Windows under coverage can be 10x slower than a laptop).
WEDGE_S = 20.0
ALLOWED_S = 10.0


def block_the_current_thread(seconds: float) -> None:
    time.sleep(seconds)


class WedgingSensor(Sensor):
    name: ClassVar[str] = "wedging"

    def __init__(self) -> None:
        self.wedged = False
        self.resets = 0

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        if not self.wedged:
            self.wedged = True
            block_the_current_thread(WEDGE_S)
        return SensorReading.empty(self.name)

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture
def active(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
    immune.shutdown()


class TestEveryTransportIsCovered:
    async def test_custom_async_transports_are_screened(self, active: Path) -> None:
        immune.init(sensor=MockSensor(), state_dir=active)
        http_client = httpx2.AsyncClient(transport=ProviderTransport(FakeProvider(CHART)))
        client = openai.AsyncOpenAI(api_key="sk-test", http_client=http_client, max_retries=0)
        completion = await client.chat.completions.create(model="m", messages=MESSAGES)
        assert completion.choices[0].message.content == "Chart [link removed]"

    async def test_the_openai_aiohttp_client_is_screened(self, active: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        try:
            aiohttp_client = openai.DefaultAioHttpClient()
        except (AttributeError, RuntimeError):
            pytest.skip("the openai aiohttp extra is not installed")
        transport_type = type(aiohttp_client._transport)
        library = HttpLibrary.owning(aiohttp_client)
        assert library is not None
        provider = FakeProvider(CHART)

        async def handle(transport: Any, request: Any) -> Any:
            await request.aread()
            return provider.handle(request, library.response_type)

        monkeypatch.setattr(transport_type, "handle_async_request", handle)
        immune.init(sensor=MockSensor(), state_dir=active)
        client = openai.AsyncOpenAI(api_key="sk-test", http_client=aiohttp_client, max_retries=0)
        completion = await client.chat.completions.create(model="m", messages=MESSAGES)
        assert completion.choices[0].message.content == "Chart [link removed]"

    def test_legacy_httpx_clients_are_screened(self, active: Path) -> None:
        provider = FakeProvider(CHART)
        immune.init(sensor=MockSensor(), state_dir=active)
        transport = httpx.MockTransport(lambda request: provider.handle(request, httpx.Response))
        with httpx.Client(transport=transport) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions", json={"model": "m", "messages": MESSAGES}
            )
        assert response.json()["choices"][0]["message"]["content"] == "Chart [link removed]"
        assert response.headers["x-immune-trace"]

    def test_coverage_names_every_interception_point(self, active: Path) -> None:
        immune.init(sensor=MockSensor(), state_dir=active)
        points = immune.coverage().points
        assert "httpx2: every client transport" in points
        assert "httpx: every client transport" in points


class TestDeadlines:
    def test_a_hanging_sensor_falls_back_to_tier_zero(self, active: Path) -> None:
        provider = FakeProvider(CHART)
        immune.init(sensor=HangingSensor(), state_dir=active, config={"sensor": {"timeout_ms": 100}})
        client = openai.OpenAI(
            api_key="sk-test", http_client=httpx2.Client(transport=provider.transport(httpx2)), max_retries=0
        )
        started = time.perf_counter()
        completion = client.chat.completions.create(model="m", messages=MESSAGES)
        assert time.perf_counter() - started < ALLOWED_S
        assert completion.choices[0].message.content == "Chart [link removed]"
        verdict = immune.verdict(completion)
        assert verdict is not None
        assert verdict.sensor.name == "tier0_only"

    def test_a_wedged_sensor_loop_is_replaced(self, active: Path) -> None:
        provider = FakeProvider(FakeReply(text="Fine."))
        sensor = WedgingSensor()
        runtime = immune.init(
            sensor=sensor, state_dir=active, config={"sensor": {"timeout_ms": 100, "deadline_margin_ms": 100}}
        )
        assert runtime is not None
        client = openai.OpenAI(
            api_key="sk-test", http_client=httpx2.Client(transport=provider.transport(httpx2)), max_retries=0
        )
        started = time.perf_counter()
        client.chat.completions.create(model="m", messages=MESSAGES)
        assert time.perf_counter() - started < ALLOWED_S
        assert runtime.loop.generation == 1
        assert sensor.resets == 1
        client.chat.completions.create(model="m", messages=MESSAGES)
        assert runtime.loop.responsive()


class TestFailureHandling:
    def test_upstream_network_errors_reach_the_sdk(self, active: Path) -> None:
        def refuse(request: httpx2.Request) -> httpx2.Response:
            raise httpx2.ConnectError("connection refused", request=request)

        immune.init(sensor=MockSensor(), state_dir=active)
        client = openai.OpenAI(
            api_key="sk-test", http_client=httpx2.Client(transport=httpx2.MockTransport(refuse)), max_retries=0
        )
        with pytest.raises(openai.APIConnectionError):
            client.chat.completions.create(model="m", messages=MESSAGES)

    def test_internal_errors_pass_the_call_through(self, active: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(*_: object) -> None:
            raise RuntimeError("bug")

        monkeypatch.setattr(CallPipeline, "gate", explode)
        provider = FakeProvider(CHART)
        immune.init(sensor=MockSensor(), state_dir=active)
        client = openai.OpenAI(
            api_key="sk-test", http_client=httpx2.Client(transport=provider.transport(httpx2)), max_retries=0
        )
        completion = client.chat.completions.create(model="m", messages=MESSAGES)
        assert EXFIL in (completion.choices[0].message.content or "")
        verdict = immune.verdict(completion)
        assert verdict is not None
        assert verdict.explanation == "passed through after an internal error"

    async def test_async_internal_errors_pass_the_call_through(
        self, active: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*_: object) -> None:
            raise RuntimeError("bug")

        monkeypatch.setattr(CallPipeline, "plan_inbound", explode)
        immune.init(sensor=MockSensor(), state_dir=active)
        http_client = httpx2.AsyncClient(transport=ProviderTransport(FakeProvider(CHART)))
        client = openai.AsyncOpenAI(api_key="sk-test", http_client=http_client, max_retries=0)
        completion = await client.chat.completions.create(model="m", messages=MESSAGES)
        assert EXFIL in (completion.choices[0].message.content or "")

    def test_fail_closed_refuses_on_internal_errors(self, active: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(*_: object) -> None:
            raise RuntimeError("bug")

        monkeypatch.setattr(CallPipeline, "gate", explode)
        provider = FakeProvider(CHART)
        immune.init(sensor=MockSensor(), state_dir=active, config={"on_internal_error": "block"})
        client = openai.OpenAI(
            api_key="sk-test", http_client=httpx2.Client(transport=provider.transport(httpx2)), max_retries=0
        )
        completion = client.chat.completions.create(model="m", messages=MESSAGES)
        assert EXFIL not in (completion.choices[0].message.content or "")


class TestProtect:
    def test_protect_keeps_the_client_configuration(self, active: Path) -> None:
        provider = FakeProvider(CHART)
        seen: list[str] = []
        http_client = httpx2.Client(
            transport=provider.transport(httpx2),
            headers={"x-team": "search"},
            event_hooks={"request": [lambda request: seen.append(request.headers["x-team"])]},
        )
        original = openai.OpenAI(api_key="sk-test", http_client=http_client, max_retries=0)
        client = immune.protect(original, sensor=MockSensor(), state_dir=active)
        assert client is original
        completion = client.chat.completions.create(model="m", messages=MESSAGES)
        assert completion.choices[0].message.content == "Chart [link removed]"
        assert seen == ["search"]

    def test_protect_wraps_mounted_transports(self, active: Path) -> None:
        provider = FakeProvider(CHART)
        http_client = httpx2.Client(mounts={"https://api.openai.com": provider.transport(httpx2)})
        client = immune.protect(
            openai.OpenAI(api_key="sk-test", http_client=http_client, max_retries=0),
            sensor=MockSensor(),
            state_dir=active,
        )
        completion = client.chat.completions.create(model="m", messages=MESSAGES)
        assert completion.choices[0].message.content == "Chart [link removed]"

    def test_protect_accepts_plain_httpx_clients(self, active: Path) -> None:
        provider = FakeProvider(CHART)
        client = immune.protect(
            httpx.Client(transport=httpx.MockTransport(lambda request: provider.handle(request, httpx.Response))),
            sensor=MockSensor(),
            state_dir=active,
        )
        payload = json.dumps({"model": "m", "messages": MESSAGES})
        response = client.post("https://api.openai.com/v1/chat/completions", content=payload)
        assert "evil.test" not in response.text

    def test_protect_rejects_unknown_objects(self, active: Path) -> None:
        with pytest.raises(TypeError):
            immune.protect(object(), sensor=MockSensor(), state_dir=active)


class TestGoogleGenAIRouting:
    def test_async_calls_are_routed_through_httpx_while_active(self) -> None:
        from google.genai._api_client import BaseApiClient

        original = BaseApiClient._use_aiohttp
        hooks = PostImportHooks()
        routing = GoogleGenAIRouting()
        routing.install(hooks)
        try:
            assert routing.applied
            assert BaseApiClient._use_aiohttp(object()) is False
        finally:
            routing.uninstall()
            hooks.clear()
        assert BaseApiClient._use_aiohttp is original


def test_azure_openai_clients_are_screened(immune_harness: Any) -> None:
    harness = immune_harness(script=CHART)
    client = openai.AzureOpenAI(
        api_key="azure-test",
        api_version="2025-04-01-preview",
        azure_endpoint="https://acme.openai.azure.com",
        http_client=harness.http_client(),
        max_retries=0,
    )
    completion = client.chat.completions.create(model="gpt-5-deployment", messages=MESSAGES)
    assert completion.choices[0].message.content == "Chart [link removed]"
