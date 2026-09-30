from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

import httpx
import httpx2
import pytest

from immune.config.spec import Spec
from immune.reflexes import ReflexSuite
from immune.testing import FakeProvider

_LOCAL = ("127.0.0.1", "::1", "localhost")

OPERATOR = "You are the support assistant for Acme Burgers. Help customers with the menu, orders and delivery."


@pytest.fixture(scope="session")
def spec() -> Spec:
    return Spec.default()


@pytest.fixture(scope="session")
def reflexes(spec: Spec) -> ReflexSuite:
    return ReflexSuite(spec)


def chat(user: str, operator: str = OPERATOR, **extra: object) -> dict[str, object]:
    return {
        "model": "gpt-5.5",
        "messages": [{"role": "system", "content": operator}, {"role": "user", "content": user}],
        **extra,
    }


def user_facing(**signals: object) -> dict[str, object]:
    return {"talks_to_end_users": 0.95, "archetype": "customer_service", **signals}


@pytest.fixture(autouse=True)
def no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live"):
        return
    original = socket.socket.connect

    def guarded(sock: socket.socket, address: Any) -> Any:
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and host not in _LOCAL and not host.startswith("/"):
            raise RuntimeError(f"test tried to reach the network: {address}")
        return original(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture
def fake_network(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeProvider], FakeProvider]:
    def install(provider: FakeProvider) -> FakeProvider:
        for module in (httpx, httpx2):
            sync = provider.transport(module)
            asynchronous = provider.async_transport(module)
            monkeypatch.setattr(module.HTTPTransport, "handle_request", _sync_handler(sync))
            monkeypatch.setattr(module.AsyncHTTPTransport, "handle_async_request", _async_handler(asynchronous))
        return provider

    return install


def _sync_handler(transport: Any) -> Callable[[Any, Any], Any]:
    def handle(_: Any, request: Any) -> Any:
        return transport.handle_request(request)

    return handle


def _async_handler(transport: Any) -> Callable[[Any, Any], Any]:
    async def handle(_: Any, request: Any) -> Any:
        return await transport.handle_async_request(request)

    return handle
