from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, ClassVar

from immune.intercept.hooks import PostImportHooks

_LOGGER = logging.getLogger("immune")


class SdkAdapter(ABC):
    name: ClassVar[str]
    module: ClassVar[str]

    def install(self, hooks: PostImportHooks) -> None:
        hooks.when_imported(self.module, self.apply)

    @abstractmethod
    def apply(self, module: ModuleType) -> None: ...

    @abstractmethod
    def uninstall(self) -> None: ...

    @property
    @abstractmethod
    def applied(self) -> bool: ...


class GoogleGenAIRouting(SdkAdapter):
    name: ClassVar[str] = "google-genai async calls routed through httpx"
    module: ClassVar[str] = "google.genai._api_client"

    def __init__(self) -> None:
        self._owner: type[Any] | None = None
        self._original: Callable[..., bool] | None = None

    @property
    def applied(self) -> bool:
        return self._owner is not None

    def apply(self, module: ModuleType) -> None:
        owner = getattr(module, "BaseApiClient", None)
        if owner is None or not callable(getattr(owner, "_use_aiohttp", None)):
            _LOGGER.warning("immune: coverage gap: this google-genai version may send async calls through aiohttp")
            return
        if self._owner is None:
            self._owner, self._original = owner, owner._use_aiohttp
            owner._use_aiohttp = _never

    def uninstall(self) -> None:
        if self._owner is not None and self._original is not None:
            self._owner._use_aiohttp = self._original
        self._owner = self._original = None


def _never(_client: Any) -> bool:
    return False


@dataclass(slots=True)
class Coverage:
    points: list[str] = field(default_factory=list)
    adapters: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def report(self) -> None:
        for warning in self.warnings:
            _LOGGER.warning("immune: coverage gap: %s", warning)


class AdapterSet:
    def __init__(self, adapters: Iterable[SdkAdapter]) -> None:
        self._adapters = list(adapters)

    def install(self, hooks: PostImportHooks) -> list[str]:
        for adapter in self._adapters:
            adapter.install(hooks)
        return [adapter.name for adapter in self._adapters]

    def uninstall(self) -> None:
        for adapter in reversed(self._adapters):
            adapter.uninstall()


class DspyTransport(SdkAdapter):
    name: ClassVar[str] = "DSPy built-in transport"
    module: ClassVar[str] = "dspy._vendor.lm15.transports._sync"
    _TYPES: ClassVar[str] = "dspy._vendor.lm15.transports._types"

    def __init__(self, flow: Callable[[], Any]) -> None:
        self._flow = flow
        self._owner: type[Any] | None = None
        self._original: Callable[..., Any] | None = None

    @property
    def applied(self) -> bool:
        return self._owner is not None

    def apply(self, module: ModuleType) -> None:
        owner = getattr(module, "StdlibTransport", None)
        if owner is None or not callable(getattr(owner, "stream", None)) or self._owner is not None:
            return
        types = importlib.import_module(self._TYPES)
        original = owner.stream
        adapter = self

        def stream(transport: Any, request: Any) -> Any:
            flow = adapter._flow()
            if flow is None:
                return original(transport, request)
            return _DspyExchange(types, original, transport, request).run(flow)

        self._owner, self._original = owner, original
        owner.stream = stream

    def uninstall(self) -> None:
        if self._owner is not None and self._original is not None:
            self._owner.stream = self._original
        self._owner = self._original = None


class _DspyExchange:
    _DROPPED = frozenset({"content-encoding", "content-length", "transfer-encoding"})

    def __init__(self, types: ModuleType, original: Callable[..., Any], transport: Any, request: Any) -> None:
        self._types = types
        self._original = original
        self._transport = transport
        self._request = request
        self._httpx = importlib.import_module("httpx2")

    def run(self, flow: Any) -> Any:
        request = self._request
        outgoing = self._httpx.Request(request.method, request.url, headers=request.headers, content=request.body)
        response = flow.handle_sync(outgoing, self._send, self._httpx.Response)
        headers = [(key, value) for key, value in response.headers.items() if key.lower() not in self._DROPPED]
        return self._types.TransportResponse(
            status=response.status_code,
            reason=response.reason_phrase,
            headers=headers,
            http_version="HTTP/1.1",
            chunks=response.iter_bytes(),
            release=lambda _consumed: response.close(),
        )

    def _send(self, outgoing: Any) -> Any:
        request = self._request
        converted = self._types.TransportRequest(
            method=outgoing.method,
            url=str(outgoing.url),
            headers=[(key, value) for key, value in outgoing.headers.items() if key.lower() != "content-length"],
            body=outgoing.content,
            connect_timeout=request.connect_timeout,
            read_timeout=request.read_timeout,
            write_timeout=request.write_timeout,
        )
        response = self._original(self._transport, converted)
        stream = type(
            "DspyBody",
            (self._httpx.SyncByteStream,),
            {"__iter__": lambda _: iter(response), "close": lambda _: response.close()},
        )()
        return self._httpx.Response(response.status, headers=response.headers, stream=stream)
