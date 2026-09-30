from __future__ import annotations

import importlib
import logging
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from immune.intercept.botocore import BotocoreHooks
from immune.intercept.flow import InterceptionFlow
from immune.intercept.hooks import PostImportHooks

_LIBRARIES = ("httpx2", "httpx")
_LOGGER = logging.getLogger("immune")


@dataclass(frozen=True, slots=True)
class HttpLibrary:
    module: ModuleType

    @property
    def name(self) -> str:
        return self.module.__name__

    @property
    def response_type(self) -> type[Any]:
        response: type[Any] = self.module.Response
        return response

    @property
    def resolves_transports(self) -> bool:
        return hasattr(self.module.Client, "_transport_for_url") and hasattr(
            self.module.AsyncClient, "_transport_for_url"
        )

    def owns(self, client: Any) -> bool:
        return isinstance(client, (self.module.Client, self.module.AsyncClient))

    def is_async(self, client: Any) -> bool:
        return isinstance(client, self.module.AsyncClient)

    @classmethod
    def loaded(cls) -> list[HttpLibrary]:
        return [cls(sys.modules[name]) for name in _LIBRARIES if name in sys.modules]

    @classmethod
    def named(cls, name: str) -> HttpLibrary:
        return cls(importlib.import_module(name))

    @classmethod
    def owning(cls, client: Any) -> HttpLibrary | None:
        return next((library for library in cls.loaded() if library.owns(client)), None)


class FlowHolder:
    def __init__(self, flow: InterceptionFlow | None = None) -> None:
        self.flow = flow


class ImmuneTransport:
    def __init__(self, holder: FlowHolder, inner: Any, library: HttpLibrary) -> None:
        self._holder = holder
        self._inner = inner
        self._library = library

    @property
    def inner(self) -> Any:
        return self._inner

    def handle_request(self, request: Any) -> Any:
        flow = self._holder.flow
        if flow is None:
            return self._inner.handle_request(request)
        return flow.handle_sync(request, self._inner.handle_request, self._library.response_type)

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> ImmuneTransport:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class ImmuneAsyncTransport:
    def __init__(self, holder: FlowHolder, inner: Any, library: HttpLibrary) -> None:
        self._holder = holder
        self._inner = inner
        self._library = library

    @property
    def inner(self) -> Any:
        return self._inner

    async def handle_async_request(self, request: Any) -> Any:
        flow = self._holder.flow
        if flow is None:
            return await self._inner.handle_async_request(request)
        return await flow.handle_async(request, self._inner.handle_async_request, self._library.response_type)

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def __aenter__(self) -> ImmuneAsyncTransport:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


_WRAPPERS = (ImmuneTransport, ImmuneAsyncTransport)


def wrap_transport(holder: FlowHolder, transport: Any, library: HttpLibrary, asynchronous: bool) -> Any:
    if transport is None or isinstance(transport, _WRAPPERS):
        return transport
    wrapper = ImmuneAsyncTransport if asynchronous else ImmuneTransport
    return wrapper(holder, transport, library)


@dataclass(frozen=True, slots=True)
class InterceptionPoint:
    library: str
    method: str

    @property
    def label(self) -> str:
        return f"{self.library}: {self.method}"


class TransportPatcher:
    def __init__(self, holder: FlowHolder, hooks: PostImportHooks) -> None:
        self._holder = holder
        self._hooks = hooks
        self._originals: list[tuple[type[Any], str, Callable[..., Any]]] = []
        self._points: list[InterceptionPoint] = []
        self._pending: set[str] = set()
        self._active = False
        self._lock = threading.RLock()

    @property
    def installed(self) -> bool:
        return self._active

    @property
    def points(self) -> tuple[InterceptionPoint, ...]:
        with self._lock:
            pending = [InterceptionPoint(name, "patched when imported") for name in sorted(self._pending)]
            return (*self._points, *pending)

    def labels(self) -> list[str]:
        return [point.label for point in self.points]

    def install(self) -> list[str]:
        with self._lock:
            if self._active:
                return self.labels()
            self._active = True
            for name in _LIBRARIES:
                self._pending.add(name)
                self._hooks.when_imported(name, self._patch_library)
            return self.labels()

    def uninstall(self) -> None:
        with self._lock:
            for owner, attribute, original in reversed(self._originals):
                setattr(owner, attribute, original)
            self._originals.clear()
            self._points.clear()
            self._pending.clear()
            self._active = False

    def _patch_library(self, module: ModuleType) -> None:
        library = HttpLibrary(module)
        with self._lock:
            if not self._active or library.name not in self._pending:
                return
            self._pending.discard(library.name)
            if library.resolves_transports:
                self._patch_resolution(library, library.module.Client, asynchronous=False)
                self._patch_resolution(library, library.module.AsyncClient, asynchronous=True)
                self._points.append(InterceptionPoint(library.name, "every client transport"))
                return
            _LOGGER.warning(
                "immune: %s has no transport resolution hook; only its default transports are covered", library.name
            )
            self._patch_default_sync(library)
            self._patch_default_async(library)
            self._points.append(InterceptionPoint(library.name, "default transports only"))

    def _patch_resolution(self, library: HttpLibrary, owner: type[Any], asynchronous: bool) -> None:
        original = owner._transport_for_url
        holder = self._holder

        def _transport_for_url(client: Any, url: Any) -> Any:
            transport = original(client, url)
            if holder.flow is None:
                return transport
            return wrap_transport(holder, transport, library, asynchronous)

        self._originals.append((owner, "_transport_for_url", original))
        owner._transport_for_url = _transport_for_url

    def _patch_default_sync(self, library: HttpLibrary) -> None:
        owner = library.module.HTTPTransport
        original = owner.handle_request
        holder = self._holder

        def handle_request(transport: Any, request: Any) -> Any:
            flow = holder.flow
            if flow is None:
                return original(transport, request)
            return flow.handle_sync(request, lambda outgoing: original(transport, outgoing), library.response_type)

        self._originals.append((owner, "handle_request", original))
        owner.handle_request = handle_request

    def _patch_default_async(self, library: HttpLibrary) -> None:
        owner = library.module.AsyncHTTPTransport
        original = owner.handle_async_request
        holder = self._holder

        async def handle_async_request(transport: Any, request: Any) -> Any:
            flow = holder.flow
            if flow is None:
                return await original(transport, request)
            return await flow.handle_async(
                request, lambda outgoing: original(transport, outgoing), library.response_type
            )

        self._originals.append((owner, "handle_async_request", original))
        owner.handle_async_request = handle_async_request


class ClientProtector:
    def __init__(self, holder: FlowHolder, botocore: BotocoreHooks | None = None) -> None:
        self._holder = holder
        self._botocore = botocore

    def protect(self, target: Any) -> Any:
        if self._botocore is not None and self._botocore.attach(target):
            return target
        clients = self._http_clients(target)
        if not clients:
            raise TypeError(
                f"immune.protect() supports httpx clients, SDK clients built on them (OpenAI, Anthropic, "
                f"google-genai) and boto3 bedrock-runtime clients; use immune.init() for {type(target).__name__}"
            )
        for client in clients:
            self._wrap(client)
        return target

    def _wrap(self, client: Any) -> None:
        library = HttpLibrary.owning(client)
        if library is None:
            return
        asynchronous = library.is_async(client)
        client._transport = wrap_transport(self._holder, client._transport, library, asynchronous)
        client._mounts = {
            pattern: wrap_transport(self._holder, transport, library, asynchronous)
            for pattern, transport in client._mounts.items()
        }

    @staticmethod
    def _http_clients(target: Any) -> list[Any]:
        candidates = [
            target,
            getattr(target, "_client", None),
            getattr(getattr(target, "_api_client", None), "_httpx_client", None),
            getattr(getattr(target, "_api_client", None), "_async_httpx_client", None),
        ]
        return [candidate for candidate in candidates if candidate is not None and HttpLibrary.owning(candidate)]
