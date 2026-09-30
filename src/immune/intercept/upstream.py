from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

_DROPPED_HEADERS = frozenset({"content-length", "content-encoding", "transfer-encoding"})
TRACE_HEADER = "x-immune-trace"


class Upstream:
    def __init__(self, response_type: type[Any]) -> None:
        self.response_type = response_type

    @staticmethod
    def rebuild(request: Any, body: bytes) -> Any:
        headers = [(key, value) for key, value in request.headers.items() if key.lower() != "content-length"]
        return type(request)(request.method, request.url, headers=headers, content=body, extensions=request.extensions)

    def respond(
        self, request: Any, template: Any | None, status: int, content: bytes, content_type: str, trace_id: str
    ) -> Any:
        headers = self._headers(template, content_type, trace_id)
        extensions = dict(template.extensions) if template is not None else {}
        return self.response_type(status, headers=headers, content=content, request=request, extensions=extensions)

    def respond_stream(self, request: Any, template: Any, stream: Any, content_type: str, trace_id: str) -> Any:
        headers = self._headers(template, content_type, trace_id)
        extensions = dict(template.extensions)
        return self.response_type(
            template.status_code, headers=headers, stream=stream, request=request, extensions=extensions
        )

    def tagged(self, response: Any, trace_id: str) -> Any:
        response.headers[TRACE_HEADER] = trace_id
        return response

    @staticmethod
    def _headers(template: Any | None, content_type: str, trace_id: str) -> list[tuple[str, str]]:
        headers: list[tuple[str, str]] = []
        if template is not None:
            headers = [(key, value) for key, value in template.headers.items() if key.lower() not in _DROPPED_HEADERS]
        if not any(key.lower() == "content-type" for key, _ in headers):
            headers.append(("content-type", content_type))
        headers.append((TRACE_HEADER, trace_id))
        return headers


class SyncUpstream(Upstream):
    def __init__(self, send: Callable[[Any], Any], response_type: type[Any]) -> None:
        super().__init__(response_type)
        self._send = send

    def send(self, request: Any) -> Any:
        return self._send(request)

    @staticmethod
    def read(response: Any) -> bytes:
        content: bytes = response.read()
        return content

    @staticmethod
    def close(response: Any) -> None:
        response.close()


class AsyncUpstream(Upstream):
    def __init__(self, send: Callable[[Any], Awaitable[Any]], response_type: type[Any]) -> None:
        super().__init__(response_type)
        self._send = send

    async def send(self, request: Any) -> Any:
        return await self._send(request)

    @staticmethod
    async def read(response: Any) -> bytes:
        content: bytes = await response.aread()
        return content

    @staticmethod
    async def close(response: Any) -> None:
        await response.aclose()

    def discard(self, pending: asyncio.Future[Any]) -> None:
        pending.cancel()
        pending.add_done_callback(_close_async_result)


_CLOSING: set[asyncio.Task[Any]] = set()


def _close_async_result(future: asyncio.Future[Any]) -> None:
    if not future.cancelled() and future.exception() is None:
        task = asyncio.ensure_future(future.result().aclose())
        _CLOSING.add(task)
        task.add_done_callback(_CLOSING.discard)
