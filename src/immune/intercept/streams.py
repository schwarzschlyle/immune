from __future__ import annotations

import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any


class ScreenedStreams:
    @staticmethod
    def sync(response_type: type[Any], chunks: Callable[[], Iterator[bytes]], close: Callable[[], None]) -> Any:
        base: type[Any] = ScreenedStreams._library(response_type).SyncByteStream

        def iterate(_: Any) -> Iterator[bytes]:
            return chunks()

        def finish(_: Any) -> None:
            close()

        return type("ScreenedSyncStream", (base,), {"__iter__": iterate, "close": finish})()

    @staticmethod
    def asynchronous(
        response_type: type[Any], chunks: Callable[[], AsyncIterator[bytes]], close: Callable[[], Awaitable[None]]
    ) -> Any:
        base: type[Any] = ScreenedStreams._library(response_type).AsyncByteStream

        def iterate(_: Any) -> AsyncIterator[bytes]:
            return chunks()

        async def finish(_: Any) -> None:
            await close()

        return type("ScreenedAsyncStream", (base,), {"__aiter__": iterate, "aclose": finish})()

    @staticmethod
    def _library(response_type: type[Any]) -> Any:
        return sys.modules[response_type.__module__.split(".")[0]]
