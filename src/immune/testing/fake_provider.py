from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from immune.codecs.base import Codec, StreamCursor, StreamKind
from immune.config.spec import Spec
from immune.core.conversation import Conversation
from immune.core.sse import ServerSentEvent, ServerSentEvents
from immune.intercept.router import EndpointRouter


@dataclass(frozen=True, slots=True)
class FakeToolCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")


@dataclass(frozen=True, slots=True)
class FakeReply:
    text: str = ""
    tool_calls: Sequence[FakeToolCall] = ()


Script = Callable[[Conversation], FakeReply]


class _Bodies:
    @staticmethod
    def openai_chat(conversation: Conversation, reply: FakeReply) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": reply.text or None, "refusal": None}
        if reply.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                }
                for call in reply.tool_calls
            ]
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": conversation.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": None,
                    "finish_reason": "tool_calls" if reply.tool_calls else "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }

    @staticmethod
    def openai_responses(conversation: Conversation, reply: FakeReply) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        if reply.text:
            output.append(
                {
                    "id": f"msg_{uuid.uuid4().hex[:8]}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": reply.text, "annotations": [], "logprobs": []}],
                }
            )
        output.extend(
            {
                "id": f"fc_{uuid.uuid4().hex[:8]}",
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": json.dumps(call.arguments),
                "status": "completed",
            }
            for call in reply.tool_calls
        )
        return {
            "id": f"resp_{uuid.uuid4().hex[:12]}",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": conversation.model,
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 20,
            },
        }

    @staticmethod
    def anthropic_messages(conversation: Conversation, reply: FakeReply) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": reply.text}] if reply.text else []
        content.extend(
            {"type": "tool_use", "id": call.call_id, "name": call.name, "input": dict(call.arguments)}
            for call in reply.tool_calls
        )
        return {
            "id": f"msg_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "model": conversation.model,
            "content": content,
            "stop_reason": "tool_use" if reply.tool_calls else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }

    @staticmethod
    def gemini(conversation: Conversation, reply: FakeReply) -> dict[str, Any]:
        parts: list[dict[str, Any]] = [{"text": reply.text}] if reply.text else []
        parts.extend(
            {"functionCall": {"id": call.call_id, "name": call.name, "args": dict(call.arguments)}}
            for call in reply.tool_calls
        )
        return {
            "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20},
            "modelVersion": conversation.model,
            "responseId": uuid.uuid4().hex[:12],
        }


class FakeProvider:
    def __init__(
        self,
        script: Script | Sequence[FakeReply] | FakeReply | None = None,
        status: int = 200,
        delta_chars: int = 8,
    ) -> None:
        self._script = self._normalize(script)
        self._status = status
        self._delta_chars = delta_chars
        self._router = EndpointRouter(Spec.default().endpoints)
        self.requests: list[dict[str, Any]] = []
        self.conversations: list[Conversation] = []

    def handle(self, request: Any, response_type: type[Any], asynchronous: bool = False) -> Any:
        content = bytes(request.content)
        routed = self._router.route(request.method, request.url.host, request.url.path, content)
        if routed is None:
            return response_type(404, json={"error": "unknown endpoint"}, request=request)
        body = json.loads(content)
        self.requests.append(body)
        if self._status >= 400:
            return response_type(self._status, json={"error": {"message": "upstream failure"}}, request=request)
        conversation = routed.codec.parse_request(routed.context)
        self.conversations.append(conversation)
        payload = getattr(_Bodies, routed.codec.name)(conversation, self._script(conversation))
        if routed.codec.is_streaming(routed.context):
            events = self._split(routed.codec, routed.codec.stream_from_body(payload))
            return response_type(
                200,
                headers={"content-type": "text/event-stream"},
                request=request,
                content=self._chunks([ServerSentEvents.serialize([event]) for event in events], asynchronous),
            )
        return response_type(200, json=payload, request=request)

    @staticmethod
    def _chunks(chunks: list[bytes], asynchronous: bool) -> _SyncChunks | _AsyncChunks:
        return _AsyncChunks(chunks) if asynchronous else _SyncChunks(chunks)

    def _split(self, codec: Codec, events: list[ServerSentEvent]) -> list[ServerSentEvent]:
        cursor = StreamCursor()
        split: list[ServerSentEvent] = []
        for event in events:
            part = codec.classify_event(event, cursor)
            if part.kind is not StreamKind.TEXT or len(part.text) <= self._delta_chars:
                split.append(event)
                continue
            text = part.text
            split.extend(
                codec.text_event(cursor, text[start : start + self._delta_chars])
                for start in range(0, len(text), self._delta_chars)
            )
        cursor.sequence = 0
        return [cursor.sequenced(event) for event in split]

    def transport(self, library: Any) -> Any:
        return _SyncTransport(self, library)

    def async_transport(self, library: Any) -> Any:
        return _AsyncTransport(self, library)

    @property
    def last_request(self) -> dict[str, Any]:
        return self.requests[-1]

    @property
    def script(self) -> Script:
        return self._script

    @staticmethod
    def _normalize(script: Script | Sequence[FakeReply] | FakeReply | None) -> Script:
        if script is None:
            return lambda _: FakeReply(text="Hello! How can I help you today?")
        if isinstance(script, FakeReply):
            return lambda _: script
        if callable(script):
            return script
        replies = list(script)
        return lambda _: replies.pop(0) if len(replies) > 1 else replies[0]


class _SyncTransport:
    def __init__(self, provider: FakeProvider, library: Any) -> None:
        self._provider = provider
        self._library = library

    def handle_request(self, request: Any) -> Any:
        request.read()
        return self._provider.handle(request, self._library.Response)

    def close(self) -> None:
        return None

    def __enter__(self) -> _SyncTransport:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class _AsyncTransport:
    def __init__(self, provider: FakeProvider, library: Any) -> None:
        self._provider = provider
        self._library = library

    async def handle_async_request(self, request: Any) -> Any:
        await request.aread()
        return self._provider.handle(request, self._library.Response, asynchronous=True)

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> _AsyncTransport:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


class _SyncChunks:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._chunks)


class _AsyncChunks:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
