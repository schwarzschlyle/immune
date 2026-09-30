from __future__ import annotations

import importlib
import json
import struct
import uuid
import zlib
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import unquote, urlsplit

from immune.codecs import CODECS
from immune.codecs.base import RequestContext
from immune.testing.fake_provider import FakeProvider, FakeReply, Script

_CODEC = CODECS["bedrock_converse"]
_STRING = 7


class EventStreamFrames:
    @staticmethod
    def encode(event_type: str, payload: Mapping[str, Any]) -> bytes:
        headers = b"".join(
            EventStreamFrames._header(name, value)
            for name, value in (
                (":event-type", event_type),
                (":content-type", "application/json"),
                (":message-type", "event"),
            )
        )
        body = json.dumps(payload).encode("utf-8")
        total = 12 + len(headers) + len(body) + 4
        prelude = struct.pack(">II", total, len(headers))
        prelude += struct.pack(">I", zlib.crc32(prelude))
        message = prelude + headers + body
        return message + struct.pack(">I", zlib.crc32(message))

    @staticmethod
    def _header(name: str, value: str) -> bytes:
        encoded_name, encoded_value = name.encode("utf-8"), value.encode("utf-8")
        return (
            struct.pack(">B", len(encoded_name))
            + encoded_name
            + struct.pack(">BH", _STRING, len(encoded_value))
            + encoded_value
        )


class _RawBody:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def stream(self, amt: int | None = None, decode_content: bool | None = None) -> Iterator[bytes]:
        yield from self._chunks

    def read(self, amt: int | None = None) -> bytes:
        return b"".join(self._chunks)

    def close(self) -> None:
        return None


class FakeBedrock:
    def __init__(self, script: Script | list[FakeReply] | FakeReply | None = None, delta_chars: int = 8) -> None:
        self._script = FakeProvider(script).script
        self._delta_chars = delta_chars
        self.requests: list[dict[str, Any]] = []

    @property
    def last_request(self) -> dict[str, Any]:
        return self.requests[-1]

    def client(self) -> Any:
        boto3 = importlib.import_module("boto3")
        client = boto3.client(
            "bedrock-runtime", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test"
        )
        client.meta.events.register("before-send.bedrock-runtime", self._send)
        return client

    def _send(self, request: Any, **_: Any) -> Any:
        awsrequest = importlib.import_module("botocore.awsrequest")
        body = json.loads(request.body)
        self.requests.append(body)
        path = unquote(urlsplit(request.url).path)
        context = RequestContext(path=path, body=body)
        reply = self._script(_CODEC.parse_request(context))
        response = self._body(reply)
        if not _CODEC.is_streaming(context):
            headers = {"content-type": "application/json", "x-amzn-requestid": uuid.uuid4().hex}
            return awsrequest.AWSResponse(request.url, 200, headers, _RawBody([json.dumps(response).encode()]))
        frames = [self._frame(event.json()) for event in self._events(response)]
        headers = {"content-type": "application/vnd.amazon.eventstream", "x-amzn-requestid": uuid.uuid4().hex}
        return awsrequest.AWSResponse(request.url, 200, headers, _RawBody(frames))

    def _events(self, response: Mapping[str, Any]) -> Iterator[Any]:
        for event in _CODEC.stream_from_body(response):
            payload = event.json()
            delta = payload.get("contentBlockDelta")
            text = (delta or {}).get("delta", {}).get("text") if isinstance(delta, Mapping) else None
            if not isinstance(text, str) or len(text) <= self._delta_chars:
                yield event
                continue
            for start in range(0, len(text), self._delta_chars):
                piece = {"contentBlockDelta": {**delta, "delta": {"text": text[start : start + self._delta_chars]}}}
                yield type(event).of(piece)

    @staticmethod
    def _frame(event: Mapping[str, Any]) -> bytes:
        ((event_type, payload),) = event.items()
        return EventStreamFrames.encode(event_type, payload)

    @staticmethod
    def _body(reply: FakeReply) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"text": reply.text}] if reply.text else []
        content.extend(
            {"toolUse": {"toolUseId": call.call_id, "name": call.name, "input": dict(call.arguments)}}
            for call in reply.tool_calls
        )
        return {
            "output": {"message": {"role": "assistant", "content": content}},
            "stopReason": "tool_use" if reply.tool_calls else "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 10, "totalTokens": 20},
            "metrics": {"latencyMs": 50},
        }
