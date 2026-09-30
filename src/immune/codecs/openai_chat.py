from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from immune.codecs.base import (
    Arguments,
    Codec,
    JsonBody,
    ReplyEdit,
    RequestContext,
    Shapes,
    StreamCursor,
    StreamKind,
    StreamPart,
    TextParts,
)
from immune.core.conversation import Channel, Conversation, PastCall, Reply, Segment, ToolCall, ToolSpec
from immune.core.documents import JsonDocument
from immune.core.sse import ServerSentEvent

_ROLE_CHANNELS = {
    "system": Channel.OPERATOR,
    "developer": Channel.OPERATOR,
    "user": Channel.USER,
    "assistant": Channel.ASSISTANT,
    "tool": Channel.DATA,
    "function": Channel.DATA,
}


class OpenAIChatCodec(Codec):
    name: ClassVar[str] = "openai_chat"
    response_prefix: ClassVar[str] = "chatcmpl-immune-"

    def accepts(self, context: RequestContext) -> bool:
        return isinstance(context.body.get("messages"), list)

    def strictly_accepts(self, context: RequestContext) -> bool:
        body = context.body
        return isinstance(body.get("model"), str) and Shapes.messages(body.get("messages"), Shapes.CHAT_ROLES)

    def parse_request(self, context: RequestContext) -> Conversation:
        body = context.body
        segments: list[Segment] = []
        history: list[PastCall] = []
        call_names: dict[str, str] = {}
        turn = 0
        for index, message in enumerate(body.get("messages") or []):
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role", ""))
            channel = _ROLE_CHANNELS.get(role)
            if channel is None:
                continue
            if channel is Channel.USER:
                turn += 1
            if role == "assistant":
                history.extend(PastCall(call, turn) for call in self._calls(message.get("tool_calls")))
                call_names.update((call.call_id, call.name) for call in self._calls(message.get("tool_calls")))
            origin = self._origin(role, message, call_names)
            for locator, text in TextParts.walk(message.get("content"), ("messages", index, "content")):
                segments.append(Segment(channel=channel, text=text, origin=origin, locator=locator, turn=turn))
        return Conversation(
            provider=self.name,
            model=str(body.get("model", "")),
            segments=tuple(segments),
            tools=self._tools(body),
            response_schema=self._schema(body),
            stream=bool(body.get("stream")),
            user_hint=body.get("safety_identifier") or body.get("user"),
            history_calls=tuple(history),
        )

    @staticmethod
    def _calls(raw: Any) -> tuple[ToolCall, ...]:
        return tuple(
            ToolCall(
                call_id=str(call.get("id")),
                name=str(call["function"].get("name", "tool")),
                arguments=Arguments.parse(call["function"].get("arguments")),
                raw_arguments=str(call["function"].get("arguments") or ""),
            )
            for call in raw or []
            if isinstance(call, Mapping) and isinstance(call.get("function"), Mapping)
        )

    @staticmethod
    def _origin(role: str, message: Mapping[str, Any], call_names: Mapping[str, str]) -> str:
        if role == "tool":
            return f"tool_result:{call_names.get(str(message.get('tool_call_id')), 'tool')}"
        if role == "function":
            return f"tool_result:{message.get('name', 'function')}"
        return role

    @staticmethod
    def _tools(body: Mapping[str, Any]) -> tuple[ToolSpec, ...]:
        tools: list[ToolSpec] = []
        for tool in body.get("tools") or []:
            if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
                function = tool["function"]
                tools.append(
                    ToolSpec(
                        str(function.get("name")),
                        str(function.get("description") or ""),
                        function.get("parameters") or {},
                    )
                )
            elif tool.get("type") == "custom" and isinstance(tool.get("custom"), Mapping):
                custom = tool["custom"]
                tools.append(ToolSpec(str(custom.get("name")), str(custom.get("description") or "")))
        tools.extend(
            ToolSpec(
                str(function.get("name")), str(function.get("description") or ""), function.get("parameters") or {}
            )
            for function in body.get("functions") or []
        )
        return tuple(tools)

    @staticmethod
    def _schema(body: Mapping[str, Any]) -> Mapping[str, Any] | None:
        response_format = body.get("response_format")
        if isinstance(response_format, Mapping) and response_format.get("type") == "json_schema":
            schema = (response_format.get("json_schema") or {}).get("schema")
            return schema if isinstance(schema, Mapping) else {}
        if isinstance(response_format, Mapping) and response_format.get("type") == "json_object":
            return {}
        return None

    def parse_response(self, body: Mapping[str, Any]) -> Reply:
        choices = body.get("choices") or [{}]
        choice = choices[0] if isinstance(choices[0], Mapping) else {}
        message = choice.get("message") or {}
        text = "".join(text for _, text in TextParts.walk(message.get("content"), ()))
        refusal = message.get("refusal")
        calls = self._calls(message.get("tool_calls"))
        return Reply(
            text=text or (refusal or ""),
            tool_calls=calls,
            stop_reason=choice.get("finish_reason"),
            refused=bool(refusal),
            response_id=body.get("id"),
        )

    def assemble_stream(self, events: list[ServerSentEvent]) -> JsonBody:
        header: Mapping[str, Any] = {}
        content: list[str] = []
        refusal: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: Any = None
        for event in events:
            if event.is_done:
                continue
            chunk = event.json()
            header = header or chunk
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                if choice.get("index", 0) != 0:
                    continue
                delta = choice.get("delta") or {}
                content.append(delta.get("content") or "")
                refusal.append(delta.get("refusal") or "")
                for piece in delta.get("tool_calls") or []:
                    self._merge_call(calls.setdefault(int(piece.get("index", 0)), self._empty_call()), piece)
                finish_reason = choice.get("finish_reason") or finish_reason
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content) or None,
            "refusal": "".join(refusal) or None,
        }
        if calls:
            message["tool_calls"] = [calls[index] for index in sorted(calls)]
        return {
            "id": header.get("id", ""),
            "object": "chat.completion",
            "created": header.get("created", int(time.time())),
            "model": header.get("model", ""),
            "system_fingerprint": header.get("system_fingerprint"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason or "stop", "logprobs": None}],
            "usage": usage,
        }

    @staticmethod
    def _empty_call() -> dict[str, Any]:
        return {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}

    @staticmethod
    def _merge_call(slot: dict[str, Any], piece: Mapping[str, Any]) -> None:
        slot["id"] = piece.get("id") or slot["id"]
        function = piece.get("function") or {}
        slot["function"]["name"] += function.get("name") or ""
        slot["function"]["arguments"] += function.get("arguments") or ""

    def render_response(self, body: Mapping[str, Any], edit: ReplyEdit) -> JsonBody:
        rendered = copy.deepcopy(dict(body))
        choice = rendered["choices"][0]
        message = choice.setdefault("message", {"role": "assistant"})
        remaining = [call for call in message.get("tool_calls") or [] if call.get("id") not in edit.removed_calls]
        if edit.refusal:
            message.update(content=None, refusal=edit.text or "")
            remaining = []
        elif edit.text is not None:
            message["content"] = edit.text
        if remaining:
            message["tool_calls"] = remaining
        else:
            message.pop("tool_calls", None)
            if choice.get("finish_reason") == "tool_calls":
                choice["finish_reason"] = "stop"
        return rendered

    def synthesize_response(self, conversation: Conversation, text: str, response_id: str, refusal: bool) -> JsonBody:
        if refusal:
            completion = self.completion(conversation.model, "", (), response_id)
            completion["choices"][0]["message"].update(content=None, refusal=text)
            return completion
        return self.completion(conversation.model, text, (), response_id)

    @staticmethod
    def completion(model: str, text: str, calls: Sequence[ToolCall], response_id: str) -> JsonBody:
        message: dict[str, Any] = {"role": "assistant", "content": text or None, "refusal": None}
        if calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
                }
                for call in calls
            ]
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {"index": 0, "message": message, "logprobs": None, "finish_reason": "tool_calls" if calls else "stop"}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def stream_from_body(self, body: Mapping[str, Any]) -> list[ServerSentEvent]:
        choice = body["choices"][0]
        message = choice.get("message") or {}
        frame = {
            "id": body.get("id", ""),
            "object": "chat.completion.chunk",
            "created": body.get("created", 0),
            "model": body.get("model", ""),
        }
        deltas: list[dict[str, Any]] = [{"role": "assistant", "content": ""}]
        if message.get("content"):
            deltas.append({"content": message["content"]})
        if message.get("refusal"):
            deltas.append({"refusal": message["refusal"]})
        deltas.extend(
            {"tool_calls": [{"index": index, "id": call["id"], "type": "function", "function": call["function"]}]}
            for index, call in enumerate(message.get("tool_calls") or [])
        )
        events = [
            ServerSentEvent.of({**frame, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            for delta in deltas
        ]
        events.append(
            ServerSentEvent.of(
                {**frame, "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}]}
            )
        )
        if body.get("usage"):
            events.append(ServerSentEvent.of({**frame, "choices": [], "usage": body["usage"]}))
        events.append(ServerSentEvent.done())
        return events

    def classify_event(self, event: ServerSentEvent, cursor: StreamCursor) -> StreamPart:
        if event.is_done:
            return StreamPart(StreamKind.HOLD)
        choices = event.json().get("choices") or []
        if not choices:
            return StreamPart(StreamKind.HOLD)
        choice = choices[0]
        delta = choice.get("delta") or {}
        if choice.get("finish_reason") or delta.get("tool_calls") or delta.get("refusal") or delta.get("function_call"):
            return StreamPart(StreamKind.HOLD)
        content = delta.get("content")
        if isinstance(content, str) and content:
            cursor.template = cursor.template or event
            return StreamPart(StreamKind.TEXT, content)
        return StreamPart(StreamKind.FORWARD)

    def text_event(self, cursor: StreamCursor, text: str) -> ServerSentEvent:
        assert cursor.template is not None
        payload = cursor.template.json()
        choice = payload["choices"][0]
        payload["choices"] = [{"index": choice.get("index", 0), "delta": {"content": text}, "finish_reason": None}]
        return ServerSentEvent.of(payload)

    def continuation(self, body: Mapping[str, Any], remainder: str, cursor: StreamCursor) -> list[ServerSentEvent]:
        tail = copy.deepcopy(dict(body))
        tail["choices"][0]["message"]["content"] = remainder or None
        return self.stream_from_body(tail)[1:]

    def append_operator_note(self, document: JsonDocument, note: str) -> bool:
        for message in document.body.get("messages") or []:
            if message.get("role") not in ("system", "developer"):
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content + note
                return True
            if isinstance(content, list):
                content.append({"type": "text", "text": note})
                return True
        return False
