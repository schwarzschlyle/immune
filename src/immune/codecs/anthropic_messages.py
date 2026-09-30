from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from immune.codecs.base import (
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
from immune.core.conversation import (
    Channel,
    Conversation,
    Locator,
    PastCall,
    Reply,
    Segment,
    ToolCall,
    ToolSpec,
)
from immune.core.documents import JsonDocument
from immune.core.sse import ServerSentEvent

_REASONING_BLOCKS = frozenset({"thinking", "redacted_thinking"})
_ANTHROPIC_ROLES = frozenset({"user", "assistant"})


class AnthropicMessagesCodec(Codec):
    name: ClassVar[str] = "anthropic_messages"
    response_prefix: ClassVar[str] = "msg_immune_"

    def accepts(self, context: RequestContext) -> bool:
        return isinstance(context.body.get("messages"), list) and "max_tokens" in context.body

    def strictly_accepts(self, context: RequestContext) -> bool:
        body = context.body
        return (
            isinstance(body.get("model"), str)
            and isinstance(body.get("max_tokens"), int)
            and Shapes.messages(body.get("messages"), _ANTHROPIC_ROLES)
        )

    def parse_request(self, context: RequestContext) -> Conversation:
        body = context.body
        segments = list(self._system_segments(body.get("system"), ("system",)))
        call_names: dict[str, str] = {}
        history: list[PastCall] = []
        turn = 0
        for index, message in enumerate(body.get("messages") or []):
            role = message.get("role")
            if role == "user" and self._is_human(message.get("content")):
                turn += 1
            base: Locator = ("messages", index, "content")
            if role == "system":
                segments.extend(self._system_segments(message.get("content"), base))
            elif role == "assistant":
                history.extend(PastCall(call, turn) for call in self._tool_uses(message.get("content")))
                segments.extend(self._assistant_segments(message.get("content"), base, turn, call_names))
            elif role == "user":
                segments.extend(self._user_segments(message.get("content"), base, turn, call_names))
        return Conversation(
            provider=self.name,
            model=str(body.get("model", "")),
            segments=tuple(segments),
            tools=self._tools(body),
            response_schema=self._schema(body),
            stream=bool(body.get("stream")),
            user_hint=(body.get("metadata") or {}).get("user_id"),
            history_calls=tuple(history),
        )

    @staticmethod
    def _is_human(content: Any) -> bool:
        if isinstance(content, str):
            return True
        return any(isinstance(block, Mapping) and block.get("type") == "text" for block in content or [])

    @staticmethod
    def _tool_uses(content: Any) -> Iterator[ToolCall]:
        if not isinstance(content, list):
            return
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                arguments = block.get("input") or {}
                yield ToolCall(
                    call_id=str(block.get("id")),
                    name=str(block.get("name", "tool")),
                    arguments=dict(arguments) if isinstance(arguments, Mapping) else {"_value": arguments},
                    raw_arguments=json.dumps(arguments),
                )

    @staticmethod
    def _system_segments(content: Any, base: Locator) -> Iterator[Segment]:
        for locator, text in TextParts.walk(content, base):
            yield Segment(Channel.OPERATOR, text, "system", locator)

    @staticmethod
    def _assistant_segments(content: Any, base: Locator, turn: int, call_names: dict[str, str]) -> Iterator[Segment]:
        if isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "tool_use":
                    call_names[str(block.get("id"))] = str(block.get("name", "tool"))
        for locator, text in TextParts.walk(content, base):
            yield Segment(Channel.ASSISTANT, text, "assistant", locator, turn)

    @staticmethod
    def _user_segments(content: Any, base: Locator, turn: int, call_names: Mapping[str, str]) -> Iterator[Segment]:
        if isinstance(content, str):
            yield Segment(Channel.USER, content, "user", base, turn)
            return
        for index, block in enumerate(content or []):
            if not isinstance(block, Mapping):
                continue
            kind = block.get("type")
            locator: Locator = (*base, index)
            if kind == "text":
                yield Segment(Channel.USER, str(block.get("text", "")), "user", (*locator, "text"), turn)
            elif kind == "tool_result":
                origin = f"tool_result:{call_names.get(str(block.get('tool_use_id')), 'tool')}"
                for inner, text in TextParts.walk(block.get("content"), (*locator, "content")):
                    yield Segment(Channel.DATA, text, origin, inner, turn)
            elif kind == "document" and (block.get("source") or {}).get("type") == "text":
                yield Segment(
                    Channel.DATA, str(block["source"].get("data", "")), "document", (*locator, "source", "data"), turn
                )

    @staticmethod
    def _tools(body: Mapping[str, Any]) -> tuple[ToolSpec, ...]:
        return tuple(
            ToolSpec(
                str(tool.get("name")),
                str(tool.get("description") or ""),
                tool.get("input_schema") or {},
                builtin="input_schema" not in tool,
            )
            for tool in body.get("tools") or []
            if isinstance(tool, Mapping)
        )

    @staticmethod
    def _schema(body: Mapping[str, Any]) -> Mapping[str, Any] | None:
        output_config = body.get("output_config") or {}
        response_format = output_config.get("format") or body.get("output_format")
        if isinstance(response_format, Mapping) and response_format.get("type") == "json_schema":
            schema = response_format.get("schema")
            return schema if isinstance(schema, Mapping) else {}
        return None

    def parse_response(self, body: Mapping[str, Any]) -> Reply:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in body.get("content") or []:
            if block.get("type") == "text":
                texts.append(str(block.get("text", "")))
            elif block.get("type") == "tool_use":
                arguments = block.get("input") or {}
                calls.append(
                    ToolCall(
                        call_id=str(block.get("id")),
                        name=str(block.get("name")),
                        arguments=dict(arguments),
                        raw_arguments=json.dumps(arguments),
                    )
                )
        stop_reason = body.get("stop_reason")
        return Reply(
            text="".join(texts),
            tool_calls=tuple(calls),
            stop_reason=stop_reason,
            refused=stop_reason == "refusal",
            response_id=body.get("id"),
        )

    def assemble_stream(self, events: list[ServerSentEvent]) -> JsonBody:
        message: dict[str, Any] = {}
        blocks: dict[int, dict[str, Any]] = {}
        partial_inputs: dict[int, str] = {}
        for event in events:
            payload = event.json()
            kind = payload.get("type")
            if kind == "message_start":
                message = dict(payload.get("message") or {})
            elif kind == "content_block_start":
                blocks[int(payload["index"])] = dict(payload.get("content_block") or {})
            elif kind == "content_block_delta":
                self._apply_delta(blocks, partial_inputs, int(payload["index"]), payload.get("delta") or {})
            elif kind == "message_delta":
                message.update(payload.get("delta") or {})
                message["usage"] = {**(message.get("usage") or {}), **(payload.get("usage") or {})}
        for index, raw in partial_inputs.items():
            blocks[index]["input"] = json.loads(raw) if raw.strip() else {}
        message["content"] = [blocks[index] for index in sorted(blocks)]
        return message

    @staticmethod
    def _apply_delta(
        blocks: dict[int, dict[str, Any]], partial_inputs: dict[int, str], index: int, delta: Mapping[str, Any]
    ) -> None:
        block = blocks.setdefault(index, {})
        kind = delta.get("type")
        if kind == "text_delta":
            block["text"] = block.get("text", "") + str(delta.get("text", ""))
        elif kind == "input_json_delta":
            partial_inputs[index] = partial_inputs.get(index, "") + str(delta.get("partial_json", ""))
        elif kind == "thinking_delta":
            block["thinking"] = block.get("thinking", "") + str(delta.get("thinking", ""))
        elif kind == "signature_delta":
            block["signature"] = str(delta.get("signature", ""))

    def render_response(self, body: Mapping[str, Any], edit: ReplyEdit) -> JsonBody:
        rendered = copy.deepcopy(dict(body))
        blocks = [
            block
            for block in rendered.get("content") or []
            if not (block.get("type") == "tool_use" and block.get("id") in edit.removed_calls)
        ]
        if edit.refusal:
            blocks = [block for block in blocks if block.get("type") != "tool_use"]
        if edit.text is not None and edit.keep_reasoning and not edit.refusal:
            texts = [index for index, block in enumerate(blocks) if block.get("type") == "text"]
            if texts:
                blocks[texts[0]] = {"type": "text", "text": edit.text}
                blocks = [block for index, block in enumerate(blocks) if index not in texts[1:]]
            else:
                blocks.append({"type": "text", "text": edit.text})
        elif edit.refusal or edit.text is not None:
            position = next(
                (
                    index
                    for index, block in enumerate(blocks)
                    if block.get("type") == "text" or block.get("type") in _REASONING_BLOCKS
                ),
                0,
            )
            blocks = [
                block for block in blocks if block.get("type") != "text" and block.get("type") not in _REASONING_BLOCKS
            ]
            blocks.insert(min(position, len(blocks)), {"type": "text", "text": edit.text or ""})
        rendered["content"] = blocks
        if edit.refusal:
            rendered["stop_reason"] = "refusal"
        elif rendered.get("stop_reason") == "tool_use" and not any(b.get("type") == "tool_use" for b in blocks):
            rendered["stop_reason"] = "end_turn"
        return rendered

    def synthesize_response(self, conversation: Conversation, text: str, response_id: str, refusal: bool) -> JsonBody:
        return {
            "id": response_id,
            "type": "message",
            "role": "assistant",
            "model": conversation.model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "refusal" if refusal else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def classify_event(self, event: ServerSentEvent, cursor: StreamCursor) -> StreamPart:
        payload = event.json()
        kind = payload.get("type")
        index = int(payload.get("index", -1))
        if kind in ("message_start", "ping"):
            return StreamPart(StreamKind.FORWARD)
        if kind == "content_block_start":
            block_type = (payload.get("content_block") or {}).get("type")
            if block_type == "text" and cursor.text_index is None:
                cursor.text_index = index
                return StreamPart(StreamKind.FORWARD)
            if block_type in _REASONING_BLOCKS and cursor.text_index is None:
                return StreamPart(StreamKind.FORWARD)
            return StreamPart(StreamKind.HOLD)
        if kind == "content_block_delta":
            delta = payload.get("delta") or {}
            if index == cursor.text_index and delta.get("type") == "text_delta":
                return StreamPart(StreamKind.TEXT, str(delta.get("text", "")))
            if cursor.text_index is None and delta.get("type") in ("thinking_delta", "signature_delta"):
                return StreamPart(StreamKind.FORWARD)
            return StreamPart(StreamKind.HOLD)
        if kind == "content_block_stop" and cursor.text_index is None:
            cursor.passed.add(index)
            return StreamPart(StreamKind.FORWARD)
        return StreamPart(StreamKind.HOLD)

    def text_event(self, cursor: StreamCursor, text: str) -> ServerSentEvent:
        payload = {
            "type": "content_block_delta",
            "index": cursor.text_index,
            "delta": {"type": "text_delta", "text": text},
        }
        return ServerSentEvent.of(payload, "content_block_delta")

    def continuation(self, body: Mapping[str, Any], remainder: str, cursor: StreamCursor) -> list[ServerSentEvent]:
        events: list[ServerSentEvent] = []
        for event in self.stream_from_body(body)[1:]:
            payload = event.json()
            index = payload.get("index")
            if index is None or index in cursor.passed:
                if index is None:
                    events.append(event)
                continue
            if index != cursor.text_index:
                events.append(event)
                continue
            if payload["type"] == "content_block_delta" and remainder:
                payload["delta"] = {"type": "text_delta", "text": remainder}
                events.append(ServerSentEvent.of(payload, event.event))
            elif payload["type"] == "content_block_stop":
                events.append(event)
        return events

    def stream_from_body(self, body: Mapping[str, Any]) -> list[ServerSentEvent]:
        usage = body.get("usage") or {"input_tokens": 0, "output_tokens": 0}
        opening = {**body, "content": [], "stop_reason": None, "stop_sequence": None, "usage": usage}
        events = [ServerSentEvent.of({"type": "message_start", "message": opening}, "message_start")]
        for index, block in enumerate(body.get("content") or []):
            events.extend(self._block_events(index, block))
        events.append(
            ServerSentEvent.of(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": body.get("stop_reason"), "stop_sequence": None},
                    "usage": {"output_tokens": usage.get("output_tokens", 0)},
                },
                "message_delta",
            )
        )
        events.append(ServerSentEvent.of({"type": "message_stop"}, "message_stop"))
        return events

    @staticmethod
    def _block_events(index: int, block: Mapping[str, Any]) -> list[ServerSentEvent]:
        kind = block.get("type")
        deltas: list[dict[str, Any]] = []
        if kind == "text":
            opening: dict[str, Any] = {"type": "text", "text": ""}
            deltas.append({"type": "text_delta", "text": block.get("text", "")})
        elif kind == "tool_use":
            opening = {**block, "input": {}}
            deltas.append({"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})})
        elif kind == "thinking":
            opening = {"type": "thinking", "thinking": "", "signature": ""}
            deltas.append({"type": "thinking_delta", "thinking": block.get("thinking", "")})
            deltas.append({"type": "signature_delta", "signature": block.get("signature", "")})
        else:
            opening = dict(block)
        events = [
            ServerSentEvent.of(
                {"type": "content_block_start", "index": index, "content_block": opening}, "content_block_start"
            )
        ]
        events.extend(
            ServerSentEvent.of({"type": "content_block_delta", "index": index, "delta": delta}, "content_block_delta")
            for delta in deltas
        )
        events.append(ServerSentEvent.of({"type": "content_block_stop", "index": index}, "content_block_stop"))
        return events

    def append_operator_note(self, document: JsonDocument, note: str) -> bool:
        system = document.body.get("system")
        if isinstance(system, str):
            document.body["system"] = system + note
            return True
        if isinstance(system, list):
            system.append({"type": "text", "text": note})
            return True
        return False
