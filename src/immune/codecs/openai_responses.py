from __future__ import annotations

import copy
import json
import time
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from immune.codecs.base import (
    Arguments,
    Codec,
    JsonBody,
    ReplyEdit,
    RequestContext,
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
}
_OUTPUT_TYPES = frozenset({"function_call_output", "custom_tool_call_output"})
_TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")


class OpenAIResponsesCodec(Codec):
    name: ClassVar[str] = "openai_responses"
    response_prefix: ClassVar[str] = "resp_immune_"
    server_side_history: ClassVar[bool] = True

    def accepts(self, context: RequestContext) -> bool:
        return "input" in context.body or "instructions" in context.body

    def strictly_accepts(self, context: RequestContext) -> bool:
        body = context.body
        items = body.get("input")
        well_formed = isinstance(items, str) or (
            isinstance(items, list)
            and all(isinstance(item, Mapping) and ("type" in item or "role" in item) for item in items)
        )
        return isinstance(body.get("model"), str) and well_formed

    def parse_request(self, context: RequestContext) -> Conversation:
        body = context.body
        segments: list[Segment] = []
        if isinstance(body.get("instructions"), str):
            segments.append(Segment(Channel.OPERATOR, body["instructions"], "instructions", ("instructions",)))
        segments.extend(self._input_segments(body.get("input")))
        return Conversation(
            provider=self.name,
            model=str(body.get("model", "")),
            segments=tuple(segments),
            tools=self._tools(body),
            response_schema=self._schema(body),
            stream=bool(body.get("stream")),
            user_hint=body.get("safety_identifier") or body.get("user"),
            previous_response_id=body.get("previous_response_id"),
            history_calls=tuple(self._history_calls(body.get("input"))),
        )

    def _input_segments(self, items: Any) -> Iterator[Segment]:
        if isinstance(items, str):
            yield Segment(Channel.USER, items, "user", ("input",), turn=1)
            return
        call_names: dict[str, str] = {}
        turn = 0
        for index, item in enumerate(items or []):
            if not isinstance(item, Mapping):
                continue
            kind = item.get("type", "message")
            if kind == "function_call":
                call_names[str(item.get("call_id"))] = str(item.get("name", "tool"))
            elif kind in _OUTPUT_TYPES:
                origin = f"tool_result:{call_names.get(str(item.get('call_id')), 'tool')}"
                for locator, text in TextParts.walk(item.get("output"), ("input", index, "output")):
                    yield Segment(Channel.DATA, text, origin, locator, turn)
            elif kind == "message" and item.get("role") in _ROLE_CHANNELS:
                channel = _ROLE_CHANNELS[str(item["role"])]
                if channel is Channel.USER:
                    turn += 1
                for locator, text in TextParts.walk(item.get("content"), ("input", index, "content")):
                    yield Segment(channel, text, str(item["role"]), locator, turn)

    @staticmethod
    def _history_calls(items: Any) -> Iterator[PastCall]:
        if not isinstance(items, list):
            return
        turn = 0
        for item in items:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("type", "message")
            if kind == "message" and item.get("role") == "user":
                turn += 1
            elif kind == "function_call":
                call = ToolCall(
                    call_id=str(item.get("call_id")),
                    name=str(item.get("name", "tool")),
                    arguments=Arguments.parse(item.get("arguments")),
                    raw_arguments=str(item.get("arguments") or ""),
                )
                yield PastCall(call, turn)

    @staticmethod
    def _tools(body: Mapping[str, Any]) -> tuple[ToolSpec, ...]:
        tools: list[ToolSpec] = []
        for tool in body.get("tools") or []:
            if tool.get("type") in ("function", "custom"):
                tools.append(
                    ToolSpec(str(tool.get("name")), str(tool.get("description") or ""), tool.get("parameters") or {})
                )
            else:
                label = tool.get("server_label") or tool.get("name") or tool.get("type")
                tools.append(ToolSpec(str(label), str(tool.get("server_description") or ""), builtin=True))
        return tuple(tools)

    @staticmethod
    def _schema(body: Mapping[str, Any]) -> Mapping[str, Any] | None:
        text_config = body.get("text")
        response_format = text_config.get("format") if isinstance(text_config, Mapping) else None
        if isinstance(response_format, Mapping) and response_format.get("type") == "json_schema":
            schema = response_format.get("schema")
            return schema if isinstance(schema, Mapping) else {}
        if isinstance(response_format, Mapping) and response_format.get("type") == "json_object":
            return {}
        return None

    def parse_response(self, body: Mapping[str, Any]) -> Reply:
        texts: list[str] = []
        calls: list[ToolCall] = []
        refused = False
        for item in body.get("output") or []:
            if item.get("type") == "message":
                for part in item.get("content") or []:
                    if part.get("type") == "output_text":
                        texts.append(str(part.get("text", "")))
                    elif part.get("type") == "refusal":
                        refused = True
                        texts.append(str(part.get("refusal", "")))
            elif item.get("type") == "function_call":
                calls.append(
                    ToolCall(
                        call_id=str(item.get("call_id")),
                        name=str(item.get("name")),
                        arguments=Arguments.parse(item.get("arguments")),
                        raw_arguments=str(item.get("arguments") or ""),
                    )
                )
        return Reply(
            text="".join(texts),
            tool_calls=tuple(calls),
            stop_reason=body.get("status"),
            refused=refused,
            response_id=body.get("id"),
        )

    def assemble_stream(self, events: list[ServerSentEvent]) -> JsonBody:
        for event in reversed(events):
            payload = event.json()
            if payload.get("type") in _TERMINAL_EVENTS and isinstance(payload.get("response"), Mapping):
                return dict(payload["response"])
        return self._assemble_partial(events)

    @staticmethod
    def _assemble_partial(events: list[ServerSentEvent]) -> JsonBody:
        response: dict[str, Any] = {}
        items: dict[int, Any] = {}
        for event in events:
            payload = event.json()
            if isinstance(payload.get("response"), Mapping):
                response = dict(payload["response"])
            if payload.get("type") == "response.output_item.done":
                items[int(payload.get("output_index", len(items)))] = payload.get("item")
        response["output"] = [items[index] for index in sorted(items)]
        return response

    def render_response(self, body: Mapping[str, Any], edit: ReplyEdit) -> JsonBody:
        rendered = copy.deepcopy(dict(body))
        output = [
            item
            for item in rendered.get("output") or []
            if not (item.get("type") == "function_call" and item.get("call_id") in edit.removed_calls)
        ]
        if edit.refusal:
            output = [item for item in output if item.get("type") != "function_call"]
        if edit.refusal or edit.text is not None:
            part: dict[str, Any] = (
                {"type": "refusal", "refusal": edit.text or ""}
                if edit.refusal
                else {"type": "output_text", "text": edit.text, "annotations": [], "logprobs": []}
            )
            messages = [item for item in output if item.get("type") == "message"]
            if messages:
                messages[0]["content"] = [part]
                output = [item for item in output if item.get("type") != "message" or item is messages[0]]
            else:
                output.append(self._message_item(f"msg_{rendered.get('id', 'immune')}", part))
        rendered["output"] = output
        return rendered

    @staticmethod
    def _message_item(item_id: str, part: Mapping[str, Any]) -> dict[str, Any]:
        return {"id": item_id, "type": "message", "role": "assistant", "status": "completed", "content": [dict(part)]}

    def synthesize_response(self, conversation: Conversation, text: str, response_id: str, refusal: bool) -> JsonBody:
        part = (
            {"type": "refusal", "refusal": text}
            if refusal
            else {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
        )
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": conversation.model,
            "output": [self._message_item(f"msg_{response_id}", part)],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

    def classify_event(self, event: ServerSentEvent, cursor: StreamCursor) -> StreamPart:
        payload = event.json()
        kind = str(payload.get("type", ""))
        output_index = int(payload.get("output_index", -1))
        if kind == "response.output_text.delta" and output_index == cursor.text_index:
            if int(payload.get("content_index", 0)) == 0:
                cursor.template = cursor.template or event
                return StreamPart(StreamKind.TEXT, str(payload.get("delta", "")))
            return StreamPart(StreamKind.HOLD)
        if cursor.text_started:
            return StreamPart(StreamKind.HOLD)
        if kind in ("response.created", "response.in_progress") or kind.startswith("response.reasoning"):
            return StreamPart(StreamKind.FORWARD)
        item_type = (payload.get("item") or {}).get("type")
        if kind == "response.output_item.added" and item_type == "message" and cursor.text_index is None:
            cursor.text_index = output_index
            return StreamPart(StreamKind.FORWARD)
        if kind in ("response.output_item.added", "response.output_item.done") and item_type == "reasoning":
            if kind.endswith("done"):
                cursor.passed.add(output_index)
            return StreamPart(StreamKind.FORWARD)
        part_type = (payload.get("part") or {}).get("type")
        if kind == "response.content_part.added" and output_index == cursor.text_index and part_type == "output_text":
            return StreamPart(StreamKind.FORWARD)
        return StreamPart(StreamKind.HOLD)

    def text_event(self, cursor: StreamCursor, text: str) -> ServerSentEvent:
        assert cursor.template is not None
        payload = cursor.template.json()
        payload["delta"] = text
        return ServerSentEvent.of(payload, cursor.template.event)

    def continuation(self, body: Mapping[str, Any], remainder: str, cursor: StreamCursor) -> list[ServerSentEvent]:
        canonical = self.stream_from_body(body)
        opening = {"response.created", "response.in_progress"}
        if cursor.text_started:
            position = next(
                (
                    index
                    for index, event in enumerate(canonical)
                    if event.json().get("type") == "response.output_text.delta"
                    and event.json().get("output_index") == cursor.text_index
                ),
                None,
            )
            if position is not None:
                delta = canonical[position].json()
                delta["delta"] = remainder
                head = [ServerSentEvent.of(delta, canonical[position].event)] if remainder else []
                return head + canonical[position + 1 :]
        announced = {"response.output_item.added", "response.content_part.added"}
        return [
            event
            for event in canonical
            if event.json().get("type") not in opening
            and event.json().get("output_index") not in cursor.passed
            and not (event.json().get("type") in announced and event.json().get("output_index") == cursor.text_index)
        ]

    def stream_from_body(self, body: Mapping[str, Any]) -> list[ServerSentEvent]:
        emitter = _ResponseEventEmitter()
        emitter.emit("response.created", response={**body, "status": "in_progress", "output": []})
        emitter.emit("response.in_progress", response={**body, "status": "in_progress", "output": []})
        for output_index, item in enumerate(body.get("output") or []):
            emitter.emit_item(output_index, item)
        emitter.emit("response.completed", response=dict(body))
        return emitter.events

    def append_operator_note(self, document: JsonDocument, note: str) -> bool:
        if isinstance(document.body.get("instructions"), str):
            document.body["instructions"] += note
            return True
        return False


class _ResponseEventEmitter:
    def __init__(self) -> None:
        self.events: list[ServerSentEvent] = []

    def emit(self, kind: str, **fields: Any) -> None:
        self.events.append(ServerSentEvent.of({"type": kind, "sequence_number": len(self.events), **fields}, kind))

    def emit_item(self, output_index: int, item: Mapping[str, Any]) -> None:
        item_id = str(item.get("id", f"item_{output_index}"))
        self.emit("response.output_item.added", output_index=output_index, item=self._opening(item))
        if item.get("type") == "message":
            for content_index, part in enumerate(item.get("content") or []):
                self._emit_part(output_index, item_id, content_index, part)
        elif item.get("type") == "function_call":
            arguments = str(item.get("arguments", ""))
            self.emit(
                "response.function_call_arguments.delta", output_index=output_index, item_id=item_id, delta=arguments
            )
            self.emit(
                "response.function_call_arguments.done", output_index=output_index, item_id=item_id, arguments=arguments
            )
        self.emit("response.output_item.done", output_index=output_index, item=dict(item))

    def _emit_part(self, output_index: int, item_id: str, content_index: int, part: Mapping[str, Any]) -> None:
        location = {"output_index": output_index, "item_id": item_id, "content_index": content_index}
        empty = {**part, "text": ""} if part.get("type") == "output_text" else {**part, "refusal": ""}
        self.emit("response.content_part.added", **location, part=empty)
        if part.get("type") == "output_text":
            self.emit("response.output_text.delta", **location, delta=part.get("text", ""), logprobs=[])
            self.emit("response.output_text.done", **location, text=part.get("text", ""), logprobs=[])
        else:
            self.emit("response.refusal.delta", **location, delta=part.get("refusal", ""))
            self.emit("response.refusal.done", **location, refusal=part.get("refusal", ""))
        self.emit("response.content_part.done", **location, part=dict(part))

    @staticmethod
    def _opening(item: Mapping[str, Any]) -> dict[str, Any]:
        opening = json.loads(json.dumps(item))
        opening["status"] = "in_progress"
        if opening.get("type") == "message":
            opening["content"] = []
        if opening.get("type") == "function_call":
            opening["arguments"] = ""
        return dict(opening)
