from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from immune.codecs.base import (
    Codec,
    JsonBody,
    ReplyEdit,
    RequestContext,
    StreamCursor,
    StreamKind,
    StreamPart,
)
from immune.core.conversation import Channel, Conversation, PastCall, Reply, Segment, ToolCall, ToolSpec
from immune.core.documents import JsonDocument
from immune.core.sse import ServerSentEvent

_MODEL = re.compile(r"/model/(?P<model>[^/]+)/(?P<operation>converse|converse-stream)$")
_REFUSED = frozenset({"guardrail_intervened", "content_filtered"})


class BedrockConverseCodec(Codec):
    name: ClassVar[str] = "bedrock_converse"
    response_prefix: ClassVar[str] = "bedrock-immune-"

    def accepts(self, context: RequestContext) -> bool:
        return isinstance(context.body.get("messages"), list) and _MODEL.search(context.path) is not None

    def is_streaming(self, context: RequestContext) -> bool:
        match = _MODEL.search(context.path)
        return bool(match and match.group("operation") == "converse-stream")

    def parse_request(self, context: RequestContext) -> Conversation:
        body = context.body
        match = _MODEL.search(context.path)
        segments = [
            Segment(Channel.OPERATOR, block["text"], "system", ("system", index, "text"))
            for index, block in enumerate(body.get("system") or [])
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ]
        history: list[PastCall] = []
        names: dict[str, str] = {}
        turn = 0
        for index, message in enumerate(body.get("messages") or []):
            role = message.get("role")
            blocks = message.get("content") or []
            if role == "user" and any("text" in block for block in blocks):
                turn += 1
            for position, block in enumerate(blocks):
                locator = ("messages", index, "content", position)
                segments.extend(self._segments(role, block, locator, turn, names, history))
        return Conversation(
            provider=self.name,
            model=match.group("model") if match else "",
            segments=tuple(segments),
            tools=self._tools(body),
            stream=self.is_streaming(context),
            history_calls=tuple(history),
        )

    @staticmethod
    def _segments(
        role: Any,
        block: Mapping[str, Any],
        locator: tuple[str | int, ...],
        turn: int,
        names: dict[str, str],
        history: list[PastCall],
    ) -> Iterator[Segment]:
        if isinstance(block.get("text"), str):
            channel = Channel.USER if role == "user" else Channel.ASSISTANT
            yield Segment(channel, block["text"], str(role), (*locator, "text"), turn)
        elif isinstance(block.get("toolUse"), Mapping):
            use = block["toolUse"]
            names[str(use.get("toolUseId"))] = str(use.get("name", "tool"))
            call = ToolCall(str(use.get("toolUseId")), str(use.get("name", "tool")), dict(use.get("input") or {}))
            history.append(PastCall(call, turn))
        elif isinstance(block.get("toolResult"), Mapping):
            result = block["toolResult"]
            origin = f"tool_result:{names.get(str(result.get('toolUseId')), 'tool')}"
            for inner, part in enumerate(result.get("content") or []):
                base = (*locator, "toolResult", "content", inner)
                if isinstance(part.get("text"), str):
                    yield Segment(Channel.DATA, part["text"], origin, (*base, "text"), turn)
                elif "json" in part:
                    yield Segment(
                        Channel.DATA, json.dumps(part["json"], ensure_ascii=False), origin, (*base, "json"), turn
                    )

    @staticmethod
    def _tools(body: Mapping[str, Any]) -> tuple[ToolSpec, ...]:
        return tuple(
            ToolSpec(
                str(tool["toolSpec"].get("name")),
                str(tool["toolSpec"].get("description") or ""),
                (tool["toolSpec"].get("inputSchema") or {}).get("json") or {},
            )
            for tool in (body.get("toolConfig") or {}).get("tools") or []
            if isinstance(tool.get("toolSpec"), Mapping)
        )

    def parse_response(self, body: Mapping[str, Any]) -> Reply:
        blocks = ((body.get("output") or {}).get("message") or {}).get("content") or []
        texts = [block["text"] for block in blocks if isinstance(block.get("text"), str)]
        calls = tuple(
            ToolCall(
                call_id=str(block["toolUse"].get("toolUseId")),
                name=str(block["toolUse"].get("name")),
                arguments=dict(block["toolUse"].get("input") or {}),
                raw_arguments=json.dumps(block["toolUse"].get("input") or {}),
            )
            for block in blocks
            if isinstance(block.get("toolUse"), Mapping)
        )
        stop = body.get("stopReason")
        return Reply(text="".join(texts), tool_calls=calls, stop_reason=stop, refused=stop in _REFUSED)

    def render_response(self, body: Mapping[str, Any], edit: ReplyEdit) -> JsonBody:
        rendered = copy.deepcopy(dict(body))
        message = rendered.setdefault("output", {}).setdefault("message", {"role": "assistant", "content": []})
        blocks = [
            block
            for block in message.get("content") or []
            if not (
                isinstance(block.get("toolUse"), Mapping) and block["toolUse"].get("toolUseId") in edit.removed_calls
            )
        ]
        if edit.refusal:
            blocks = [block for block in blocks if "toolUse" not in block]
        if edit.refusal or edit.text is not None:
            position = next((index for index, block in enumerate(blocks) if "text" in block), 0)
            blocks = [block for block in blocks if "text" not in block]
            blocks.insert(min(position, len(blocks)), {"text": edit.text or ""})
        message["content"] = blocks
        if edit.refusal:
            rendered["stopReason"] = "guardrail_intervened"
        elif rendered.get("stopReason") == "tool_use" and not any("toolUse" in block for block in blocks):
            rendered["stopReason"] = "end_turn"
        return rendered

    def synthesize_response(self, conversation: Conversation, text: str, response_id: str, refusal: bool) -> JsonBody:
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "guardrail_intervened" if refusal else "end_turn",
            "usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
            "metrics": {"latencyMs": 0},
        }

    def assemble_stream(self, events: list[ServerSentEvent]) -> JsonBody:
        blocks: dict[int, dict[str, Any]] = {}
        partial: dict[int, str] = {}
        body: dict[str, Any] = {"output": {"message": {"role": "assistant", "content": []}}}
        for event in events:
            payload = event.json()
            if "contentBlockStart" in payload:
                start = payload["contentBlockStart"]
                use = (start.get("start") or {}).get("toolUse")
                if isinstance(use, Mapping):
                    blocks[int(start.get("contentBlockIndex", 0))] = {"toolUse": {**use, "input": {}}}
            elif "contentBlockDelta" in payload:
                delta = payload["contentBlockDelta"]
                index = int(delta.get("contentBlockIndex", 0))
                content = delta.get("delta") or {}
                if isinstance(content.get("text"), str):
                    block = blocks.setdefault(index, {"text": ""})
                    block["text"] = block.get("text", "") + content["text"]
                elif isinstance(content.get("toolUse"), Mapping):
                    partial[index] = partial.get(index, "") + str(content["toolUse"].get("input", ""))
            elif "messageStop" in payload:
                body["stopReason"] = payload["messageStop"].get("stopReason")
            elif "metadata" in payload:
                body.update({key: value for key, value in payload["metadata"].items() if key in ("usage", "metrics")})
        for index, raw in partial.items():
            if index in blocks and "toolUse" in blocks[index]:
                blocks[index]["toolUse"]["input"] = json.loads(raw) if raw.strip() else {}
        body["output"]["message"]["content"] = [blocks[index] for index in sorted(blocks)]
        return body

    def stream_from_body(self, body: Mapping[str, Any]) -> list[ServerSentEvent]:
        events = [ServerSentEvent.of({"messageStart": {"role": "assistant"}})]
        for index, block in enumerate(((body.get("output") or {}).get("message") or {}).get("content") or []):
            events.extend(self._block_events(index, block))
        events.append(ServerSentEvent.of({"messageStop": {"stopReason": body.get("stopReason", "end_turn")}}))
        metadata = {key: body[key] for key in ("usage", "metrics") if key in body}
        if metadata:
            events.append(ServerSentEvent.of({"metadata": metadata}))
        return events

    @staticmethod
    def _block_events(index: int, block: Mapping[str, Any]) -> list[ServerSentEvent]:
        if isinstance(block.get("toolUse"), Mapping):
            use = block["toolUse"]
            start = {"toolUse": {"toolUseId": use.get("toolUseId"), "name": use.get("name")}}
            delta = {"toolUse": {"input": json.dumps(use.get("input") or {})}}
            return [
                ServerSentEvent.of({"contentBlockStart": {"start": start, "contentBlockIndex": index}}),
                ServerSentEvent.of({"contentBlockDelta": {"delta": delta, "contentBlockIndex": index}}),
                ServerSentEvent.of({"contentBlockStop": {"contentBlockIndex": index}}),
            ]
        text = str(block.get("text", ""))
        return [
            ServerSentEvent.of({"contentBlockDelta": {"delta": {"text": text}, "contentBlockIndex": index}}),
            ServerSentEvent.of({"contentBlockStop": {"contentBlockIndex": index}}),
        ]

    def classify_event(self, event: ServerSentEvent, cursor: StreamCursor) -> StreamPart:
        payload = event.json()
        if "messageStart" in payload:
            return StreamPart(StreamKind.FORWARD)
        delta = payload.get("contentBlockDelta")
        if isinstance(delta, Mapping):
            index = int(delta.get("contentBlockIndex", 0))
            content = delta.get("delta") or {}
            if isinstance(content.get("text"), str) and cursor.text_index in (None, index):
                cursor.text_index = index
                return StreamPart(StreamKind.TEXT, content["text"])
            if "reasoningContent" in content and cursor.text_index is None:
                return StreamPart(StreamKind.FORWARD)
        return StreamPart(StreamKind.HOLD)

    def text_event(self, cursor: StreamCursor, text: str) -> ServerSentEvent:
        return ServerSentEvent.of(
            {"contentBlockDelta": {"delta": {"text": text}, "contentBlockIndex": cursor.text_index}}
        )

    def continuation(self, body: Mapping[str, Any], remainder: str, cursor: StreamCursor) -> list[ServerSentEvent]:
        events: list[ServerSentEvent] = []
        for event in self.stream_from_body(body)[1:]:
            payload = event.json()
            block = (
                payload.get("contentBlockDelta") or payload.get("contentBlockStop") or payload.get("contentBlockStart")
            )
            index = block.get("contentBlockIndex") if isinstance(block, Mapping) else None
            if index is None or index != cursor.text_index:
                events.append(event)
            elif "contentBlockDelta" in payload:
                if remainder:
                    events.append(self.text_event(cursor, remainder))
            else:
                events.append(event)
        return events

    def append_operator_note(self, document: JsonDocument, note: str) -> bool:
        system = document.body.get("system")
        if isinstance(system, list):
            system.append({"text": note})
            return True
        return False
