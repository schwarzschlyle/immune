from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from immune.codecs.base import Codec, JsonBody, ReplyEdit, RequestContext, StreamCursor, StreamKind, StreamPart
from immune.core.conversation import Channel, Conversation, PastCall, Reply, Segment, ToolCall, ToolSpec
from immune.core.documents import JsonDocument, TextEdit
from immune.core.sse import ServerSentEvent

_MODEL_IN_PATH = re.compile(r"models/(?P<model>[^/:]+):(?P<method>generateContent|streamGenerateContent)")


class GeminiCodec(Codec):
    name: ClassVar[str] = "gemini"

    def accepts(self, context: RequestContext) -> bool:
        return isinstance(context.body.get("contents"), list)

    def strictly_accepts(self, context: RequestContext) -> bool:
        contents = context.body.get("contents")
        return (
            _MODEL_IN_PATH.search(context.path) is not None
            and isinstance(contents, list)
            and all(isinstance(item, Mapping) and isinstance(item.get("parts"), list) for item in contents)
        )

    def is_streaming(self, context: RequestContext) -> bool:
        match = _MODEL_IN_PATH.search(context.path)
        return bool(match and match.group("method") == "streamGenerateContent")

    def parse_request(self, context: RequestContext) -> Conversation:
        body = context.body
        match = _MODEL_IN_PATH.search(context.path)
        segments: list[Segment] = []
        system_key = "systemInstruction" if "systemInstruction" in body else "system_instruction"
        for index, part in enumerate((body.get(system_key) or {}).get("parts") or []):
            if isinstance(part.get("text"), str):
                segments.append(Segment(Channel.OPERATOR, part["text"], "system", (system_key, "parts", index, "text")))
        turn = 0
        history: list[PastCall] = []
        for index, content in enumerate(body.get("contents") or []):
            role = content.get("role", "user")
            parts = content.get("parts") or []
            if role == "user" and any("text" in part for part in parts):
                turn += 1
            channel = Channel.USER if role == "user" else Channel.ASSISTANT
            for part_index, part in enumerate(parts):
                locator = ("contents", index, "parts", part_index)
                if isinstance(part.get("text"), str) and not part.get("thought"):
                    segments.append(Segment(channel, part["text"], role, (*locator, "text"), turn))
                elif isinstance(part.get("functionCall"), Mapping):
                    call = part["functionCall"]
                    arguments = dict(call.get("args") or {})
                    history.append(
                        PastCall(
                            ToolCall(
                                str(call.get("id") or f"past_{index}_{part_index}"), str(call.get("name")), arguments
                            ),
                            turn,
                        )
                    )
                elif isinstance(part.get("functionResponse"), Mapping):
                    response = part["functionResponse"]
                    segments.append(
                        Segment(
                            Channel.DATA,
                            json.dumps(response.get("response"), ensure_ascii=False),
                            f"tool_result:{response.get('name', 'tool')}",
                            (*locator, "functionResponse", "response"),
                            turn,
                        )
                    )
        return Conversation(
            provider=self.name,
            model=match.group("model") if match else "",
            segments=tuple(segments),
            tools=self._tools(body),
            response_schema=self._schema(body),
            stream=self.is_streaming(context),
            history_calls=tuple(history),
        )

    @staticmethod
    def _tools(body: Mapping[str, Any]) -> tuple[ToolSpec, ...]:
        tools: list[ToolSpec] = []
        for tool in body.get("tools") or []:
            declarations = tool.get("functionDeclarations") or tool.get("function_declarations") or []
            tools.extend(
                ToolSpec(
                    str(item.get("name")),
                    str(item.get("description") or ""),
                    item.get("parameters") or item.get("parametersJsonSchema") or {},
                )
                for item in declarations
            )
            tools.extend(
                ToolSpec(str(key), builtin=True)
                for key in tool
                if key not in ("functionDeclarations", "function_declarations")
            )
        return tuple(tools)

    @staticmethod
    def _schema(body: Mapping[str, Any]) -> Mapping[str, Any] | None:
        config = body.get("generationConfig") or body.get("generation_config") or {}
        schema = config.get("responseJsonSchema") or config.get("responseSchema")
        if isinstance(schema, Mapping):
            return schema
        return {} if config.get("responseMimeType") == "application/json" else None

    def parse_response(self, body: Mapping[str, Any]) -> Reply:
        candidate = (body.get("candidates") or [{}])[0]
        texts: list[str] = []
        calls: list[ToolCall] = []
        for index, part in enumerate((candidate.get("content") or {}).get("parts") or []):
            if isinstance(part.get("text"), str) and not part.get("thought"):
                texts.append(part["text"])
            elif isinstance(part.get("functionCall"), Mapping):
                call = part["functionCall"]
                arguments = dict(call.get("args") or {})
                calls.append(
                    ToolCall(
                        call_id=str(call.get("id") or f"call_{index}"),
                        name=str(call.get("name")),
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments),
                    )
                )
        finish_reason = candidate.get("finishReason")
        refused = finish_reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII")
        return Reply(
            text="".join(texts),
            tool_calls=tuple(calls),
            stop_reason=finish_reason,
            refused=refused,
            response_id=body.get("responseId"),
        )

    def assemble_stream(self, events: list[ServerSentEvent]) -> JsonBody:
        chunks = [event.json() for event in events if not event.is_done]
        if not chunks:
            return {"candidates": []}
        merged = copy.deepcopy(chunks[-1])
        parts: list[dict[str, Any]] = []
        for chunk in chunks:
            candidate = (chunk.get("candidates") or [{}])[0]
            for part in (candidate.get("content") or {}).get("parts") or []:
                self._merge_part(parts, part)
        candidates = merged.setdefault("candidates", [{}])
        candidates[0]["content"] = {"role": "model", "parts": parts}
        return dict(merged)

    @staticmethod
    def _merge_part(parts: list[dict[str, Any]], part: Mapping[str, Any]) -> None:
        mergeable = "text" in part and not part.get("thought")
        if mergeable and parts and "text" in parts[-1] and not parts[-1].get("thought"):
            parts[-1]["text"] += part["text"]
        else:
            parts.append(dict(part))

    def render_response(self, body: Mapping[str, Any], edit: ReplyEdit) -> JsonBody:
        rendered = copy.deepcopy(dict(body))
        candidate = rendered.setdefault("candidates", [{}])[0]
        content = candidate.setdefault("content", {"role": "model", "parts": []})
        kept: list[dict[str, Any]] = []
        for index, part in enumerate(content.get("parts") or []):
            call = part.get("functionCall")
            if isinstance(call, Mapping):
                call_id = str(call.get("id") or f"call_{index}")
                if edit.refusal or call_id in edit.removed_calls:
                    continue
            elif "text" in part and (edit.refusal or edit.text is not None):
                continue
            kept.append(part)
        if edit.refusal or edit.text is not None:
            kept.insert(0, {"text": edit.text or ""})
        content["parts"] = kept
        if not any("functionCall" in part for part in kept):
            candidate["finishReason"] = candidate.get("finishReason") or "STOP"
        return rendered

    def synthesize_response(self, conversation: Conversation, text: str, response_id: str, refusal: bool) -> JsonBody:
        return {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 0, "candidatesTokenCount": 0, "totalTokenCount": 0},
            "modelVersion": conversation.model,
            "responseId": response_id,
        }

    def classify_event(self, event: ServerSentEvent, cursor: StreamCursor) -> StreamPart:
        candidate = (event.json().get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        if candidate.get("finishReason") or any("functionCall" in part for part in parts):
            return StreamPart(StreamKind.HOLD)
        texts = [part["text"] for part in parts if isinstance(part.get("text"), str) and not part.get("thought")]
        thoughts = [part for part in parts if part.get("thought")]
        if thoughts and not texts and not cursor.text_started:
            return StreamPart(StreamKind.FORWARD)
        if texts and not thoughts:
            cursor.template = cursor.template or event
            return StreamPart(StreamKind.TEXT, "".join(texts))
        return StreamPart(StreamKind.HOLD)

    def text_event(self, cursor: StreamCursor, text: str) -> ServerSentEvent:
        assert cursor.template is not None
        payload = cursor.template.json()
        candidate = payload["candidates"][0]
        candidate["content"] = {"role": "model", "parts": [{"text": text}]}
        candidate.pop("finishReason", None)
        payload.pop("usageMetadata", None)
        return ServerSentEvent.of(payload)

    def continuation(self, body: Mapping[str, Any], remainder: str, cursor: StreamCursor) -> list[ServerSentEvent]:
        tail = copy.deepcopy(dict(body))
        candidate = (tail.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        kept = [part for part in parts if "text" not in part]
        candidate["content"] = {"role": "model", "parts": ([{"text": remainder}] if remainder else []) + kept}
        return [ServerSentEvent.of(tail)]

    def stream_from_body(self, body: Mapping[str, Any]) -> list[ServerSentEvent]:
        final = copy.deepcopy(dict(body))
        candidate = (final.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        texts = [part for part in parts if "text" in part]
        if not texts:
            return [ServerSentEvent.of(final)]
        opening = {key: value for key, value in final.items() if key != "usageMetadata"}
        opening["candidates"] = [{"content": {"role": "model", "parts": texts}, "index": candidate.get("index", 0)}]
        candidate["content"] = {"role": "model", "parts": [part for part in parts if "text" not in part]}
        return [ServerSentEvent.of(opening), ServerSentEvent.of(final)]

    def append_operator_note(self, document: JsonDocument, note: str) -> bool:
        for key in ("systemInstruction", "system_instruction"):
            instruction = document.body.get(key)
            if isinstance(instruction, Mapping) and isinstance(instruction.get("parts"), list):
                instruction["parts"].append({"text": note})
                return True
        return False

    def apply_edits(self, document: JsonDocument, edits: Iterable[TextEdit]) -> None:
        textual: list[TextEdit] = []
        for edit in edits:
            if isinstance(document.get(edit.locator), str):
                textual.append(edit)
            else:
                document.set(edit.locator, {"content": edit.replacement})
        document.apply(textual)
