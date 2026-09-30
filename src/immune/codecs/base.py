from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from immune.core.conversation import Conversation, Locator, Reply
from immune.core.documents import JsonDocument, TextEdit
from immune.core.sse import ServerSentEvent

JsonBody = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReplyEdit:
    text: str | None = None
    removed_calls: frozenset[str] = field(default_factory=frozenset)
    refusal: bool = False
    replaced: bool = False
    appendix: str = ""
    keep_reasoning: bool = False

    @property
    def is_noop(self) -> bool:
        return self.text is None and not self.removed_calls and not self.refusal


class StreamKind(Enum):
    FORWARD = "forward"
    TEXT = "text"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class StreamPart:
    kind: StreamKind
    text: str = ""


class StreamCursor:
    def __init__(self) -> None:
        self.text_started = False
        self.template: ServerSentEvent | None = None
        self.text_index: int | None = None
        self.passed: set[int] = set()
        self.sequence = 0

    def sequenced(self, event: ServerSentEvent) -> ServerSentEvent:
        if event.is_done or '"sequence_number"' not in event.data:
            return event
        payload = event.json()
        if isinstance(payload, dict) and "sequence_number" in payload:
            payload["sequence_number"] = self.sequence
            self.sequence += 1
            return ServerSentEvent.of(payload, event.event)
        return event


@dataclass(frozen=True, slots=True)
class RequestContext:
    path: str
    body: Mapping[str, Any]
    signed: bool = False

    @property
    def wants_stream(self) -> bool:
        return bool(self.body.get("stream"))


class Codec(ABC):
    name: ClassVar[str]
    response_prefix: ClassVar[str] = "immune-"
    server_side_history: ClassVar[bool] = False

    def accepts(self, context: RequestContext) -> bool:
        return True

    def strictly_accepts(self, context: RequestContext) -> bool:
        return self.accepts(context) and isinstance(context.body.get("model"), str)

    def is_streaming(self, context: RequestContext) -> bool:
        return context.wants_stream

    @abstractmethod
    def parse_request(self, context: RequestContext) -> Conversation: ...

    @abstractmethod
    def parse_response(self, body: Mapping[str, Any]) -> Reply: ...

    @abstractmethod
    def assemble_stream(self, events: list[ServerSentEvent]) -> JsonBody: ...

    @abstractmethod
    def render_response(self, body: Mapping[str, Any], edit: ReplyEdit) -> JsonBody: ...

    @abstractmethod
    def synthesize_response(
        self, conversation: Conversation, text: str, response_id: str, refusal: bool
    ) -> JsonBody: ...

    @abstractmethod
    def stream_from_body(self, body: Mapping[str, Any]) -> list[ServerSentEvent]: ...

    @abstractmethod
    def append_operator_note(self, document: JsonDocument, note: str) -> bool: ...

    def parse_stream(self, events: list[ServerSentEvent]) -> Reply:
        return self.parse_response(self.assemble_stream(events))

    def classify_event(self, event: ServerSentEvent, cursor: StreamCursor) -> StreamPart:
        return StreamPart(StreamKind.HOLD)

    def text_event(self, cursor: StreamCursor, text: str) -> ServerSentEvent:
        raise NotImplementedError

    def continuation(self, body: Mapping[str, Any], remainder: str, cursor: StreamCursor) -> list[ServerSentEvent]:
        return self.stream_from_body(body)

    def apply_edits(self, document: JsonDocument, edits: Iterable[TextEdit]) -> None:
        document.apply(edits)


class TextParts:
    TEXT_TYPES: ClassVar[frozenset[str]] = frozenset({"text", "input_text", "output_text"})

    @classmethod
    def walk(cls, content: Any, base: Locator, field_name: str = "text") -> Iterator[tuple[Locator, str]]:
        if isinstance(content, str):
            yield base, content
            return
        if not isinstance(content, list):
            return
        for index, part in enumerate(content):
            if isinstance(part, str):
                yield (*base, index), part
            elif (
                isinstance(part, Mapping)
                and part.get("type") in cls.TEXT_TYPES
                and isinstance(part.get(field_name), str)
            ):
                yield (*base, index, field_name), part[field_name]


class Arguments:
    @staticmethod
    def parse(raw: Any) -> dict[str, Any]:
        if isinstance(raw, Mapping):
            return dict(raw)
        if not isinstance(raw, str) or not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}


class Structured:
    @staticmethod
    def parse(text: str, expected: bool) -> Mapping[str, Any] | None:
        stripped = text.strip()
        if not expected or not stripped.startswith(("{", "[")):
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else {"items": parsed}


class Shapes:
    CHAT_ROLES: ClassVar[frozenset[str]] = frozenset({"system", "developer", "user", "assistant", "tool", "function"})

    @staticmethod
    def messages(value: Any, roles: frozenset[str]) -> bool:
        return (
            isinstance(value, list)
            and bool(value)
            and all(isinstance(item, Mapping) and item.get("role") in roles for item in value)
        )
