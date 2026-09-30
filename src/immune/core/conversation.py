from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

Locator = tuple[str | int, ...]


class Channel(StrEnum):
    OPERATOR = "operator"
    USER = "user"
    DATA = "data"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int

    def contains(self, other: Span) -> bool:
        return self.start <= other.start and other.end <= self.end


@dataclass(frozen=True, slots=True)
class Segment:
    channel: Channel
    text: str
    origin: str
    locator: Locator
    turn: int = 0
    span: Span | None = None
    confidence: float = 1.0
    historical: bool = False

    @property
    def is_embedded(self) -> bool:
        return self.span is not None

    def with_text(self, text: str) -> Segment:
        return replace(self, text=text)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    builtin: bool = False


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    raw_arguments: str = ""

    @property
    def signature(self) -> str:
        return f"{self.name}:{json.dumps(self.arguments, sort_keys=True, default=str)}"

    def describe(self) -> str:
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.arguments.items()))
        return f"{self.name}({rendered})"


@dataclass(frozen=True, slots=True)
class PastCall:
    call: ToolCall
    turn: int


@dataclass(frozen=True, slots=True)
class Conversation:
    provider: str
    model: str
    segments: tuple[Segment, ...]
    tools: tuple[ToolSpec, ...] = ()
    response_schema: Mapping[str, Any] | None = None
    stream: bool = False
    user_hint: str | None = None
    previous_response_id: str | None = None
    history_calls: tuple[PastCall, ...] = ()

    def by_channel(self, channel: Channel) -> tuple[Segment, ...]:
        return tuple(segment for segment in self.segments if segment.channel is channel)

    @property
    def operator(self) -> tuple[Segment, ...]:
        return self.by_channel(Channel.OPERATOR)

    @property
    def data(self) -> tuple[Segment, ...]:
        return self.by_channel(Channel.DATA)

    @property
    def users(self) -> tuple[Segment, ...]:
        return self.by_channel(Channel.USER)

    @property
    def operator_text(self) -> str:
        return "\n\n".join(segment.text for segment in self.operator)

    @property
    def latest_user(self) -> Segment | None:
        users = self.users
        return users[-1] if users else None

    @property
    def latest_turn(self) -> int:
        return max((segment.turn for segment in self.segments), default=0)

    @property
    def turn_count(self) -> int:
        return len({segment.turn for segment in self.users})

    @property
    def has_tools(self) -> bool:
        return bool(self.tools)

    @property
    def has_context(self) -> bool:
        return bool(self.data)

    @property
    def expects_structure(self) -> bool:
        return self.response_schema is not None

    def tool(self, name: str) -> ToolSpec | None:
        return next((tool for tool in self.tools if tool.name == name), None)

    def recent(self, limit: int) -> tuple[Segment, ...]:
        dialogue = [segment for segment in self.segments if segment.channel in (Channel.USER, Channel.ASSISTANT)]
        return tuple(dialogue[-limit:])

    def with_segments(self, segments: tuple[Segment, ...]) -> Conversation:
        return replace(self, segments=segments)

    def with_history(self, history: tuple[Segment, ...]) -> Conversation:
        if not history:
            return self
        operators = tuple(segment for segment in self.segments if segment.channel is Channel.OPERATOR)
        rest = tuple(segment for segment in self.segments if segment.channel is not Channel.OPERATOR)
        return replace(self, segments=(*operators, *history, *rest))

    @property
    def current_data(self) -> tuple[Segment, ...]:
        return tuple(segment for segment in self.data if not segment.historical)

    def calls_in_turn(self, turn: int) -> tuple[ToolCall, ...]:
        return tuple(past.call for past in self.history_calls if past.turn == turn)

    @property
    def assistant_before_latest_user(self) -> str:
        latest = self.latest_user
        if latest is None:
            return ""
        position = self.segments.index(latest)
        preceding: list[str] = []
        for segment in reversed(self.segments[:position]):
            if segment.channel is Channel.USER:
                break
            if segment.channel is Channel.ASSISTANT:
                preceding.append(segment.text)
        return "\n".join(reversed(preceding))


@dataclass(frozen=True, slots=True)
class Reply:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    structured: Mapping[str, Any] | None = None
    stop_reason: str | None = None
    refused: bool = False
    response_id: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.text and not self.tool_calls
