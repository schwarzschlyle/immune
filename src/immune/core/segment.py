from __future__ import annotations

import re
from collections.abc import Iterator

from immune.core.conversation import Channel, Conversation, Segment, Span
from immune.sessions.provenance import Provenance

_TAGGED = re.compile(
    r"<(?P<tag>untrusted|documents?|context|email|search_results|web_page|webpage|tool_output|retrieved|source)"
    r"\b(?P<attrs>[^>]{0,300})>(?P<body>.*?)</(?P=tag)\s*>",
    re.S | re.I,
)
_SOURCE_ATTRIBUTE = re.compile(r"source\s*=\s*[\"'](?P<source>[^\"']{1,80})[\"']", re.I)
_LABELED = re.compile(
    r"(?im)^[ \t]*(?:context|documents?|retrieved(?:[ \t]+documents?)?|search[ \t]+results|email|web[ \t]+page)"
    r"[ \t]*:[ \t]*\n"
)
_LABEL_END = re.compile(r"(?im)^[ \t]*(?:question|task|user|instructions?|request)[ \t]*:")
_EMBEDDED_PLACEHOLDER = "[embedded data]"
_MIN_LABELED_CHARS = 200


class UntrustedMarker:
    @staticmethod
    def wrap(text: str, source: str = "data") -> str:
        safe_source = re.sub(r"[^A-Za-z0-9_.:-]", "_", source)[:80] or "data"
        return f'<untrusted source="{safe_source}">{text}</untrusted>'


class EmbeddedDataSegmenter:
    def __init__(self, memory: Provenance | None = None) -> None:
        self._memory = memory

    def split(self, conversation: Conversation) -> Conversation:
        segments: list[Segment] = []
        for segment in conversation.segments:
            if segment.channel not in (Channel.USER, Channel.OPERATOR) or segment.historical:
                segments.append(segment)
                continue
            embedded = list(self._embedded(segment))
            if embedded:
                segments.append(segment.with_text(self._without(segment.text, embedded)))
                segments.extend(embedded)
            elif segment.channel is Channel.USER and self._memory is not None and self._memory.recognizes(segment.text):
                segments.append(
                    Segment(Channel.DATA, segment.text, "provenance", segment.locator, segment.turn, confidence=0.8)
                )
            else:
                segments.append(segment)
        return conversation.with_segments(tuple(segments))

    def _embedded(self, segment: Segment) -> Iterator[Segment]:
        tagged = list(_TAGGED.finditer(segment.text))
        for match in tagged:
            source = _SOURCE_ATTRIBUTE.search(match.group("attrs"))
            tag = match.group("tag").lower()
            origin = f"embedded:{source.group('source')}" if source else f"embedded:{tag}"
            yield Segment(
                Channel.DATA,
                match.group("body"),
                origin,
                segment.locator,
                segment.turn,
                span=Span(match.start("body"), match.end("body")),
                confidence=1.0 if tag == "untrusted" else 0.9,
            )
        if tagged:
            return
        for match in _LABELED.finditer(segment.text):
            end_match = _LABEL_END.search(segment.text, match.end())
            end = end_match.start() if end_match else len(segment.text)
            if end - match.end() >= _MIN_LABELED_CHARS:
                yield Segment(
                    Channel.DATA,
                    segment.text[match.end() : end],
                    "embedded:labeled",
                    segment.locator,
                    segment.turn,
                    span=Span(match.end(), end),
                    confidence=0.6,
                )

    @staticmethod
    def _without(text: str, embedded: list[Segment]) -> str:
        for segment in sorted(embedded, key=lambda item: item.span.start if item.span else 0, reverse=True):
            assert segment.span is not None
            text = text[: segment.span.start] + _EMBEDDED_PLACEHOLDER + text[segment.span.end :]
        return text
