from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from immune.core.conversation import Span


@dataclass(frozen=True, slots=True)
class PatternMatch:
    name: str
    span: Span
    text: str


class PatternSet:
    def __init__(self, patterns: Mapping[str, str]) -> None:
        self._compiled = {name: re.compile(pattern) for name, pattern in patterns.items()}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._compiled)

    def scan(self, text: str) -> Iterator[PatternMatch]:
        for name, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                yield PatternMatch(name=name, span=Span(match.start(), match.end()), text=match.group(0))

    def matches(self, text: str) -> list[PatternMatch]:
        return list(self.scan(text))

    def first(self, text: str) -> PatternMatch | None:
        return next(self.scan(text), None)


class SpanRemover:
    @staticmethod
    def remove(text: str, spans: list[Span], replacement: str = "") -> str:
        merged = SpanRemover.merge(spans)
        for span in reversed(merged):
            text = text[: span.start] + replacement + text[span.end :]
        return text

    @staticmethod
    def merge(spans: list[Span]) -> list[Span]:
        merged: list[Span] = []
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            if merged and span.start <= merged[-1].end:
                merged[-1] = Span(merged[-1].start, max(merged[-1].end, span.end))
            else:
                merged.append(span)
        return merged
