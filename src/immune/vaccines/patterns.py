from __future__ import annotations

import importlib
import re
from collections.abc import Iterator, Sequence
from typing import Any

from immune.core.conversation import Span

MAX_REPEAT = 1_000
_REPEATS = frozenset({"MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT"})
_FIX = "use a bounded quantifier such as {0,200} instead of * and {1,200} instead of +"


class UnsafePattern(ValueError):
    pass


class PatternSafety:
    def __init__(self) -> None:
        try:
            self._parser: Any = importlib.import_module("re._parser")
            self._unbounded: int = importlib.import_module("re._constants").MAXREPEAT
        except ImportError:
            self._parser = None
            self._unbounded = -1

    def check(self, pattern: str) -> None:
        try:
            re.compile(pattern)
        except re.error as error:
            raise UnsafePattern(f"{pattern!r} is not a valid regular expression: {error}") from error
        if self._parser is None:
            self._check_text(pattern)
            return
        self._walk(self._parser.parse(pattern), pattern, inside=False)

    def _walk(self, items: Any, pattern: str, inside: bool) -> None:
        for op, value in items:
            if str(op) in _REPEATS:
                _, high, body = value
                if high == self._unbounded or high > MAX_REPEAT:
                    raise UnsafePattern(f"{pattern!r} repeats without a bound; {_FIX}")
                if inside:
                    raise UnsafePattern(f"{pattern!r} nests one repetition inside another, which can hang on long text")
                self._walk(body, pattern, inside=True)
                continue
            for child in self._children(value):
                self._walk(child, pattern, inside)

    def _children(self, value: Any) -> Iterator[Any]:
        if isinstance(value, self._parser.SubPattern):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from self._children(item)

    @staticmethod
    def _check_text(pattern: str) -> None:
        if re.search(r"(?<!\\)[*+]|\{\d*,\}", pattern):
            raise UnsafePattern(f"{pattern!r} repeats without a bound; {_FIX}")


class TextMatcher:
    def __init__(self, keywords: Sequence[str], patterns: Sequence[str]) -> None:
        self._keywords = (
            re.compile(r"(?<!\w)(?:" + "|".join(re.escape(keyword.strip()) for keyword in keywords) + r")(?!\w)", re.I)
            if keywords
            else None
        )
        self._patterns = [re.compile(pattern) for pattern in patterns]

    def matches(self, text: str) -> Iterator[tuple[str, Span]]:
        if self._keywords is not None:
            for match in self._keywords.finditer(text):
                yield f"keyword {match.group(0)!r}", Span(*match.span())
        for pattern in self._patterns:
            for match in pattern.finditer(text):
                yield f"pattern /{pattern.pattern}/", Span(*match.span())
