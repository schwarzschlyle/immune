from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from immune.codecs.base import Codec, StreamCursor, StreamKind
from immune.core.conversation import Span
from immune.core.sse import ServerSentEvent, ServerSentEvents

_LOGGER = logging.getLogger("immune")
_OPEN_URL = re.compile(r"(?:https?://|www\.)\S*$", re.I)
_WORD = re.compile(r"\S+")
_RECHECK_CHARS = 24
Hold = Callable[[str], int | None]
Replacements = Sequence[tuple[Span, str]]


class IncrementalEvents:
    def __init__(self) -> None:
        self._buffer = ""
        self._decoder_tail = b""

    def feed(self, chunk: bytes) -> list[ServerSentEvent]:
        data = self._decoder_tail + chunk
        try:
            text = data.decode("utf-8")
            self._decoder_tail = b""
        except UnicodeDecodeError as error:
            text = data[: error.start].decode("utf-8")
            self._decoder_tail = data[error.start :]
        self._buffer += text.replace("\r\n", "\n")
        blocks = self._buffer.split("\n\n")
        self._buffer = blocks.pop()
        return ServerSentEvents.parse("\n\n".join(blocks)) if blocks else []

    def close(self) -> list[ServerSentEvent]:
        remaining, self._buffer = self._buffer, ""
        return ServerSentEvents.parse(remaining) if remaining.strip() else []

    def pending(self) -> bytes:
        remaining, self._buffer = self._buffer, ""
        tail, self._decoder_tail = self._decoder_tail, b""
        return remaining.encode("utf-8", "surrogatepass") + tail


class Holdback:
    """Releases streamed text as it arrives, keeping a short tail back.

    ``hold`` returns where the first candidate starts (a link, a key-shaped string, a copied passage), or None. From
    there the text waits until the reply is complete and Jev has decided; ``complete`` then releases it with Jev's
    confirmed replacements applied.
    """

    def __init__(self, hold: Hold, tail_words: int = 16, max_tail_chars: int = 512) -> None:
        self._hold = hold
        self._tail_words = tail_words
        self._max_tail_chars = max_tail_chars
        self._raw = ""
        self._released = 0
        self._checked = 0
        self._holding = False
        self.emitted = ""

    @property
    def holding(self) -> bool:
        return self._holding

    @property
    def raw(self) -> str:
        return self._raw

    @property
    def released_raw(self) -> str:
        return self._raw[: self._released]

    def add(self, text: str) -> str:
        self._raw += text
        if self._holding:
            return ""
        if len(self._raw) - self._checked < _RECHECK_CHARS and not text.endswith(("\n", " ")):
            return ""
        self._checked = len(self._raw)
        return self._release(self._boundary(self._hold(self._raw)), ())

    def complete(self, full_text: str, replacements: Replacements = ()) -> str:
        if full_text.startswith(self._raw):
            self._raw = full_text
        else:
            _LOGGER.warning("immune: streamed text diverged from the assembled reply; using the assembled text")
            self._raw = self._raw[: self._released] + full_text[self._released :]
        return self._release(len(self._raw), replacements)

    def _release(self, boundary: int, replacements: Replacements) -> str:
        if boundary <= self._released:
            return ""
        output = self._render(replacements, self._released, boundary)
        self._released = boundary
        self.emitted += output
        return output

    def _boundary(self, hold: int | None) -> int:
        text = self._raw
        words = list(_WORD.finditer(text))
        candidate = words[-self._tail_words].start() if len(words) >= self._tail_words else 0
        candidate = max(candidate, len(text) - self._max_tail_chars)
        candidate = min(candidate, self._open_construct(text))
        if words and not text[-1:].isspace():
            candidate = min(candidate, words[-1].start())
        if hold is not None:
            self._holding = True
            candidate = min(candidate, hold)
        return max(candidate, self._released)

    @staticmethod
    def _open_construct(text: str) -> int:
        positions = [len(text)]
        bracket = text.rfind("[")
        if bracket >= 0:
            closing = text.find("]", bracket)
            if (
                closing < 0
                or (text[closing + 1 : closing + 2] == "(" and text.find(")", closing) < 0)
                or closing == len(text) - 1
            ):
                positions.append(bracket - 1 if bracket > 0 and text[bracket - 1] == "!" else bracket)
        angle = text.rfind("<")
        if angle >= 0 and text.find(">", angle) < 0:
            positions.append(angle)
        url = _OPEN_URL.search(text)
        if url is not None:
            positions.append(url.start())
        return min(positions)

    def _render(self, spans: Sequence[tuple[Span, str]], start: int, end: int) -> str:
        text = self._raw
        pieces: list[str] = []
        cursor = start
        for span, replacement in spans:
            if span.end <= start or span.start >= end:
                continue
            if span.start > cursor:
                pieces.append(text[cursor : span.start])
            if span.start >= start:
                pieces.append(replacement)
            cursor = max(cursor, span.end)
        if cursor < end:
            pieces.append(text[cursor:end])
        return "".join(pieces)


class ProgressiveScreen:
    def __init__(self, codec: Codec, holdback: Holdback) -> None:
        self._codec = codec
        self._holdback = holdback
        self._cursor = StreamCursor()
        self._events = IncrementalEvents()
        self._holding = False
        self._accounted = 0
        self._batch: list[ServerSentEvent] = []
        self.raw_events: list[ServerSentEvent] = []

    @property
    def emitted_text(self) -> str:
        return self._holdback.emitted

    def feed(self, chunk: bytes) -> bytes:
        return self._bytes(self.feed_events(self._events.feed(chunk)))

    def drain(self) -> bytes:
        return self._bytes(self.feed_events(self._events.close()))

    def feed_events(self, events: list[ServerSentEvent]) -> list[ServerSentEvent]:
        return self._sequenced(self._screen(events))

    def unscreened(self) -> bytes:
        return ServerSentEvents.serialize(self.raw_events)

    def complete(self, full_text: str, replacements: Replacements = ()) -> str:
        return self._holdback.complete(full_text, replacements)

    def continuation(self, final_body: Mapping[str, Any], remainder: str) -> list[ServerSentEvent]:
        return self._sequenced(self._codec.continuation(final_body, remainder, self._cursor))

    def fallback(self) -> bytes:
        return self._bytes(self.fallback_events()) + self._events.pending()

    def fallback_events(self) -> list[ServerSentEvent]:
        pending: list[ServerSentEvent] = list(self._batch)
        unreleased = self._holdback.raw[len(self._holdback.released_raw) :]
        if unreleased and self._cursor.text_started:
            pending.append(self._codec.text_event(self._cursor, unreleased))
        pending.extend(self.raw_events[self._accounted :])
        self._batch = []
        self._accounted = len(self.raw_events)
        self._holding = True
        return self._sequenced(pending)

    def _screen(self, events: list[ServerSentEvent]) -> list[ServerSentEvent]:
        self.raw_events.extend(events)
        self._batch = []
        for event in events:
            if self._holding:
                continue
            part = self._codec.classify_event(event, self._cursor)
            if part.kind is StreamKind.FORWARD:
                self._batch.append(event)
                self._accounted += 1
            elif part.kind is StreamKind.TEXT:
                self._cursor.text_started = True
                self._accounted += 1
                text = self._holdback.add(part.text)
                if text:
                    self._batch.append(self._codec.text_event(self._cursor, text))
            else:
                self._holding = True
        released, self._batch = self._batch, []
        return released

    def _sequenced(self, events: Sequence[ServerSentEvent]) -> list[ServerSentEvent]:
        return [self._cursor.sequenced(event) for event in events]

    @staticmethod
    def _bytes(events: Sequence[ServerSentEvent]) -> bytes:
        return ServerSentEvents.serialize(events) if events else b""
