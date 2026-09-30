from __future__ import annotations

import codecs
import sys
from collections.abc import Sequence
from typing import TextIO


def _encoding_of(stream: TextIO) -> str:
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        return codecs.lookup(encoding).name
    except LookupError:
        return "utf-8"


class Console:
    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout
        # Windows pipes and legacy consoles use a code page (cp1252, cp437), not UTF-8, so output must fit it.
        self._encoding = _encoding_of(self._stream)
        self._rule = "─" if self._fits("─") else "-"

    def _fits(self, text: str) -> bool:
        try:
            text.encode(self._encoding)
        except UnicodeEncodeError:
            return False
        return True

    def line(self, text: str = "") -> None:
        self.write(f"{text}\n")

    def write(self, text: str) -> None:
        if not self._fits(text):
            text = text.encode(self._encoding, errors="replace").decode(self._encoding)
        self._stream.write(text)

    def heading(self, text: str) -> None:
        self.line(text)
        self.line(self._rule * min(len(text), 100))

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        cells = [[str(cell) for cell in row] for row in rows]
        widths = [max([len(header), *(len(row[index]) for row in cells)]) for index, header in enumerate(headers)]
        self.line("  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)).rstrip())
        self.line("  ".join(self._rule * width for width in widths))
        for row in cells:
            self.line("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)).rstrip())
