from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Iterable, MutableMapping, MutableSequence, Sequence
from dataclasses import dataclass
from typing import Any

from immune.core.conversation import Locator, Span
from immune.errors import CodecError


@dataclass(frozen=True, slots=True)
class TextEdit:
    locator: Locator
    replacement: str
    span: Span | None = None


class JsonDocument:
    def __init__(self, body: MutableMapping[str, Any]) -> None:
        self._body = copy.deepcopy(body)

    @property
    def body(self) -> MutableMapping[str, Any]:
        return self._body

    def get(self, locator: Locator) -> Any:
        node: Any = self._body
        for key in locator:
            node = node[key]
        return node

    def set(self, locator: Locator, value: Any) -> None:
        if not locator:
            raise CodecError("cannot replace the document root")
        parent = self.get(locator[:-1])
        key = locator[-1]
        writable_list = isinstance(parent, MutableSequence) and isinstance(key, int)
        writable_map = isinstance(parent, MutableMapping) and isinstance(key, str)
        if writable_list or writable_map:
            parent[key] = value
        else:
            raise CodecError(f"locator {locator!r} does not address a writable field")

    def apply(self, edits: Iterable[TextEdit]) -> None:
        grouped: dict[Locator, list[TextEdit]] = defaultdict(list)
        for edit in edits:
            grouped[edit.locator].append(edit)
        for locator, group in grouped.items():
            self.set(locator, self.rewrite(self.get(locator), group))

    @staticmethod
    def rewrite(original: Any, edits: Sequence[TextEdit]) -> str:
        if not isinstance(original, str):
            raise CodecError("text edits require a string field")
        whole = [edit for edit in edits if edit.span is None]
        if whole:
            return whole[-1].replacement
        text = original
        for edit in sorted(edits, key=lambda item: item.span.start if item.span else 0, reverse=True):
            assert edit.span is not None
            text = text[: edit.span.start] + edit.replacement + text[edit.span.end :]
        return text
