from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

from immune.reflexes.findings import Cleaned, Finding
from immune.reflexes.tools import Destination

T = TypeVar("T")
_MAX_TEXT = 32_768
_CAPACITY = 1024


class BoundedMemo(Generic[T]):
    def __init__(self, compute: Callable[[str], T], capacity: int = _CAPACITY, max_text: int = _MAX_TEXT) -> None:
        self._compute = compute
        self._capacity = capacity
        self._max_text = max_text
        self._entries: OrderedDict[str, T] = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, text: str) -> T:
        if len(text) > self._max_text:
            return self._compute(text)
        with self._lock:
            if text in self._entries:
                self._entries.move_to_end(text)
                return self._entries[text]
        value = self._compute(text)
        with self._lock:
            self._entries[text] = value
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        return value


InboundScan = tuple[tuple[Finding, ...], str]


class TextMemo:
    def __init__(
        self,
        reveal: Callable[[str], Cleaned],
        clean: Callable[[str], Cleaned],
        destinations: Callable[[str], set[Destination]],
        inbound: Callable[[str], InboundScan],
    ) -> None:
        self.reveal = BoundedMemo(reveal)
        self.clean = BoundedMemo(clean)
        self.destinations = BoundedMemo(lambda text: frozenset(destinations(text)))
        self.inbound = BoundedMemo(inbound)
