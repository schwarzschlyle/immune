from __future__ import annotations

import hashlib
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

from immune.state.backend import StateBackend

_WORD = re.compile(r"\w+")
_KEY = "provenance"
_SWEEP_EVERY = 500


class Shingles:
    def __init__(self, window: int = 8) -> None:
        self._window = window

    def of(self, text: str) -> set[str]:
        words = [word.lower() for word in _WORD.findall(text)]
        if len(words) < self._window:
            return set()
        return {
            hashlib.blake2b(" ".join(words[index : index + self._window]).encode("utf-8"), digest_size=8).hexdigest()
            for index in range(len(words) - self._window + 1)
        }


class Provenance(ABC):
    def __init__(self, window: int = 8, min_matches: int = 3, min_fraction: float = 0.3) -> None:
        self._shingles = Shingles(window)
        self._min_matches = min_matches
        self._min_fraction = min_fraction

    @abstractmethod
    def absorb(self, text: str) -> None: ...

    @abstractmethod
    def recognizes(self, text: str) -> bool: ...

    def _matched(self, matches: int, total: int) -> bool:
        return total > 0 and matches >= self._min_matches and matches / total >= self._min_fraction


class ProvenanceMemory(Provenance):
    def __init__(
        self, window: int = 8, capacity: int = 200_000, min_matches: int = 3, min_fraction: float = 0.3
    ) -> None:
        super().__init__(window, min_matches, min_fraction)
        self._capacity = capacity
        self._fingerprints: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def is_empty(self) -> bool:
        return not self._fingerprints

    def absorb(self, text: str) -> None:
        fingerprints = self._shingles.of(text)
        if not fingerprints:
            return
        with self._lock:
            for fingerprint in fingerprints:
                self._fingerprints[fingerprint] = None
                self._fingerprints.move_to_end(fingerprint)
            while len(self._fingerprints) > self._capacity:
                self._fingerprints.popitem(last=False)

    def recognizes(self, text: str) -> bool:
        if not self._fingerprints:
            return False
        fingerprints = self._shingles.of(text)
        with self._lock:
            matches = sum(1 for fingerprint in fingerprints if fingerprint in self._fingerprints)
        return self._matched(matches, len(fingerprints))


class SharedProvenanceMemory(Provenance):
    def __init__(self, backend: StateBackend, ttl_s: float = 86_400.0, window: int = 8) -> None:
        super().__init__(window)
        self._backend = backend
        self._ttl_s = ttl_s
        self._absorbed = 0

    def absorb(self, text: str) -> None:
        fingerprints = self._shingles.of(text)
        if not fingerprints:
            return
        now = time.time()
        self._backend.touch(_KEY, fingerprints, now)
        self._absorbed += 1
        if self._absorbed % _SWEEP_EVERY == 0:
            self._backend.forget_before(_KEY, now - self._ttl_s)

    def recognizes(self, text: str) -> bool:
        fingerprints = sorted(self._shingles.of(text))
        if not fingerprints:
            return False
        matches = self._backend.seen(_KEY, fingerprints, time.time() - self._ttl_s)
        return self._matched(matches, len(fingerprints))
