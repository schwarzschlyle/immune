from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_MASKS = (
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<email>"),
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b"), "<date>"),
    (re.compile(r"\b[0-9a-f]{8,}\b", re.I), "<hex>"),
    (re.compile(r"\d+(?:[.,]\d+)*"), "<n>"),
)
_WORD = re.compile(r"[\w<>]+")


class TemplateMasker:
    def mask(self, text: str) -> str:
        for pattern, replacement in _MASKS:
            text = pattern.sub(replacement, text)
        return text


@dataclass(frozen=True, slots=True)
class MinHashSignature:
    values: tuple[int, ...]

    def similarity(self, other: MinHashSignature) -> float:
        if not self.values and not other.values:
            return 1.0
        if not self.values or len(self.values) != len(other.values):
            return 0.0
        return sum(a == b for a, b in zip(self.values, other.values, strict=True)) / len(self.values)


class MinHasher:
    def __init__(self, permutations: int = 64, shingle: int = 4) -> None:
        self._permutations = permutations
        self._shingle = shingle

    def shingles(self, text: str) -> set[str]:
        words = [word.lower() for word in _WORD.findall(text)]
        if len(words) < self._shingle:
            return {" ".join(words)} if words else set()
        return {" ".join(words[index : index + self._shingle]) for index in range(len(words) - self._shingle + 1)}

    def signature(self, text: str) -> MinHashSignature:
        shingles = self.shingles(text)
        if not shingles:
            return MinHashSignature(())
        return MinHashSignature(
            tuple(min(self._hash(shingle, seed) for shingle in shingles) for seed in range(self._permutations))
        )

    @staticmethod
    def _hash(value: str, seed: int) -> int:
        digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8, salt=seed.to_bytes(16, "little"))
        return int.from_bytes(digest.digest(), "little")
