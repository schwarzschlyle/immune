from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from immune.core.conversation import Span
from immune.reflexes.patterns import PatternSet, SpanRemover

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,24}\b")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b")
_SSN = re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b")
_PHONE = re.compile(r"(?<![\w+])\+?\d{1,3}?[ .-]?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\w)")


@dataclass(frozen=True, slots=True)
class SensitiveMatch:
    kind: str
    span: Span
    value: str


class Luhn:
    @staticmethod
    def valid(number: str) -> bool:
        digits = [int(character) for character in number if character.isdigit()]
        if not 13 <= len(digits) <= 19:
            return False
        checksum = 0
        for position, digit in enumerate(reversed(digits)):
            doubled = digit * 2 if position % 2 == 1 else digit
            checksum += doubled - 9 if doubled > 9 else doubled
        return checksum % 10 == 0


class Iban:
    @staticmethod
    def valid(candidate: str) -> bool:
        compact = candidate.replace(" ", "")
        if not 15 <= len(compact) <= 34:
            return False
        rearranged = compact[4:] + compact[:4]
        numeric = "".join(str(int(character, 36)) for character in rearranged)
        return int(numeric) % 97 == 1


class PersonalDataScanner:
    def find(self, text: str) -> Iterator[SensitiveMatch]:
        for match in _EMAIL.finditer(text):
            yield SensitiveMatch("email", Span(*match.span()), match.group(0))
        for match in _CARD.finditer(text):
            if Luhn.valid(match.group(0)):
                yield SensitiveMatch("card", Span(*match.span()), match.group(0))
        for match in _IBAN.finditer(text):
            if Iban.valid(match.group(0)):
                yield SensitiveMatch("iban", Span(*match.span()), match.group(0))
        for match in _SSN.finditer(text):
            yield SensitiveMatch("ssn", Span(*match.span()), match.group(0))
        for match in _PHONE.finditer(text):
            yield SensitiveMatch("phone", Span(*match.span()), match.group(0))


class SecretScanner:
    def __init__(self, patterns: PatternSet) -> None:
        self._patterns = patterns

    def find(self, text: str) -> Iterator[SensitiveMatch]:
        for match in self._patterns.scan(text):
            yield SensitiveMatch(f"secret:{match.name}", match.span, match.text)


class Redactor:
    def __init__(self, secrets: SecretScanner, personal: PersonalDataScanner) -> None:
        self._secrets = secrets
        self._personal = personal

    def mask(self, text: str) -> str:
        matches = [*self._secrets.find(text), *self._personal.find(text)]
        if not matches:
            return text
        by_start = sorted(matches, key=lambda match: (match.span.start, -match.span.end))
        kept: list[SensitiveMatch] = []
        for match in by_start:
            if kept and match.span.start < kept[-1].span.end:
                continue
            kept.append(match)
        for match in reversed(kept):
            placeholder = f"[{match.kind.split(':')[0].upper()}]"
            text = text[: match.span.start] + placeholder + text[match.span.end :]
        return text

    def remove(self, text: str, replacement: str) -> str:
        return SpanRemover.remove(text, [match.span for match in self._secrets.find(text)], replacement)
