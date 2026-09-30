from __future__ import annotations

import base64
import binascii
import itertools
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from urllib.parse import parse_qsl, unquote, urlsplit

from immune.core.conversation import Span
from immune.reflexes.findings import Finding
from immune.reflexes.patterns import SpanRemover

_MARKDOWN_LINK = re.compile(
    r"(?P<image>!?)\[(?P<label>[^\[\]\n]{0,500})\]\(\s*<?(?P<url>[^)\s>]{1,4000})>?(?:\s+\"[^\"\n]{0,500}\")?\s*\)"
)
_HTML_URL = re.compile(
    r"<(?P<tag>img|a|iframe|source|video|audio|link)\b[^<>]{0,500}?\b(?:src|href)\s*=\s*[\"'](?P<url>[^\"'<>]{1,4000})[\"'][^<>]{0,500}>",
    re.I,
)
_BARE_URL = re.compile(r"https?://[^\s<>\"'()\[\]]{1,4000}")
_PAYLOAD_VALUE = re.compile(r"^(?:[A-Za-z0-9+/_=-]{24,}|(?:%[0-9A-Fa-f]{2}){4,}.*|[0-9a-fA-F]{32,})$")
_PLACEHOLDER = re.compile(r"\{\{?[^}]{1,80}\}\}?|\$\{[^}]{1,80}\}|<[A-Z_]{3,40}>")
_DANGEROUS_MARKUP = (
    re.compile(r"<script\b[^>]{0,500}>.{0,20000}?</script\s*>", re.S | re.I),
    re.compile(
        r"<(?:iframe|object|embed|frameset|frame|applet|form)\b[^>]{0,500}>(?:.{0,20000}?</(?:iframe|object|embed|frameset|frame|applet|form)\s*>)?",
        re.S | re.I,
    ),
    re.compile(r"<(?:meta|base|link)\b[^>]{0,500}>", re.I),
    re.compile(r"\son[a-z]{3,20}\s*=\s*(?:\"[^\"]{0,2000}\"|'[^']{0,2000}'|[^\s>]{1,200})", re.I),
    re.compile(r"(?:javascript|vbscript)\s*:[^\s\"')>]{0,2000}|data\s*:\s*text/html[^\s\"')>]{0,4000}", re.I),
)
_NUMBER = re.compile(
    r"(?P<currency>[$€£¥])\s?(?P<amount>\d[\d,]*(?:\.\d+)?)|(?P<plain>\d[\d,]*(?:\.\d+)?)\s?(?P<unit>%|percent\b|dollars?\b|usd\b|eur\b)",
    re.I,
)
_WORD = re.compile(r"\w+")
_TOKEN = re.compile(r"\w[\w-]{3,}\w")
_PAIRS_TO_LEAK = 2
_CONTENT_WORD = 4
_SHARED_WORDS = 3
_SHARED_FRACTION = 0.6
_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,5}$")
_COMMON = frozenset(
    [
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "your",
        "what",
        "when",
        "where",
        "which",
        "their",
        "there",
        "about",
        "would",
        "could",
        "should",
        "these",
        "those",
        "into",
        "only",
        "other",
        "some",
        "than",
        "then",
        "them",
        "they",
        "very",
        "also",
        "just",
        "like",
        "more",
        "most",
        "such",
        "much",
        "many",
        "make",
        "made",
        "been",
        "were",
        "said",
        "each",
        "does",
        "done",
        "here",
        "over",
        "under",
        "after",
        "before",
        "while",
        "because",
    ]
)
_IDENTIFIER = re.compile(
    r"^(?:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{40}|[0-9a-f]{64})$", re.I
)
_ENCODED = re.compile(r"^[A-Za-z0-9+/_=-]+$")


class ContextVocabulary:
    def __init__(self, text: str = "") -> None:
        words = [word.lower() for word in _WORD.findall(text)]
        self._words = frozenset(word for word in words if len(word) >= _CONTENT_WORD and word not in _COMMON)
        self._tokens = frozenset(token for token in self._candidates(text) if self._distinctive(token))
        self._pairs = frozenset(
            (first, second) for first, second in itertools.pairwise(words) if len(first) > 2 and len(second) > 2
        )

    def leaks(self, value: str) -> bool:
        if not self._tokens and not self._pairs:
            return False
        if any(token in self._tokens for token in self._candidates(value)):
            return True
        words = [word.lower() for word in _WORD.findall(value)]
        if sum(1 for pair in itertools.pairwise(words) if pair in self._pairs) >= _PAIRS_TO_LEAK:
            return True
        content = {word for word in words if len(word) >= _CONTENT_WORD}
        shared = content & self._words
        return len(shared) >= _SHARED_WORDS and len(shared) >= _SHARED_FRACTION * len(content)

    @staticmethod
    def _candidates(text: str) -> set[str]:
        return {token.lower() for pattern in (_WORD, _TOKEN) for token in pattern.findall(text)}

    @staticmethod
    def _distinctive(token: str) -> bool:
        has_digit = any(character.isdigit() for character in token)
        has_letter = any(character.isalpha() for character in token)
        return (has_digit and has_letter and len(token) >= 5) or len(token) >= 14


@dataclass(frozen=True, slots=True)
class LinkReference:
    url: str
    host: str
    span: Span
    is_image: bool


class LinkInspector:
    def references(self, text: str) -> Iterator[LinkReference]:
        seen: set[tuple[int, int]] = set()
        for match in _MARKDOWN_LINK.finditer(text):
            seen.add(match.span("url"))
            yield self._reference(match.group("url"), Span(*match.span()), bool(match.group("image")))
        for match in _HTML_URL.finditer(text):
            seen.add(match.span("url"))
            is_image = match.group("tag").lower() in ("img", "source", "video", "audio")
            yield self._reference(match.group("url"), Span(*match.span()), is_image)
        for match in _BARE_URL.finditer(text):
            if any(start <= match.start() < end for start, end in seen):
                continue
            yield self._reference(match.group(0), Span(*match.span()), False)

    @staticmethod
    def hosts(text: str) -> set[str]:
        return {reference.host for reference in LinkInspector().references(text) if reference.host}

    @staticmethod
    def _reference(url: str, span: Span, is_image: bool) -> LinkReference:
        host = (urlsplit(url.strip()).hostname or "").lower()
        return LinkReference(url=url, host=host, span=span, is_image=is_image)


class PayloadDetector:
    def __init__(self, vocabulary: ContextVocabulary | None = None) -> None:
        self._vocabulary = vocabulary or ContextVocabulary()

    def carries(self, url: str, is_image: bool) -> bool:
        parts = urlsplit(url.strip())
        if _PLACEHOLDER.search(unquote(url)):
            return True
        words, chars = (5, 80) if is_image else (6, 120)
        parameters = [piece.partition("=")[2] for piece in parts.query.split("&") if piece]
        parameters.append(parts.fragment)
        if any(self._parameter(value, words, chars) for value in parameters if value):
            return True
        if self._vocabulary.leaks(" ".join(unquote(value.replace("+", " ")) for value in parameters)):
            return True
        segments = [segment for segment in parts.path.split("/") if segment]
        if any(self._segment(unquote(segment)) for segment in segments):
            return True
        return any(self._label(label) for label in (parts.hostname or "").split(".")[:-2])

    def nested(self, url: str) -> Iterator[str]:
        for _, value in parse_qsl(urlsplit(url.strip()).query, keep_blank_values=True):
            candidate = unquote(value)
            if candidate.lower().startswith(("http://", "https://")):
                yield candidate
                yield from self.nested(candidate)

    def _parameter(self, raw: str, words: int, chars: int) -> bool:
        decoded = unquote(raw.replace("+", " "))
        if decoded.lower().startswith(("http://", "https://")):
            return False
        token = not any(character.isspace() for character in decoded)
        if token and len(raw) >= 24 and _PAYLOAD_VALUE.match(raw) and not _IDENTIFIER.match(decoded):
            return True
        return len(decoded.split()) >= words or len(decoded) >= chars or self._meaningful(decoded)

    def _segment(self, segment: str) -> bool:
        stem = _EXTENSION.sub("", segment)
        return len(segment.split()) >= 6 or len(segment) >= 120 or self._meaningful(stem)

    def _label(self, label: str) -> bool:
        if len(label) >= 24 and _PAYLOAD_VALUE.match(label):
            return True
        return self._meaningful(label.replace("-", " ").replace("_", " "))

    def _meaningful(self, value: str) -> bool:
        if self._vocabulary.leaks(value):
            return True
        decoded = EncodedText.decode(value)
        return decoded is not None and (len(decoded.split()) >= 3 or self._vocabulary.leaks(decoded))


class EncodedText:
    @staticmethod
    def decode(value: str) -> str | None:
        compact = value.strip()
        if len(compact) < 12 or not _ENCODED.match(compact):
            return None
        for decoder in (EncodedText._base64, EncodedText._hex):
            text = decoder(compact)
            if text is not None and EncodedText._readable(text):
                return text
        return None

    @staticmethod
    def _base64(value: str) -> str | None:
        padded = value + "=" * (-len(value) % 4)
        try:
            return base64.b64decode(padded.replace("-", "+").replace("_", "/"), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            return None

    @staticmethod
    def _hex(value: str) -> str | None:
        try:
            return bytes.fromhex(value).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return None

    @staticmethod
    def _readable(text: str) -> bool:
        printable = sum(1 for character in text if character.isprintable())
        letters = sum(1 for character in text if character.isalpha())
        return bool(text) and printable / len(text) > 0.95 and letters / len(text) > 0.5


class ExfiltrationLinkDetector:
    def __init__(self, inspector: LinkInspector | None = None) -> None:
        self._inspector = inspector or LinkInspector()

    def findings(self, text: str, trusted_hosts: Iterable[str], context: str = "") -> Iterator[Finding]:
        trusted = {host.lower() for host in trusted_hosts}
        payloads = PayloadDetector(ContextVocabulary(context))
        for reference in self._inspector.references(text):
            target = self._exfiltration_target(reference, trusted, payloads)
            if target is None:
                continue
            kind = "image" if reference.is_image else "link"
            yield Finding("output.exfil_link", f"{kind} to {target} carries data", reference.span, subject=target)

    def _exfiltration_target(
        self, reference: LinkReference, trusted: set[str], payloads: PayloadDetector
    ) -> str | None:
        untrusted = reference.host and not self._is_trusted(reference.host, trusted)
        if untrusted and payloads.carries(reference.url, reference.is_image):
            return reference.host
        for nested in payloads.nested(reference.url):
            host = (urlsplit(nested).hostname or "").lower()
            if host and not self._is_trusted(host, trusted) and payloads.carries(nested, reference.is_image):
                return host
        return None

    @staticmethod
    def _is_trusted(host: str, trusted: set[str]) -> bool:
        return any(host == known or host.endswith(f".{known}") for known in trusted)


class MarkupSanitizer:
    def findings(self, text: str) -> Iterator[Finding]:
        if "<" not in text and ":" not in text:
            return
        for pattern in _DANGEROUS_MARKUP:
            for match in pattern.finditer(text):
                yield Finding("output.unsafe_markup", match.group(0)[:40], Span(*match.span()))


class Canary:
    def __init__(self, token: str) -> None:
        self.token = token
        self._pattern = re.compile(re.escape(token.rsplit(":", maxsplit=1)[-1]), re.I)

    @property
    def note(self) -> str:
        return f"\n\n[{self.token}]"

    def findings(self, text: str, threat: str = "output.canary_leak") -> Iterator[Finding]:
        for match in self._pattern.finditer(text):
            yield Finding(threat, "deployment canary", Span(*match.span()))


class PromptCopyDetector:
    def __init__(self, window: int = 12) -> None:
        self._window = window

    def findings(self, output: str, operator: str) -> Iterator[Finding]:
        source_words = [word.lower() for word in _WORD.findall(operator)]
        if len(source_words) < self._window:
            return
        shingles = {tuple(source_words[i : i + self._window]) for i in range(len(source_words) - self._window + 1)}
        tokens = list(_WORD.finditer(output))
        words = [token.group(0).lower() for token in tokens]
        spans = [
            Span(tokens[index].start(), tokens[index + self._window - 1].end())
            for index in range(len(words) - self._window + 1)
            if tuple(words[index : index + self._window]) in shingles
        ]
        for span in SpanRemover.merge(spans):
            yield Finding("output.prompt_copy", "copied operator instructions", span)


class GroundedNumbers:
    def findings(self, output: str, grounding: str) -> Iterator[Finding]:
        known = {self._normalize(match) for match in _NUMBER.finditer(grounding)}
        for match in _NUMBER.finditer(output):
            if self._normalize(match) not in known:
                yield Finding("business.ungrounded_number", match.group(0).strip(), Span(*match.span()))

    @staticmethod
    def _normalize(match: re.Match[str]) -> str:
        amount = (match.group("amount") or match.group("plain") or "").replace(",", "")
        return amount.rstrip("0").rstrip(".") if "." in amount else amount
