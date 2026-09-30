from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Iterator
from urllib.parse import unquote

from immune.core.conversation import Span
from immune.reflexes.findings import Cleaned, Finding
from immune.reflexes.patterns import PatternSet, SpanRemover

_BASE64 = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{24,}={0,2}(?![A-Za-z0-9+/_-])")
_HEX = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){16,}(?![0-9A-Fa-f])")
_PERCENT = re.compile(r"(?:%[0-9A-Fa-f]{2}){6,}")
_TRANSCRIPT_LINE = re.compile(r"(?im)^[ \t]*(?:user|human|assistant|ai|system|model)[ \t]*:")
_COMMENT = re.compile(r"<!--(?P<body>.*?)-->", re.S)
_HIDDEN_ELEMENT = re.compile(
    r"<(?P<tag>[a-zA-Z][a-zA-Z0-9]{0,15})\b(?P<attrs>[^>]{0,500})>(?P<body>.{0,5000}?)</(?P=tag)\s*>", re.S
)
_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0(?![.\d])|opacity\s*:\s*0(?![.\d])|"
    r"aria-hidden\s*=\s*[\"']true|\bhidden\b|color\s*:\s*(?:#fff(?:fff)?|white)\b",
    re.I,
)
_WORDS = re.compile(r"[A-Za-z]{2,}")
_FAKE_TRANSCRIPT_TURNS = 4


class _Printable:
    @staticmethod
    def text(raw: bytes) -> str | None:
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        printable = sum(character.isprintable() or character.isspace() for character in decoded)
        if printable / max(len(decoded), 1) < 0.95 or len(_WORDS.findall(decoded)) < 3 or " " not in decoded:
            return None
        return decoded


class EncodedPayloadDecoder:
    def findings(self, text: str) -> Iterator[Finding]:
        for match in _BASE64.finditer(text):
            decoded = self._base64(match.group(0))
            if decoded:
                yield Finding("inbound.encoded_payload", "base64 text", Span(*match.span()), decoded)
        for match in _HEX.finditer(text):
            decoded = _Printable.text(binascii.unhexlify(match.group(0)))
            if decoded:
                yield Finding("inbound.encoded_payload", "hex text", Span(*match.span()), decoded)
        for match in _PERCENT.finditer(text):
            decoded = _Printable.text(unquote(match.group(0)).encode("utf-8", errors="ignore"))
            if decoded:
                yield Finding("inbound.encoded_payload", "percent-encoded text", Span(*match.span()), decoded)

    @staticmethod
    def _base64(candidate: str) -> str | None:
        padded = candidate + "=" * (-len(candidate) % 4)
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                decoded = _Printable.text(decoder(padded))
            except (binascii.Error, ValueError):
                continue
            if decoded:
                return decoded
        return None


class TemplateTokenDetector:
    def __init__(self, patterns: PatternSet) -> None:
        self._patterns = patterns

    def findings(self, text: str) -> Iterator[Finding]:
        for match in self._patterns.scan(text):
            yield Finding("inbound.template_tokens", match.name, match.span)
        turns = _TRANSCRIPT_LINE.findall(text)
        if len(turns) >= _FAKE_TRANSCRIPT_TURNS:
            yield Finding("inbound.template_tokens", f"fake transcript with {len(turns)} turns")


class GuardAddressedDetector:
    def __init__(self, patterns: PatternSet, template_tokens: PatternSet) -> None:
        self._patterns = patterns
        self._template_tokens = template_tokens

    def findings(self, text: str) -> Iterator[Finding]:
        for match in self._patterns.scan(text):
            yield Finding("inbound.guard_addressed", match.name, match.span)

    def defang(self, text: str) -> str:
        spans = [match.span for match in self._patterns.scan(text)]
        spans.extend(match.span for match in self._template_tokens.scan(text))
        return SpanRemover.remove(text, spans, " ").strip()


class HiddenMarkupStripper:
    def clean(self, text: str) -> Cleaned:
        if "<" not in text:
            return Cleaned(text=text)
        spans: list[Span] = []
        hidden: list[str] = []
        for match in _COMMENT.finditer(text):
            if len(_WORDS.findall(match.group("body"))) >= 2:
                spans.append(Span(*match.span()))
                hidden.append(match.group("body").strip())
        for match in _HIDDEN_ELEMENT.finditer(text):
            if _HIDDEN_STYLE.search(match.group("attrs")) and _WORDS.search(match.group("body")):
                spans.append(Span(*match.span()))
                hidden.append(match.group("body").strip())
        if not spans:
            return Cleaned(text=text)
        findings = (
            Finding("inbound.hidden_content", f"{len(spans)} hidden markup regions", payload="\n".join(hidden)),
        )
        return Cleaned(text=SpanRemover.remove(text, spans), hidden="\n".join(hidden), findings=findings)
