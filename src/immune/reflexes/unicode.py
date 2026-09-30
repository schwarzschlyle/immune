from __future__ import annotations

from collections import Counter

from immune.reflexes.findings import Cleaned, Finding

_TAG_START, _TAG_END = 0xE0000, 0xE007F
_BIDI = frozenset({0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)})
_ZERO_WIDTH = frozenset({0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x180E})
_VARIATION_BASIC = range(0xFE00, 0xFE10)
_VARIATION_SUPPLEMENT = range(0xE0100, 0xE01F0)
_THREAT = "inbound.smuggled_characters"


def _is_variation(code: int) -> bool:
    return code in _VARIATION_BASIC or code in _VARIATION_SUPPLEMENT


def _variation_byte(code: int) -> int:
    return code - 0xFE00 if code in _VARIATION_BASIC else code - 0xE0100 + 16


def _is_latin_word_char(character: str) -> bool:
    return character.isascii() and character.isalnum()


class UnicodeReveal:
    def reveal(self, text: str) -> Cleaned:
        if text.isascii():
            return Cleaned(text=text)
        kept: list[str] = []
        tags: list[str] = []
        variation_payload = bytearray()
        counts: Counter[str] = Counter()
        index = 0
        while index < len(text):
            character = text[index]
            code = ord(character)
            if _TAG_START <= code <= _TAG_END:
                decoded = code - _TAG_START
                if 0x20 <= decoded <= 0x7E:
                    tags.append(chr(decoded))
                counts["tag characters"] += 1
            elif code in _BIDI:
                counts["bidirectional controls"] += 1
            elif code in _ZERO_WIDTH and self._between_latin_letters(text, index):
                counts["zero-width characters inside words"] += 1
            elif _is_variation(code):
                run_end = self._variation_run_end(text, index)
                if run_end - index >= 2:
                    variation_payload.extend(_variation_byte(ord(item)) for item in text[index:run_end])
                    counts["variation-selector payload"] += run_end - index
                    index = run_end
                    continue
                kept.append(character)
            else:
                kept.append(character)
            index += 1
        hidden = "".join(tags) + variation_payload.decode("utf-8", errors="ignore")
        findings = tuple(
            Finding(threat=_THREAT, evidence=f"{count} {kind}", payload=hidden or None)
            for kind, count in counts.items()
        )
        return Cleaned(text="".join(kept), hidden=hidden, findings=findings)

    @staticmethod
    def _between_latin_letters(text: str, index: int) -> bool:
        return (
            0 < index < len(text) - 1 and _is_latin_word_char(text[index - 1]) and _is_latin_word_char(text[index + 1])
        )

    @staticmethod
    def _variation_run_end(text: str, start: int) -> int:
        end = start
        while end < len(text) and _is_variation(ord(text[end])):
            end += 1
        return end
