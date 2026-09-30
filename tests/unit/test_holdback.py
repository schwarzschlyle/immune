from __future__ import annotations

from immune.core.conversation import Span
from immune.core.streaming import Holdback

WORDS = " ".join(f"word{index}" for index in range(40)) + " "


def marker(text: str) -> int | None:
    position = text.find("MARK")
    return position if position >= 0 else None


def test_text_is_released_with_a_tail_held_back() -> None:
    holdback = Holdback(lambda _: None)
    released = holdback.add(WORDS)
    assert released
    assert WORDS.startswith(released)
    assert len(released) < len(WORDS)
    assert not holdback.holding


def test_release_stops_at_the_first_candidate_until_the_reply_is_complete() -> None:
    holdback = Holdback(marker)
    before = holdback.add(WORDS)
    during = holdback.add("MARK then more text after it ")
    assert holdback.holding
    assert "MARK" not in before + during
    assert holdback.add(WORDS) == ""
    full = WORDS + "MARK then more text after it " + WORDS
    start = full.index("MARK")
    rest = holdback.complete(full, [(Span(start, start + 4), "[redacted]")])
    assert before + during + rest == full.replace("MARK", "[redacted]")


def test_a_declined_candidate_is_released_as_written() -> None:
    holdback = Holdback(marker)
    released = holdback.add(WORDS) + holdback.add("MARK stays ")
    full = WORDS + "MARK stays "
    assert released + holdback.complete(full) == full
