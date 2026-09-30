from __future__ import annotations

import os
import sys
import textwrap

from immune.types import Verdict

_STYLES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "blue": "34", "cyan": "36"}


class Console:
    WIDTH = 110

    def __init__(self, color: bool | None = None) -> None:
        self.color = sys.stdout.isatty() and "NO_COLOR" not in os.environ if color is None else color

    def paint(self, text: str, *styles: str) -> str:
        if not self.color or not styles:
            return text
        return f"\x1b[{';'.join(_STYLES[style] for style in styles)}m{text}\x1b[0m"

    def line(self, text: str = "") -> None:
        print(text, flush=True)

    def chapter(self, number: int, title: str, what: str) -> None:
        self.line()
        self.line(self.paint("━" * self.WIDTH, "dim"))
        self.line(self.paint(f"{number:>2}. {title}", "bold", "blue"))
        for row in textwrap.wrap(what, self.WIDTH - 4):
            self.line(f"    {row}")

    def step(self, text: str) -> None:
        self.line()
        self.line(self.paint(f"  ▸ {text}", "bold"))

    def field(self, label: str, value: object, style: str | None = None) -> None:
        text = str(value)
        rows = [row for raw in text.splitlines() or [""] for row in (textwrap.wrap(raw, self.WIDTH - 22) or [""])]
        for index, row in enumerate(rows[:8]):
            head = self.paint(f"{label:<16}", "dim") if index == 0 else " " * 16
            self.line(f"    {head} {self.paint(row, style) if style else row}")
        if len(rows) > 8:
            self.line(f"    {' ' * 16} {self.paint(f'… {len(rows) - 8} more lines', 'dim')}")

    def verdict(self, verdict: Verdict | None) -> None:
        if verdict is None:
            self.field("verdict", "none: Immune did not screen this call", "dim")
            return
        style = "red" if verdict.blocked else ("yellow" if verdict.hits else "green")
        self.field("verdict", f"{verdict.action.value} (would {verdict.would_action.value})", style)
        for hit in verdict.hits:
            state = "enforced" if hit.enforced else "observed"
            floor = f" {hit.floor}" if hit.floor else ""
            evidence = f" · {hit.evidence[0]}" if hit.evidence else ""
            self.field("", f"{hit.threat} p={hit.probability:.2f} {state}{floor}{evidence}")
        sensor = verdict.sensor
        if sensor.calls:
            self.field(
                "jev", f"{sensor.calls} requests, {sensor.latency_ms:.0f} ms, {sensor.input_tokens:,} tokens", "dim"
            )

    def check(self, ok: bool, text: str) -> None:
        mark = self.paint("✔", "green") if ok else self.paint("✘", "red")
        self.line(f"    {mark} {text}")

    def note(self, text: str) -> None:
        for row in textwrap.wrap(text, self.WIDTH - 6):
            self.line(self.paint(f"    ℹ {row}", "dim"))
