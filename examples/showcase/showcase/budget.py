from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from immune.types import Verdict

PRICE_INPUT_PER_M = float(os.environ.get("OPENAI_PRICE_INPUT_PER_M", "0.75"))
PRICE_OUTPUT_PER_M = float(os.environ.get("OPENAI_PRICE_OUTPUT_PER_M", "4.50"))
SPEND_FILE = Path(os.environ.get("DEMO_SPEND_FILE", str(Path.home() / ".immune" / "demo-spend.json")))


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Counts OpenAI calls, tokens and dollars, and stops live calls before the budget runs out.

    The dollar total is kept in a small file shared by every run of the showcase and the notebook, so the budget
    covers all runs together. Prices default to gpt-5.4-mini's list prices; override them with
    OPENAI_PRICE_INPUT_PER_M and OPENAI_PRICE_OUTPUT_PER_M.
    """

    def __init__(self, live: bool, max_calls: int, budget_usd: float, spend_file: Path = SPEND_FILE) -> None:
        self.live = live
        self.max_calls = max_calls
        self.budget_usd = budget_usd
        self.spend_file = spend_file
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.jev_tokens = 0
        self.run_usd = 0.0
        self._previous = self._read()
        self._lock = threading.Lock()

    @property
    def total_usd(self) -> float:
        return self._previous + self.run_usd

    def check(self) -> None:
        if not self.live:
            return
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"this run reached DEMO_MAX_CALLS={self.max_calls} OpenAI calls")
        if self.total_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"the demos have spent ${self.total_usd:.4f} of DEMO_BUDGET_USD=${self.budget_usd:.2f} "
                f"(tracked in {self.spend_file})"
            )

    def charge(self, usage: Any) -> None:
        if not self.live or usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0)
        cost = prompt * PRICE_INPUT_PER_M / 1e6 + completion * PRICE_OUTPUT_PER_M / 1e6
        with self._lock:
            self.calls += 1
            self.input_tokens += prompt
            self.output_tokens += completion
            self.run_usd += cost
            self._write()

    def charge_jev(self, verdict: Verdict | None) -> None:
        if verdict is not None:
            with self._lock:
                self.jev_tokens += verdict.sensor.input_tokens

    def summary(self) -> str:
        if not self.live:
            return f"offline: no OpenAI spend · Jev (scripted) tokens {self.jev_tokens:,}"
        return (
            f"{self.calls} OpenAI calls · {self.input_tokens:,} in / {self.output_tokens:,} out tokens · "
            f"${self.run_usd:.4f} this run · ${self.total_usd:.4f} of ${self.budget_usd:.2f} budget used · "
            f"Jev {self.jev_tokens:,} tokens"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "live": self.live,
            "calls": self.calls,
            "max_calls": self.max_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "jev_tokens": self.jev_tokens,
            "run_usd": round(self.run_usd, 6),
            "total_usd": round(self.total_usd, 6),
            "budget_usd": self.budget_usd,
        }

    def _read(self) -> float:
        try:
            return float(json.loads(self.spend_file.read_text(encoding="utf-8"))["usd"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0.0

    def _write(self) -> None:
        self.spend_file.parent.mkdir(parents=True, exist_ok=True)
        record = {"usd": round(self.total_usd, 6), "updated": datetime.now(UTC).isoformat(timespec="seconds")}
        self.spend_file.write_text(json.dumps(record), encoding="utf-8")
