from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

HOME = Path(__file__).resolve().parents[1]
REPO = HOME.parents[1]
LIBRARY = REPO / "scenarios"
IMMUNE_CONFIG = HOME / "immune.yaml"
KNOWLEDGE = HOME / "knowledge"
DATA = HOME / "data"
RUNS = HOME / "runs"
DEFAULT_MODEL = "gpt-5.4-mini"

Tracing = Literal["off", "on", "recorder"]


def load_env(*paths: Path) -> list[str]:
    """Read KEY=value lines into os.environ without printing them or replacing variables that are already set."""
    loaded: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.strip().partition("=")
            if not separator or name.startswith("#") or not name.isidentifier() or name in os.environ:
                continue
            os.environ[name] = value.strip().strip("'\"")
            loaded.append(name)
    return loaded


@dataclass(frozen=True, slots=True)
class Options:
    live: bool
    model: str
    tracing: Tracing
    max_calls: int
    budget_usd: float

    @property
    def jev(self) -> bool:
        return self.live and bool(os.environ.get("TYPESAFE_API_KEY"))

    @classmethod
    def detect(cls, offline: bool = False) -> Options:
        load_env(HOME / ".env", REPO / ".env")
        live = not offline and bool(os.environ.get("OPENAI_API_KEY"))
        langsmith = os.environ.get("LANGSMITH_TRACING", "").lower() in ("1", "true") and bool(
            os.environ.get("LANGSMITH_API_KEY")
        )
        return cls(
            live=live,
            model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
            tracing="on" if langsmith and live else "recorder",
            max_calls=int(os.environ.get("DEMO_MAX_CALLS", "80")),
            budget_usd=float(os.environ.get("DEMO_BUDGET_USD", "3.00")),
        )

    def describe(self) -> str:
        if not self.live:
            return "offline: scripted model and scripted Jev answers, no keys needed"
        jev = "live Jev" if self.jev else "no TYPESAFE_API_KEY: floor candidates act without Jev"
        return f"live: OpenAI {self.model}, {jev}"
