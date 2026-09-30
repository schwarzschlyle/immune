"""The composition root: Immune, the OpenAI client, tracing, the budget and the four features."""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import immune
from immune.sensing.sensor import Sensor
from immune.types import Verdict
from showcase.budget import Ledger
from showcase.config import IMMUNE_CONFIG, RUNS, Options
from showcase.features import HelpCenter, Ops, Ordering, Session, Triage, Turn
from showcase.llm import LLM
from showcase.offline import scripted_sensor
from showcase.tools import ToolLog
from showcase.tracing import Tracer


class Platform:
    def __init__(self, options: Options, sensor: Sensor | None = None, vaccines: list[Path] | None = None) -> None:
        RUNS.mkdir(exist_ok=True)
        self.options = options
        self.ledger = Ledger(options.live, options.max_calls, options.budget_usd)
        self.tracer = Tracer(options.tracing)
        self.verdicts: deque[Verdict] = deque(maxlen=500)
        self.tools = ToolLog()
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.start(sensor, vaccines)
        self.llm = LLM(options, self.ledger, self.tracer)
        self.ordering = Ordering(self.llm, self.tools, self.tracer)
        self.help_center = HelpCenter(self.llm, self.tracer)
        self.triage = Triage(self.llm, self.tracer)
        self.ops = Ops(self.llm, self.tools, self.tracer)

    def start(self, sensor: Sensor | None = None, vaccines: list[Path] | None = None, **config: Any) -> None:
        """(Re)start Immune with immune.yaml, plus any overrides for a demo."""
        immune.shutdown()
        if sensor is None and not self.options.jev:
            sensor = scripted_sensor()
        settings: Any = IMMUNE_CONFIG
        if config:
            from immune.config.loader import SettingsLoader

            base = SettingsLoader().load(IMMUNE_CONFIG).model_dump()
            settings = _merge(base, config)
        immune.init(config=settings, sensor=sensor, vaccines=vaccines)
        immune.on_verdict(self.verdicts.append)

    SITES = ("ordering", "help-center", "ops-assistant")

    def ask(self, site: str, session_id: str, message: str) -> Turn:
        """Send one message to a feature by site name."""
        session = self.session(session_id)
        if site == "help-center":
            return self.help_center.answer(session, message)
        if site == "ops-assistant":
            return self.ops.turn(session, message)
        return self.ordering.turn(session, message)

    def agent(self, site: str) -> Ordering | Ops:
        return self.ops if site == "ops-assistant" else self.ordering

    @contextlib.contextmanager
    def paused(self) -> Iterator[None]:
        """Take Immune out of the process, for demos that build their own private runtimes."""
        immune.shutdown()
        try:
            yield
        finally:
            self.start()

    def session(self, session_id: str) -> Session:
        with self._lock:
            return self._sessions.setdefault(session_id, Session(session_id))

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def close(self) -> None:
        immune.shutdown()
        self.tracer.close()


def _merge(base: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged
