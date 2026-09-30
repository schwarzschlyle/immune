from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from immune.core.conversation import ToolCall
from immune.reflexes.tools import Destination
from immune.sensing.signals import SensorReading
from immune.types import Taint

_AFFIRMATION = re.compile(
    r"^\s*(?:yes|y|yeah|yep|yup|confirm(?:ed)?|proceed|go ahead|ok(?:ay)?|sure|do it|approved?)\b[\s.!]*",
    re.I,
)
_AFFIRMATION_MAX_CHARS = 80


@dataclass(frozen=True, slots=True)
class PendingConfirmation:
    signature: str
    description: str
    expires_at: float


class ConfirmationBook:
    def __init__(self, ttl_s: float = 600.0) -> None:
        self._ttl_s = ttl_s
        self._pending: dict[str, PendingConfirmation] = {}
        self.redeemed: set[str] = set()

    def request(self, call: ToolCall, now: float | None = None) -> PendingConfirmation:
        moment = time.time() if now is None else now
        pending = PendingConfirmation(call.signature, call.describe(), moment + self._ttl_s)
        self._pending[call.signature] = pending
        return pending

    def redeem(self, call: ToolCall, latest_user_text: str, now: float | None = None) -> bool:
        moment = time.time() if now is None else now
        pending = self._pending.get(call.signature)
        if pending is None or pending.expires_at < moment or not self.affirms(latest_user_text):
            return False
        del self._pending[call.signature]
        self.redeemed.add(call.signature)
        return True

    @staticmethod
    def affirms(text: str) -> bool:
        return len(text) <= _AFFIRMATION_MAX_CHARS and bool(_AFFIRMATION.match(text))

    @property
    def pending(self) -> tuple[PendingConfirmation, ...]:
        return tuple(self._pending.values())

    def restore(self, entries: Iterable[PendingConfirmation], now: float | None = None) -> None:
        moment = time.time() if now is None else now
        for entry in entries:
            if entry.expires_at >= moment and entry.signature not in self.redeemed:
                current = self._pending.get(entry.signature)
                if current is None or current.expires_at < entry.expires_at:
                    self._pending[entry.signature] = entry


class SessionRisk:
    def __init__(
        self, prior: float = 0.01, base_rate: float = 0.02, decay: float = 0.8, max_evidence: float = 2.5
    ) -> None:
        self._prior_logit = self._logit(prior)
        self._base_odds = base_rate / (1 - base_rate)
        self._decay = decay
        self._max_evidence = max_evidence
        self.logit = self._prior_logit
        self.updated_at = 0.0

    def update(self, turn_probability: float) -> float:
        probability = min(max(turn_probability, 1e-4), 1 - 1e-4)
        evidence = math.log((probability / (1 - probability)) / self._base_odds)
        bounded = max(-self._max_evidence, min(self._max_evidence, evidence))
        self.logit = self._prior_logit + self._decay * (self.logit - self._prior_logit) + bounded
        self.updated_at = time.time()
        return self.probability

    @property
    def probability(self) -> float:
        return 1 / (1 + math.exp(-self.logit))

    @staticmethod
    def _logit(probability: float) -> float:
        return math.log(probability / (1 - probability))


@dataclass(slots=True)
class SessionState:
    session_id: str
    anonymous: bool = False
    taint: Taint = Taint.CLEAN
    locked: bool = False
    minor: bool = False
    risk: SessionRisk = field(default_factory=SessionRisk)
    confirmations: ConfirmationBook = field(default_factory=ConfirmationBook)
    trusted_destinations: set[Destination] = field(default_factory=set)
    data_destinations: set[Destination] = field(default_factory=set)
    call_counts: Counter[str] = field(default_factory=Counter)
    current_turn: int = -1
    calls_this_turn: int = 0
    last_input_digest: str | None = None
    last_input_reading: SensorReading | None = None
    last_seen: float = field(default_factory=time.time)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def begin_turn(self, turn: int) -> None:
        if turn != self.current_turn:
            self.current_turn = turn
            self.calls_this_turn = 0
            self.call_counts.clear()

    def record_call(self, call: ToolCall) -> int:
        self.calls_this_turn += 1
        self.call_counts[call.signature] += 1
        return self.call_counts[call.signature]

    def taint_with(self, level: Taint) -> None:
        self.taint = self.taint.escalate(level)
