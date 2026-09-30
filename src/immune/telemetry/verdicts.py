from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from immune.types import Verdict

_LOGGER = logging.getLogger("immune")
_LAST: contextvars.ContextVar[Verdict | None] = contextvars.ContextVar("immune_last_verdict", default=None)
# Frameworks such as LangChain run each step in a copy of the context, so a verdict recorded there never reaches the
# caller's _LAST. A site or session block shares one holder with every copy made inside it, and hands the latest
# verdict back to the caller when the block ends.
_SCOPE: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar("immune_verdict_scope", default=None)
_UNSET = object()


class VerdictIndex:
    def __init__(self, capacity: int = 10_000) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[str, Verdict] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, verdict: Verdict, *keys: str | None) -> None:
        with self._lock:
            for key in (verdict.trace_id, *keys):
                if key:
                    self._entries[key] = verdict
                    self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        self.remember_last(verdict)

    def lookup(self, subject: Any) -> Verdict | None:
        for key in self._keys(subject):
            with self._lock:
                verdict = self._entries.get(key)
            if verdict is not None:
                return verdict
        return None

    @staticmethod
    def remember_last(verdict: Verdict | None) -> None:
        _LAST.set(verdict)
        holder = _SCOPE.get()
        if holder is not None:
            holder[0] = verdict

    @staticmethod
    def forget_last() -> None:
        VerdictIndex.remember_last(None)

    @staticmethod
    @contextlib.contextmanager
    def scope() -> Iterator[None]:
        """Make the latest verdict recorded inside the block, in any copy of the context, the caller's latest."""
        holder: list[Any] = [_UNSET]
        token = _SCOPE.set(holder)
        try:
            yield
        finally:
            _SCOPE.reset(token)
            if holder[0] is not _UNSET:
                VerdictIndex.remember_last(holder[0])

    @staticmethod
    def last() -> Verdict | None:
        return _LAST.get()

    def owns(self, verdict: Verdict) -> bool:
        with self._lock:
            return self._entries.get(verdict.trace_id) is verdict

    @staticmethod
    def _keys(subject: Any) -> list[str]:
        if isinstance(subject, str):
            return [subject]
        keys = [getattr(subject, name, None) for name in ("id", "response_id", "_request_id")]
        metadata = getattr(subject, "response_metadata", None)  # LangChain messages keep the provider's id here
        if isinstance(metadata, Mapping):
            keys.append(metadata.get("id"))
        response = getattr(subject, "response", None)
        if response is not None:
            keys.append(getattr(response, "headers", {}).get("x-immune-trace"))
        return [key for key in keys if isinstance(key, str)]


class VerdictRecorder:
    def __init__(self, log_mode: str, verdict_log: Path | None = None) -> None:
        self._log_mode = log_mode
        self._verdict_log = verdict_log
        self._lock = threading.Lock()

    def record(self, verdict: Verdict) -> None:
        if self._log_mode != "off" and verdict.hits:
            _LOGGER.info(
                "immune %s site=%s action=%s would=%s threats=%s%s",
                verdict.trace_id,
                verdict.site,
                verdict.action.value,
                verdict.would_action.value,
                ",".join(sorted(verdict.threats())),
                self._evidence(verdict),
            )
        if self._verdict_log is not None:
            self._append(verdict)

    def _evidence(self, verdict: Verdict) -> str:
        if self._log_mode != "full":
            return ""
        details = "; ".join(f"{hit.threat}: {', '.join(hit.evidence)}" for hit in verdict.hits if hit.evidence)
        return f" evidence={details}" if details else ""

    def _append(self, verdict: Verdict) -> None:
        assert self._verdict_log is not None
        line = json.dumps(verdict.to_dict(), sort_keys=True, default=str) + "\n"
        with self._lock:
            try:
                self._verdict_log.parent.mkdir(parents=True, exist_ok=True)
                with self._verdict_log.open("a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError as error:
                _LOGGER.warning("immune: cannot append to %s: %s", self._verdict_log, error)
