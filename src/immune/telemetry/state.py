from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Any

from immune.state.backend import StateBackend
from immune.state.local import LocalBackend

_LOGGER = logging.getLogger("immune")


class StateStore:
    def __init__(self, backend: StateBackend, location: str | None = None) -> None:
        self._backend = backend
        self._location = location or backend.name

    @classmethod
    def local(cls, directory: Path) -> StateStore:
        return cls(LocalBackend(directory), str(directory))

    @property
    def backend(self) -> StateBackend:
        return self._backend

    @property
    def location(self) -> str:
        return self._location

    def read_json(self, name: str) -> Any:
        raw = self._backend.get(name)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as error:
            _LOGGER.warning("immune: ignoring unreadable state %s: %s", name, error)
            return None

    def write_json(self, name: str, payload: Any) -> None:
        self._backend.put(name, self._encode(payload))

    def update_json(self, name: str, change: Callable[[Any], Any]) -> Any:
        def apply(current: bytes | None) -> bytes:
            return self._encode(change(json.loads(current) if current else None))

        return json.loads(self._backend.update(name, apply))

    def append_line(self, name: str, record: Any) -> None:
        self._backend.append(name, json.dumps(record, sort_keys=True, default=str).encode("utf-8"))

    def read_lines(self, name: str) -> list[Any]:
        return [json.loads(line) for line in self._backend.records(name)]

    def secret(self) -> bytes:
        self._backend.put_if_absent("secret", self._encode({"key": secrets.token_bytes(32).hex()}))
        stored = self.read_json("secret") or {}
        try:
            return bytes.fromhex(stored["key"])
        except (KeyError, TypeError, ValueError):
            _LOGGER.warning("immune: regenerating unreadable deployment secret")
            key = secrets.token_bytes(32)
            self.write_json("secret", {"key": key.hex()})
            return key

    def canary(self) -> str:
        self._backend.put_if_absent("canary", self._encode({"token": f"ref:{secrets.token_hex(6)}"}))
        stored = self.read_json("canary") or {}
        return str(stored.get("token") or "ref:000000000000")

    def flush(self) -> None:
        self._backend.flush()

    def close(self) -> None:
        self._backend.close()

    @staticmethod
    def _encode(payload: Any) -> bytes:
        return json.dumps(payload, indent=2, sort_keys=True, default=str).encode("utf-8")
