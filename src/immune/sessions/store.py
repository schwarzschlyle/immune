from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict

from immune.sessions.state import SessionState
from immune.state.backend import StateBackend
from immune.state.codec import SessionCodec

_LOGGER = logging.getLogger("immune")


class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> SessionState: ...

    def save(self, state: SessionState) -> None:
        return None


class InMemorySessionStore(SessionStore):
    def __init__(self, ttl_s: float = 86_400.0, capacity: int = 50_000) -> None:
        self._ttl_s = ttl_s
        self._capacity = capacity
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()
        self._guard = threading.Lock()

    def get(self, session_id: str) -> SessionState:
        now = time.time()
        with self._guard:
            state = self._sessions.get(session_id)
            if state is None or now - state.last_seen > self._ttl_s:
                state = SessionState(session_id=session_id)
                self._sessions[session_id] = state
            state.last_seen = now
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._capacity:
                self._sessions.popitem(last=False)
            return state


class BackendSessionStore(SessionStore):
    def __init__(self, backend: StateBackend, ttl_s: float = 86_400.0) -> None:
        self._backend = backend
        self._ttl_s = ttl_s

    def get(self, session_id: str) -> SessionState:
        raw = self._backend.get(self._key(session_id))
        state = self._decode(raw) if raw is not None else None
        state = state or SessionState(session_id=session_id)
        state.last_seen = time.time()
        return state

    def save(self, state: SessionState) -> None:
        def merge(current: bytes | None) -> bytes:
            stored = self._decode(current) if current is not None else None
            return SessionCodec.encode(SessionCodec.merge(stored, state) if stored is not None else state)

        self._backend.update(self._key(state.session_id), merge, self._ttl_s)

    @staticmethod
    def _key(session_id: str) -> str:
        return f"session:{session_id}"

    @staticmethod
    def _decode(raw: bytes) -> SessionState | None:
        try:
            return SessionCodec.decode(raw)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            _LOGGER.warning("immune: discarding unreadable session state: %s", error)
            return None
