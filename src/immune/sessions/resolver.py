from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from immune.core.conversation import Conversation
from immune.state.backend import StateBackend
from immune.state.local import LocalBackend
from immune.telemetry.verdicts import VerdictIndex

_SESSION: contextvars.ContextVar[str | None] = contextvars.ContextVar("immune_session", default=None)
_SITE: contextvars.ContextVar[str | None] = contextvars.ContextVar("immune_site", default=None)


class CallContext:
    @staticmethod
    @contextlib.contextmanager
    def session(session_id: str) -> Iterator[None]:
        token = _SESSION.set(session_id)
        try:
            with VerdictIndex.scope():
                yield
        finally:
            _SESSION.reset(token)

    @staticmethod
    @contextlib.contextmanager
    def site(name: str) -> Iterator[None]:
        token = _SITE.set(name)
        try:
            with VerdictIndex.scope():
                yield
        finally:
            _SITE.reset(token)

    @staticmethod
    def current_session() -> str | None:
        return _SESSION.get()

    @staticmethod
    def current_site() -> str | None:
        return _SITE.get()


class IdentitySource(StrEnum):
    EXPLICIT = "explicit"
    USER = "user"
    CHAIN = "chain"
    ANONYMOUS = "anonymous"


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    session_id: str
    source: IdentitySource
    continued: bool = True

    @property
    def anonymous(self) -> bool:
        return self.source is IdentitySource.ANONYMOUS


class SessionResolver:
    def __init__(self, backend: StateBackend | None = None, ttl_s: float = 86_400.0) -> None:
        self._backend = backend or LocalBackend()
        self._ttl_s = ttl_s

    def resolve(self, conversation: Conversation, trace_id: str, chainable: bool = False) -> SessionIdentity:
        explicit = CallContext.current_session()
        if explicit:
            return SessionIdentity(explicit, IdentitySource.EXPLICIT)
        if conversation.user_hint:
            return SessionIdentity(f"user:{conversation.user_hint}", IdentitySource.USER)
        if conversation.previous_response_id:
            chained = self._backend.get(self._key(conversation.previous_response_id))
            if chained is not None:
                return SessionIdentity(chained.decode("utf-8"), IdentitySource.CHAIN)
            return SessionIdentity(f"chain:{trace_id}", IdentitySource.CHAIN, continued=False)
        if chainable:
            return SessionIdentity(f"chain:{trace_id}", IdentitySource.CHAIN)
        return SessionIdentity(f"anonymous:{trace_id}", IdentitySource.ANONYMOUS)

    def remember(self, response_id: str | None, identity: SessionIdentity) -> None:
        if response_id and not identity.anonymous:
            self._backend.put(self._key(response_id), identity.session_id.encode("utf-8"), self._ttl_s)

    @staticmethod
    def _key(response_id: str) -> str:
        return f"chain:{response_id}"
