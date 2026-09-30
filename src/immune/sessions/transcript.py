from __future__ import annotations

import json
import logging
from collections.abc import Callable

from immune.core.conversation import Channel, Conversation, Segment
from immune.state.backend import StateBackend

_LOGGER = logging.getLogger("immune")
_KEPT = (Channel.USER, Channel.DATA, Channel.ASSISTANT)


class TranscriptCache:
    def __init__(
        self,
        backend: StateBackend,
        redact: Callable[[str], str],
        ttl_s: float = 86_400.0,
        segments: int = 16,
        chars: int = 4_000,
    ) -> None:
        self._backend = backend
        self._redact = redact
        self._ttl_s = ttl_s
        self._segments = segments
        self._chars = chars

    def load(self, response_id: str) -> tuple[Segment, ...]:
        raw = self._backend.get(self._key(response_id))
        if raw is None:
            return ()
        try:
            records = json.loads(raw)
            return tuple(
                Segment(
                    channel=Channel(record["channel"]),
                    text=record["text"],
                    origin=record["origin"],
                    locator=("history", index),
                    turn=int(record["turn"]),
                    historical=True,
                )
                for index, record in enumerate(records)
            )
        except (KeyError, TypeError, ValueError) as error:
            _LOGGER.warning("immune: ignoring unreadable transcript for %s: %s", response_id, error)
            return ()

    def save(self, response_id: str, conversation: Conversation, reply: str) -> None:
        segments = [segment for segment in conversation.segments if segment.channel in _KEPT]
        records = [self._record(segment.channel, segment.text, segment.origin, segment.turn) for segment in segments]
        if reply:
            records.append(self._record(Channel.ASSISTANT, reply, "assistant", conversation.latest_turn))
        payload = json.dumps(records[-self._segments :]).encode("utf-8")
        self._backend.put(self._key(response_id), payload, self._ttl_s)

    def _record(self, channel: Channel, text: str, origin: str, turn: int) -> dict[str, object]:
        return {"channel": channel.value, "text": self._redact(text)[: self._chars], "origin": origin, "turn": turn}

    @staticmethod
    def _key(response_id: str) -> str:
        return f"transcript:{response_id}"
