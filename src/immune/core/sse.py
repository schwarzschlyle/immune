from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

DONE = "[DONE]"


@dataclass(frozen=True, slots=True)
class ServerSentEvent:
    data: str
    event: str | None = None

    @property
    def is_done(self) -> bool:
        return self.data.strip() == DONE

    def json(self) -> Any:
        return json.loads(self.data)

    @classmethod
    def of(cls, payload: Any, event: str | None = None) -> ServerSentEvent:
        return cls(data=json.dumps(payload, separators=(",", ":")), event=event)

    @classmethod
    def done(cls) -> ServerSentEvent:
        return cls(data=DONE)


class ServerSentEvents:
    @staticmethod
    def parse(raw: bytes | str) -> list[ServerSentEvent]:
        text = raw.decode("utf-8", "surrogatepass") if isinstance(raw, bytes) else raw
        return list(ServerSentEvents._events(text.replace("\r\n", "\n").split("\n\n")))

    @staticmethod
    def _events(blocks: Iterable[str]) -> Iterator[ServerSentEvent]:
        for block in blocks:
            event_name: str | None = None
            data_lines: list[str] = []
            for line in block.split("\n"):
                if not line or line.startswith(":"):
                    continue
                name, _, value = line.partition(":")
                value = value[1:] if value.startswith(" ") else value
                if name == "event":
                    event_name = value
                elif name == "data":
                    data_lines.append(value)
            if data_lines:
                yield ServerSentEvent(data="\n".join(data_lines), event=event_name)

    @staticmethod
    def serialize(events: Iterable[ServerSentEvent]) -> bytes:
        chunks = []
        for event in events:
            header = f"event: {event.event}\n" if event.event else ""
            chunks.append(f"{header}data: {event.data}\n\n")
        return "".join(chunks).encode("utf-8", "surrogatepass")
