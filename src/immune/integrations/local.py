from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx2

from immune.codecs.openai_chat import OpenAIChatCodec
from immune.core.conversation import ToolCall
from immune.core.runtime import Runtime
from immune.telemetry.verdicts import VerdictIndex
from immune.types import Verdict

_ENDPOINT = "https://immune.local/v1/chat/completions"

__all__ = ["LocalGuard", "LocalOutcome"]


@dataclass(frozen=True, slots=True)
class LocalOutcome:
    verdict: Verdict
    forwarded: dict[str, Any]
    reply: dict[str, Any]

    @property
    def reply_text(self) -> str:
        return str(self.reply["choices"][0]["message"].get("content") or "")

    @property
    def reply_calls(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = self.reply["choices"][0]["message"].get("tool_calls") or []
        return calls

    def forwarded_content(self, index: int) -> str:
        return str(self.forwarded["messages"][index].get("content") or "")


class LocalGuard:
    def __init__(self, runtime: Runtime, model: str = "local") -> None:
        self._runtime = runtime
        self._model = model

    async def screen(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[str] = (),
        reply_text: str = "",
        reply_calls: Sequence[ToolCall] = (),
    ) -> LocalOutcome:
        body: dict[str, Any] = {"model": self._model, "messages": list(messages)}
        if tools:
            body["tools"] = [
                {"type": "function", "function": {"name": name, "description": name.replace("_", " ")}}
                for name in dict.fromkeys(tools)
            ]
        forwarded: dict[str, Any] = {}

        async def upstream(outgoing: httpx2.Request) -> httpx2.Response:
            forwarded.update(json.loads(outgoing.content))
            completion = OpenAIChatCodec.completion(self._model, reply_text, reply_calls, f"local-{uuid.uuid4().hex}")
            return httpx2.Response(200, json=completion, request=outgoing)

        request = httpx2.Request("POST", _ENDPOINT, json=body)
        response = await self._runtime.flow.handle_async(request, upstream, httpx2.Response)
        await response.aread()
        verdict = VerdictIndex.last()
        if verdict is None:
            raise RuntimeError("immune produced no verdict for a local screening call")
        return LocalOutcome(verdict=verdict, forwarded=forwarded or body, reply=response.json())
