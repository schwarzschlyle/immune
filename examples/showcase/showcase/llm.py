"""The app's single door to OpenAI: live, offline (scripted) or replaying a recorded reply, with a budget."""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import openai

import immune
from immune.testing import FakeProvider
from immune.testing.fake_provider import FakeReply, Script
from immune.types import Verdict
from showcase.budget import Ledger
from showcase.config import Options
from showcase.offline import ScriptedModel
from showcase.tracing import Tracer

MAX_OUTPUT_TOKENS = 400


class LLM:
    def __init__(self, options: Options, ledger: Ledger, tracer: Tracer) -> None:
        self._options = options
        self._ledger = ledger
        self.last_stream_verdict: Verdict | None = None
        self._tracer = tracer
        self._replays: list[Any] = []
        self._live: Any = None
        self._offline: Any = None

    @property
    def replaying(self) -> bool:
        return bool(self._replays)

    @contextlib.contextmanager
    def replay(self, script: Script | list[FakeReply]) -> Iterator[None]:
        """Serve recorded model replies instead of calling OpenAI, through the same client code and Immune."""
        self._replays.append(self._fake(script))
        try:
            yield
        finally:
            self._replays.pop()

    def chat(self, **request: Any) -> Any:
        charged = self._before(request)
        response = self._client().chat.completions.create(**request)
        self._after(response, charged)
        return response

    def parse(self, **request: Any) -> Any:
        charged = self._before(request)
        response = self._client().chat.completions.parse(**request)
        self._after(response, charged)
        return response

    def respond(self, **request: Any) -> Any:
        charged = self._before(request, responses=True)
        response = self._client().responses.create(**request)
        self._after(response, charged)
        return response

    def stream(self, **request: Any) -> Iterator[str]:
        """Yield text deltas; the verdict is in ``last_stream_verdict`` once the stream is consumed.

        LangSmith's wrapper iterates streams in a copied context, so ``immune.verdict()`` with no argument can't see
        a streamed call's verdict. Looking it up by a chunk (its response id) works either way.
        """
        charged = self._before(request)
        request["stream"] = True
        request["stream_options"] = {"include_usage": True}
        usage = None
        last = None
        self.last_stream_verdict = None
        for chunk in self._client().chat.completions.create(**request):
            last = chunk
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        if charged:
            self._ledger.charge(usage)
        self.last_stream_verdict = immune.verdict(last) if last is not None else None
        self._ledger.charge_jev(self.last_stream_verdict)

    def _before(self, request: dict[str, Any], responses: bool = False) -> bool:
        charged = self._options.live and not self._replays
        if charged:
            self._ledger.check()
        request.setdefault("model", self._options.model)
        if responses:
            request.setdefault("reasoning", {"effort": "none"})
            request.setdefault("max_output_tokens", MAX_OUTPUT_TOKENS)
        else:
            request.setdefault("reasoning_effort", "none")
            request.setdefault("max_completion_tokens", MAX_OUTPUT_TOKENS)
        return charged

    def _after(self, response: Any, charged: bool) -> None:
        if charged:
            self._ledger.charge(getattr(response, "usage", None))
        self._ledger.charge_jev(immune.verdict(response))

    def _client(self) -> Any:
        if self._replays:
            return self._replays[-1]
        if self._options.live:
            if self._live is None:
                self._live = self._tracer.wrap(openai.OpenAI(max_retries=1))
            return self._live
        if self._offline is None:
            self._offline = self._fake(ScriptedModel())
        return self._offline

    def _fake(self, script: Script | list[FakeReply]) -> Any:
        http = _http_library()
        provider = FakeProvider(script)
        client = openai.OpenAI(
            api_key="sk-offline", max_retries=0, http_client=http.Client(transport=provider.transport(http))
        )
        return self._tracer.wrap(client)


def _http_library() -> ModuleType:
    for base in openai.DefaultHttpxClient.__mro__:
        root = base.__module__.split(".")[0]
        if root in ("httpx", "httpx2"):
            return importlib.import_module(root)
    return importlib.import_module("httpx")
