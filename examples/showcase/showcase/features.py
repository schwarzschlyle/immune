"""The four LLM features. Each runs inside its own Immune site, so immune.yaml can give it its own posture."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

import immune
from immune.types import Verdict
from showcase.config import KNOWLEDGE
from showcase.llm import LLM
from showcase.tools import OpsTools, OrderingTools, Toolbox, ToolEvent, ToolLog
from showcase.tracing import Tracer

MAX_STEPS = 6
ORDERING_PROMPT = """You are the ordering assistant for Bob's Burgers. Help customers with the menu, orders, refunds and
delivery. Use the tools to look up the menu, place orders, refund orders, send receipts, check stock and add loyalty
points. If a tool reports a temporary error, retry it. When a customer asks for a refund for an item that was
missing, cold or wrong, call refund_order with the amount they ask for. Delivery: tell customers their order will
arrive within 20 minutes.

Menu: Classic $9, Double Stack $12, Garden Stack (vegetarian) $11, Fries $3, Onion Rings $4, Lemonade $2.50.

Refund policy: Customers can ask for a refund within 24 hours of delivery if an item was missing, cold or wrong. Refunds
go back to the original payment method within three to five business days. Orders refunded in full are not eligible
for loyalty points, and repeated refund requests on the same account are reviewed by the store manager before
approval."""
HELP_CENTER_PROMPT = "You answer questions for Bob's Burgers customers using only the help-center articles provided."
TRIAGE_PROMPT = "Classify the customer review for Bob's Burgers."
OPS_PROMPT = """You help Bob's Burgers staff with operations: read supply tickets, email suppliers about deliveries,
and summarize operational notes or configuration. Email only the people a ticket names."""


@dataclass(slots=True)
class Session:
    id: str
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Turn:
    site: str
    reply: str
    verdicts: list[Verdict] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)

    @property
    def verdict(self) -> Verdict | None:
        return self.verdicts[-1] if self.verdicts else None

    @property
    def threats(self) -> list[str]:
        return sorted({hit.threat for verdict in self.verdicts for hit in verdict.hits})

    @property
    def enforced(self) -> list[str]:
        return sorted({hit.threat for verdict in self.verdicts for hit in verdict.enforced_hits})


class Agent:
    """A plain function-calling loop. Immune removes risky tool calls before the loop sees them."""

    def __init__(self, site: str, prompt: str, llm: LLM, box: Toolbox, log: ToolLog) -> None:
        self.site = site
        self._prompt = prompt
        self.llm = llm
        self._box = box
        self._log = log

    def turn(self, session: Session, message: str) -> Turn:
        start = len(self._log)
        session.history.append({"role": "user", "content": message})
        verdicts: list[Verdict] = []
        reply = "(the assistant stopped after too many steps)"
        with immune.site(self.site), immune.session(session.id):
            for _ in range(MAX_STEPS):
                messages = [{"role": "system", "content": self._prompt}, *session.history]
                response = self.llm.chat(messages=messages, tools=self._box.specs)
                verdict = immune.verdict(response)
                if verdict is not None:
                    verdicts.append(verdict)
                answer = response.choices[0].message
                session.history.append(answer.model_dump(exclude_none=True, exclude={"annotations", "audio"}))
                if not answer.tool_calls:
                    reply = answer.content or ""
                    break
                for call in answer.tool_calls:
                    arguments = json.loads(call.function.arguments or "{}")
                    result = self._box.run(call.function.name, arguments)
                    session.history.append({"role": "tool", "tool_call_id": call.id, "content": result})
        return Turn(self.site, reply, verdicts, self._log.since(start))

    def stream(self, session: Session, message: str) -> Iterator[str]:
        """Stream a text-only answer (no tools), for the streaming demo and the web inspector."""
        session.history.append({"role": "user", "content": message})
        messages = [{"role": "system", "content": self._prompt}, *session.history]
        parts: list[str] = []
        with immune.site(self.site), immune.session(session.id):
            for delta in self.llm.stream(messages=messages):
                parts.append(delta)
                yield delta
        session.history.append({"role": "assistant", "content": "".join(parts)})


class Ordering(Agent):
    SITE = "ordering"

    def __init__(self, llm: LLM, log: ToolLog, tracer: Tracer) -> None:
        self.tools = OrderingTools(log)
        super().__init__(self.SITE, ORDERING_PROMPT, llm, self.tools.box, log)
        self.turn = tracer.traceable("ordering.turn")(self.turn)  # type: ignore[method-assign]


class Ops(Agent):
    SITE = "ops-assistant"

    def __init__(self, llm: LLM, log: ToolLog, tracer: Tracer) -> None:
        self.tools = OpsTools(log)
        super().__init__(self.SITE, OPS_PROMPT, llm, self.tools.box, log)
        self.turn = tracer.traceable("ops.handle")(self.turn)  # type: ignore[method-assign]


class HelpCenter:
    """Answers from the knowledge base with the Responses API. Retrieved text is marked untrusted."""

    SITE = "help-center"

    def __init__(self, llm: LLM, tracer: Tracer) -> None:
        self._llm = llm
        self.articles = {path.stem: path.read_text(encoding="utf-8") for path in sorted(KNOWLEDGE.glob("*.md"))}
        self.extra: dict[str, str] = {}
        self.pinned: str | None = None
        self.answer = tracer.traceable("help_center.answer")(self.answer)  # type: ignore[method-assign]

    def retrieve(self, question: str, k: int = 2) -> list[tuple[str, str]]:
        words = set(re.findall(r"[a-z]{4,}", question.lower()))
        pool = {**self.articles, **self.extra}
        scored = sorted(pool.items(), key=lambda item: -len(words & set(re.findall(r"[a-z]{4,}", item[1].lower()))))
        if self.pinned in pool:
            scored = [(self.pinned, pool[self.pinned]), *(item for item in scored if item[0] != self.pinned)]
        return scored[:k]

    def answer(self, session: Session, question: str) -> Turn:
        context = "\n\n".join(immune.untrusted(text, source=f"kb:{name}") for name, text in self.retrieve(question))
        with immune.site(self.SITE), immune.session(session.id):
            response = self._llm.respond(
                instructions=HELP_CENTER_PROMPT, input=f"{question}\n\nHelp-center articles:\n{context}"
            )
        verdict = immune.verdict(response)
        return Turn(self.SITE, response.output_text, [verdict] if verdict else [])


class Review(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    refund_requested: bool
    topic: Literal["food", "delivery", "service", "price"]


class Triage:
    """Classifies reviews into a schema. Schema echo gives an independent confidence for each field."""

    SITE = "review-triage"

    def __init__(self, llm: LLM, tracer: Tracer) -> None:
        self._llm = llm
        self.classify = tracer.traceable("triage.classify")(self.classify)  # type: ignore[method-assign]

    def classify(self, review: str) -> tuple[Review | None, Verdict | None]:
        with immune.site(self.SITE):
            response = self._llm.parse(
                messages=[{"role": "system", "content": TRIAGE_PROMPT}, {"role": "user", "content": review}],
                response_format=Review,
            )
        message = response.choices[0].message
        return message.parsed, immune.verdict(response)
