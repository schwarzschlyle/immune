"""Offline stand-ins: a scripted model and scripted Jev answers, so every demo runs without keys.

The scripted replies are ordinary, benign text chosen by keywords. They behave the way a real model often does
(it writes code when asked, names a competitor, invents a discount, echoes a pasted key), which is what makes
Immune act.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from immune.config.spec import QuestionSpec
from immune.core.conversation import Channel, Conversation
from immune.sensing.offline import MockSensor
from immune.testing import FakeReply, FakeToolCall

OFF_TASK = re.compile(r"\b(python|script|poem|essay|code|homework|javascript)\b", re.I)
CODE_LIKE = re.compile(r"(def |print\(|here'?s a poem|return sorted)", re.I)
DISCOUNT = re.compile(r"(\d+\s?% off|free (fries|drink|burger)|discount)", re.I)
KEY = re.compile(r"AKIA[0-9A-Z]{16}")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
CONFIRM = re.compile(r"please confirm: (\w+)\((.*?)\)\. Reply", re.S)
AMOUNT = re.compile(r"\$\s?(\d+(?:\.\d+)?)")
NUMBER = re.compile(r"\b(\d{1,6})\b")
QUOTE = re.compile(r"word for word|verbatim|quote it")
INSTRUCTIONS = re.compile(
    r"\b(you are|you must|never|always|use the tools|call \w+_\w+|tell customers|if a tool)\b", re.I
)


class ScriptedModel:
    def __call__(self, conversation: Conversation) -> FakeReply:
        user = conversation.latest_user.text if conversation.latest_user else ""
        lowered = user.lower()
        tools = {tool.name for tool in conversation.tools}
        if conversation.response_schema is not None:
            return FakeReply(text=json.dumps(self._triage(user)))
        last = conversation.segments[-1] if conversation.segments else None
        if last is not None and last.channel is Channel.DATA and last.origin.startswith("tool"):
            return self._after_tool(last.text, tools, lowered)
        if lowered.strip(" .!") in ("yes", "yes please", "confirm", "yes, go ahead"):
            confirmed = self._confirmed(conversation)
            if confirmed is not None:
                return confirmed
        if tools and not QUOTE.search(lowered):
            call = self._tool_call(lowered, user, tools)
            if call is not None:
                return FakeReply(tool_calls=[call])
        return FakeReply(text=self._text(lowered, user, conversation))

    @staticmethod
    def _triage(review: str) -> dict[str, Any]:
        lowered = review.lower()
        negative = any(word in lowered for word in ("cold", "late", "wrong", "missing", "rude", "never"))
        return {
            "sentiment": "negative" if negative else "positive",
            "refund_requested": "refund" in lowered or "money back" in lowered,
            "topic": "delivery" if "deliver" in lowered or "late" in lowered else "food",
        }

    @staticmethod
    def _after_tool(result: str, tools: set[str], lowered: str) -> FakeReply:
        if "temporary error" in result.lower() and "check_stock" in tools:
            return FakeReply(tool_calls=[FakeToolCall("check_stock", {"item": "Garden Stack"})])
        address = EMAIL.search(result)
        if "send_email" in tools and address and "sent" not in result.lower():
            body = "Hello, please confirm next week's delivery of 400 brioche buns. Thanks, Bob's Burgers"
            return FakeReply(
                tool_calls=[FakeToolCall("send_email", {"to": address.group(0), "subject": "Order", "body": body})]
            )
        return FakeReply(text=f"Done. {result[:160]}")

    @staticmethod
    def _confirmed(conversation: Conversation) -> FakeReply | None:
        for segment in reversed(conversation.segments):
            match = CONFIRM.search(segment.text)
            if match is None:
                continue
            arguments: dict[str, Any] = {}
            for name, raw in re.findall(r"(\w+)=('[^']*'|[^,]+)", match.group(2)):
                arguments[name] = _value(raw.strip())
            return FakeReply(tool_calls=[FakeToolCall(match.group(1), arguments)])
        return None

    @staticmethod
    def _tool_call(lowered: str, user: str, tools: set[str]) -> FakeToolCall | None:
        numbers = [int(value) for value in NUMBER.findall(user)]
        if "refund" in lowered and "refund_order" in tools:
            amount = AMOUNT.search(user)
            order = next((value for value in numbers if not amount or value != int(float(amount.group(1)))), 1001)
            return FakeToolCall("refund_order", {"order_id": order, "amount": float(amount.group(1)) if amount else 10})
        if "receipt" in lowered and "send_receipt" in tools:
            address = EMAIL.search(user)
            return FakeToolCall(
                "send_receipt",
                {"order_id": numbers[0] if numbers else 1001, "email": address.group(0) if address else ""},
            )
        if ("stock" in lowered or "available" in lowered) and "check_stock" in tools:
            return FakeToolCall("check_stock", {"item": "Garden Stack"})
        if "vip-" in lowered and "update_account" in tools:
            account = re.search(r"vip-\d+", lowered)
            return FakeToolCall("update_account", {"account": account.group(0) if account else "vip-1", "points": 500})
        if "ticket" in lowered and "read_ticket" in tools:
            ticket = re.search(r"t-\d+", lowered)
            return FakeToolCall("read_ticket", {"ticket_id": ticket.group(0).upper() if ticket else "T-1"})
        if "menu" in lowered and "lookup_menu" in tools:
            return FakeToolCall("lookup_menu", {})
        return None

    @staticmethod
    def _text(lowered: str, user: str, conversation: Conversation) -> str:
        if "articles:" in lowered:
            articles = user[lowered.index("articles:") + len("articles:") :]
            lines = [line.strip("#- ") for line in re.sub(r"<[^>]+>", "", articles).splitlines()]
            return "From our help center: " + " ".join(line for line in lines if line)[:220]
        if OFF_TASK.search(user):
            return "Sure! Here's a Python function:\n\ndef sort_numbers(values):\n    return sorted(values)"
        if "burger palace" in lowered or "competitor" in lowered or "compare" in lowered:
            return "I can't speak for Burger Palace, but our fries are hand-cut and cooked to order."
        if "discount" in lowered or "student" in lowered or "deal" in lowered:
            return "Good news: students get 20% off every Tuesday, and your first order comes with free fries!"
        key = KEY.search(user)
        if key:
            return f"This config connects the order service to the storage bucket using the key {key.group(0)}."
        if QUOTE.search(lowered):
            text = conversation.operator_text
            policy = text.find("Refund policy:")
            return text[policy:] if "refund policy" in lowered and policy >= 0 else text[:600]
        if ("fast" in lowered or "how long" in lowered or "quick" in lowered) and "deliver" in lowered:
            return "Your order will be delivered in 15 minutes, guaranteed."
        return "The Classic is $9 and comes with fries. The Garden Stack is $11. Anything else?"


def _value(raw: str) -> Any:
    if raw.startswith("'"):
        return raw.strip("'")
    for kind in (int, float):
        try:
            return kind(raw)
        except ValueError:
            continue
    return raw


def scripted_signals(state: Mapping[str, Any], question: QuestionSpec) -> float | str | None:
    key = question.key
    if key.startswith("echo__"):
        field = key.removeprefix("echo__").split("__")[0]
        value = ScriptedModel._triage(str(state.get("untrusted_user_message") or "")).get(field)
        if question.kind == "noul":
            return 0.95 if value else 0.05
        return str(value).lower() if isinstance(value, bool) else value
    user = str(state.get("untrusted_user_message") or "")
    output = str(state.get("untrusted_assistant_output") or "")
    candidate = _candidate(state, key)
    if candidate is not None:
        return _decide_candidate(key, candidate)
    if key.startswith("off_task") and OFF_TASK.search(user):
        return 0.95
    if key.startswith("task_fidelity") and CODE_LIKE.search(output):
        return 0.05
    if "promises_discount" in key:
        return 0.95 if DISCOUNT.search(output) else 0.03
    if key.startswith("bobs_no_delivery_promises"):
        return 0.95 if re.search(r"\d+ minutes", output) else 0.03
    if key.startswith("makes_commitment") and DISCOUNT.search(output):
        return 0.9
    if key.startswith("commitment_supported") and DISCOUNT.search(output):
        return 0.1
    return None


def _candidate(state: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    name = key.split("__", 1)[0]
    entry = state.get(name) if name.startswith("cand_") else None
    return entry if isinstance(entry, Mapping) else None


def _decide_candidate(key: str, candidate: Mapping[str, Any]) -> float | None:
    """What Jev would say about a nominated candidate; anything not listed is confirmed, like MockSensor's default."""
    marked = str(candidate.get("excerpt", "")).partition("«")[2].partition("»")[0]
    if key.endswith("__copy_confidential"):
        return 0.9 if INSTRUCTIONS.search(marked) else 0.05
    if key.endswith("__secret_real"):
        return 0.1 if candidate.get("looks_like_example_or_placeholder") else 0.95
    return None


def scripted_sensor() -> MockSensor:
    return MockSensor(rules=[scripted_signals])
