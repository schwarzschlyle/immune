"""Fake tools. They record what they would do and return plausible results; nothing is sent or charged."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from showcase.config import DATA

MENU = {
    "Classic": 9.00,
    "Double Stack": 12.00,
    "Garden Stack": 11.00,
    "Fries": 3.00,
    "Onion Rings": 4.00,
    "Lemonade": 2.50,
}


@dataclass(frozen=True, slots=True)
class ToolEvent:
    site: str
    tool: str
    arguments: Mapping[str, Any]
    result: str


@dataclass(slots=True)
class ToolLog:
    events: list[ToolEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, event: ToolEvent) -> None:
        with self._lock:
            self.events.append(event)

    def since(self, index: int) -> list[ToolEvent]:
        with self._lock:
            return self.events[index:]

    def __len__(self) -> int:
        return len(self.events)


def schema(
    name: str, description: str, properties: Mapping[str, Any], required: tuple[str, ...] = ()
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": dict(properties), "required": list(required)},
        },
    }


class Toolbox:
    def __init__(
        self, site: str, log: ToolLog, handlers: Mapping[str, Callable[..., str]], specs: list[dict[str, Any]]
    ):
        self.site = site
        self.log = log
        self.specs = specs
        self._handlers = dict(handlers)

    def run(self, name: str, arguments: Mapping[str, Any]) -> str:
        handler = self._handlers.get(name)
        try:
            result = str(handler(**arguments)) if handler is not None else f"unknown tool {name}"
        except TypeError as error:
            result = f"bad arguments for {name}: {error}"
        self.log.add(ToolEvent(self.site, name, dict(arguments), result))
        return result


class OrderingTools:
    """Tools for the ordering assistant. check_stock can be made flaky to demonstrate loop breaking."""

    def __init__(self, log: ToolLog) -> None:
        self.flaky_stock = False
        self.box = Toolbox(
            "ordering",
            log,
            {
                "lookup_menu": self.lookup_menu,
                "place_order": self.place_order,
                "refund_order": self.refund_order,
                "send_receipt": self.send_receipt,
                "check_stock": self.check_stock,
                "update_account": self.update_account,
            },
            [
                schema("lookup_menu", "List menu items and prices", {}),
                schema(
                    "place_order",
                    "Place an order for the customer",
                    {"items": {"type": "array", "items": {"type": "string"}}},
                    ("items",),
                ),
                schema(
                    "refund_order",
                    "Refund an order, fully or partly",
                    {"order_id": {"type": "integer"}, "amount": {"type": "number"}},
                    ("order_id", "amount"),
                ),
                schema(
                    "send_receipt",
                    "Email a receipt for an order",
                    {"order_id": {"type": "integer"}, "email": {"type": "string"}},
                    ("order_id", "email"),
                ),
                schema("check_stock", "Check whether a menu item is in stock", {"item": {"type": "string"}}, ("item",)),
                schema(
                    "update_account",
                    "Add loyalty points to a customer account",
                    {"account": {"type": "string"}, "points": {"type": "integer"}},
                    ("account", "points"),
                ),
            ],
        )

    @staticmethod
    def lookup_menu() -> str:
        return json.dumps(MENU)

    @staticmethod
    def place_order(items: list[str]) -> str:
        total = sum(MENU.get(item, 0.0) for item in items)
        return f"order 5521 placed: {', '.join(items)} (total ${total:.2f})"

    @staticmethod
    def refund_order(order_id: int, amount: float) -> str:
        return f"refund of ${float(amount):.2f} recorded for order {order_id} (demo: no money moved)"

    @staticmethod
    def send_receipt(order_id: int, email: str) -> str:
        return f"receipt for order {order_id} queued to {email} (demo: not sent)"

    def check_stock(self, item: str) -> str:
        if self.flaky_stock:
            return "temporary error: the stock service did not answer, please retry"
        return f"{item} is in stock"

    @staticmethod
    def update_account(account: str, points: int) -> str:
        return f"{points} loyalty points recorded for {account} (demo: not applied)"


class OpsTools:
    """Tools for the internal ops assistant: read a ticket and email a supplier."""

    def __init__(self, log: ToolLog) -> None:
        self.tickets: dict[str, str] = {str(row["id"]): str(row["text"]) for row in _jsonl("tickets.jsonl")}
        self.box = Toolbox(
            "ops-assistant",
            log,
            {"read_ticket": self.read_ticket, "send_email": self.send_email},
            [
                schema(
                    "read_ticket", "Read a support or supply ticket", {"ticket_id": {"type": "string"}}, ("ticket_id",)
                ),
                schema(
                    "send_email",
                    "Send an email",
                    {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
                    ("to", "subject", "body"),
                ),
            ],
        )

    def read_ticket(self, ticket_id: str) -> str:
        return self.tickets.get(ticket_id.upper(), f"no ticket {ticket_id}")

    @staticmethod
    def send_email(to: str, subject: str, body: str) -> str:
        return f"email to {to} sent (demo: not sent)"


def _jsonl(name: str) -> list[dict[str, Any]]:
    path = DATA / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
