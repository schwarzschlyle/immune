from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

from immune.core.conversation import ToolSpec

_TOKEN = re.compile(r"[a-z]+")
_EGRESS = frozenset(
    {
        "send",
        "email",
        "mail",
        "post",
        "publish",
        "upload",
        "http",
        "https",
        "request",
        "fetch",
        "browse",
        "webhook",
        "notify",
        "message",
        "slack",
        "sms",
        "share",
        "tweet",
        "push",
        "submit",
        "transfer",
        "forward",
        "reply",
        "invite",
        "call",
        "dm",
        "chat",
        "discord",
        "telegram",
        "teams",
    }
)
_WRITES = frozenset(
    {
        "create",
        "update",
        "delete",
        "remove",
        "set",
        "write",
        "insert",
        "place",
        "order",
        "book",
        "cancel",
        "refund",
        "pay",
        "transfer",
        "deploy",
        "merge",
        "commit",
        "edit",
        "modify",
        "save",
        "put",
        "patch",
        "add",
        "drop",
        "move",
        "rename",
        "assign",
        "approve",
        "reject",
        "schedule",
        "purchase",
        "buy",
    }
)
_EXECUTES = frozenset(
    {
        "exec",
        "execute",
        "run",
        "shell",
        "bash",
        "command",
        "terminal",
        "python",
        "eval",
        "interpreter",
        "script",
        "sandbox",
    }
)
_PRIVATE = frozenset(
    {
        "customer",
        "account",
        "profile",
        "inbox",
        "email",
        "emails",
        "mail",
        "files",
        "drive",
        "calendar",
        "contacts",
        "crm",
        "database",
        "db",
        "query",
        "record",
        "records",
        "user",
        "users",
        "patient",
        "invoice",
        "payment",
        "secret",
        "credential",
        "ledger",
        "history",
    }
)
_IRREVERSIBLE = frozenset(
    {
        "delete",
        "drop",
        "destroy",
        "pay",
        "transfer",
        "refund",
        "publish",
        "deploy",
        "purchase",
        "buy",
        "cancel",
        "wire",
        "remove",
        "merge",
    }
)
_BUILTIN_EGRESS = frozenset({"web_search", "web_fetch", "fetch", "browser", "computer", "mcp"})

Capability = Literal["egress", "writes_state", "executes", "reads_private", "irreversible"]


@dataclass(frozen=True, slots=True)
class ToolCapabilities:
    egress: bool = False
    writes_state: bool = False
    executes: bool = False
    reads_private: bool = False
    irreversible: bool = False
    configured: bool = False

    def has(self, capability: str) -> bool:
        return bool(getattr(self, capability, False))

    @property
    def names(self) -> frozenset[str]:
        return frozenset(
            name for name in ("egress", "writes_state", "executes", "reads_private", "irreversible") if self.has(name)
        )

    def overridden(self, **values: bool | None) -> ToolCapabilities:
        chosen = {name: value for name, value in values.items() if value is not None}
        return replace(self, **chosen, configured=True) if chosen else self


class CapabilityInferrer:
    def infer(self, tool: ToolSpec) -> ToolCapabilities:
        name_tokens = set(_TOKEN.findall(self._split(tool.name)))
        description_tokens = set(_TOKEN.findall(tool.description.lower()))
        tokens = name_tokens | {token for token in description_tokens if token in _EXECUTES}
        builtin_egress = tool.builtin and any(marker in tool.name.lower() for marker in _BUILTIN_EGRESS)
        executes = bool(tokens & _EXECUTES)
        return ToolCapabilities(
            egress=builtin_egress or bool(name_tokens & _EGRESS),
            writes_state=bool(name_tokens & _WRITES) or executes,
            executes=executes,
            reads_private=bool(name_tokens & _PRIVATE),
            irreversible=bool(name_tokens & _IRREVERSIBLE) or executes,
        )

    @staticmethod
    def _split(name: str) -> str:
        return re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name).lower()
