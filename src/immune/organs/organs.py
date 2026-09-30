from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import ClassVar

from immune.core.conversation import Channel, Conversation, Reply, ToolCall, ToolSpec
from immune.organs.base import Organ
from immune.profiling.capabilities import ToolCapabilities
from immune.profiling.profile import SiteProfile
from immune.reflexes import ReflexSuite
from immune.reflexes.findings import Finding
from immune.reflexes.tools import Destination, StringLeaves
from immune.state.backend import StateBackend
from immune.state.local import LocalBackend

_DIRECTIVE = re.compile(
    r"(?i)\b(?:always|must|never|instead|before|after|whenever|also)\b[^.\n]{0,120}"
    r"\b(?:send|forward|bcc|cc|copy|include|upload|call|use|pass|redirect|change|replace)\b"
)
_SCHEMA_HOSTS = ("json-schema.org",)
_SENSITIVE = re.compile(
    r"(?i)\b(?:bcc|password|passcode|token|api[ _-]?key|secret|credential|ssh|private key|card number|ssn)\b"
)
_BASELINES = "tool-baselines"
_DESCRIPTION_INSTRUCTIONS = re.compile(
    r"(?i)<\s*(?:important|system|instructions?)\s*>|\bignore (?:all |any )?(?:previous|prior|other)\b|"
    r"\bdo not (?:tell|inform|mention)\b[^.]{0,60}\buser\b|\bbefore (?:using|calling) (?:this|any) tool\b[^.]{0,80}"
    r"\b(?:read|send|include|call)\b|\b(?:send|forward|upload)\b[^.]{0,60}\b(?:ssh|credential|password|token|api key)",
)


class AgentOrgan(Organ):
    name: ClassVar[str] = "agent"

    def __init__(self, reflexes: ReflexSuite, backend: StateBackend | None = None) -> None:
        self._reflexes = reflexes
        self._backend = backend or LocalBackend()
        self._baselines: dict[str, str] = {}
        self._lock = threading.Lock()

    def inspect_request(self, site: str, conversation: Conversation, profile: SiteProfile) -> list[Finding]:
        names = {tool.name for tool in conversation.tools}
        findings: list[Finding] = []
        for tool in conversation.tools:
            text = self._text(tool)
            if self._changed(site, tool.name, text):
                findings.append(self._finding("tool description changed after first use", tool.name, text))
            findings.extend(self._finding(reason, tool.name, text) for reason in self._reasons(tool.name, text, names))
        return findings

    def _reasons(self, name: str, text: str, names: set[str]) -> list[str]:
        reflexes = self._reflexes
        reasons: list[str] = []
        if reflexes.unicode.reveal(text).findings:
            reasons.append("hidden characters inside a tool description")
        if reflexes.hidden_markup.clean(text).findings:
            reasons.append("hidden content inside a tool description")
        if _DESCRIPTION_INSTRUCTIONS.search(text) or any(True for _ in reflexes.template_tokens.findings(text)):
            reasons.append("instructions inside a tool description")
        directive = _DIRECTIVE.search(text) is not None
        destinations = reflexes.destinations.from_text(text) - self._schema_hosts(text)
        others = [other for other in names - {name} if len(other) > 2 and re.search(rf"\b{re.escape(other)}\b", text)]
        if others and directive and (destinations or _SENSITIVE.search(text)):
            reasons.append(f"tool description tries to change how {', '.join(sorted(others))} behaves")
        if directive and destinations:
            reasons.append("tool description directs data to an address")
        return reasons

    def _changed(self, site: str, name: str, text: str) -> bool:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = f"{site}:{name}"
        with self._lock:
            baseline = self._baselines.get(key)
        if baseline is None:
            baseline = self._remember(key, digest)
        return baseline != digest

    def _remember(self, key: str, digest: str) -> str:
        def add(current: bytes | None) -> bytes:
            stored: dict[str, str] = json.loads(current) if current else {}
            stored.setdefault(key, digest)
            return json.dumps(stored, sort_keys=True).encode("utf-8")

        stored = json.loads(self._backend.update(_BASELINES, add))
        with self._lock:
            self._baselines.update(stored)
            return self._baselines[key]

    @staticmethod
    def _text(tool: ToolSpec) -> str:
        leaves = [value for _, value in StringLeaves.of(tool.parameters)]
        return "\n".join([tool.description, *leaves])

    @staticmethod
    def _schema_hosts(text: str) -> set[Destination]:
        return {Destination("host", host) for host in _SCHEMA_HOSTS if host in text}

    @staticmethod
    def _finding(reason: str, tool: str, description: str) -> Finding:
        return Finding("tool.description_poisoning", reason, payload=f"{tool}: {description}", subject=tool)


class CodingOrgan(Organ):
    name: ClassVar[str] = "coding"

    def __init__(self, reflexes: ReflexSuite) -> None:
        self._reflexes = reflexes

    def inspect_call(self, call: ToolCall, capabilities: ToolCapabilities, profile: SiteProfile) -> list[Finding]:
        return list(self._reflexes.commands.findings(call.arguments, executes=capabilities.executes))


class BusinessOrgan(Organ):
    name: ClassVar[str] = "business"

    def __init__(self, reflexes: ReflexSuite) -> None:
        self._reflexes = reflexes

    def inspect_reply(self, conversation: Conversation, reply: Reply, profile: SiteProfile) -> list[Finding]:
        grounding = "\n".join(
            segment.text for segment in conversation.segments if segment.channel in (Channel.OPERATOR, Channel.DATA)
        )
        return list(self._reflexes.numbers.findings(reply.text, grounding))


class PipelineOrgan(Organ):
    name: ClassVar[str] = "pipeline"


class CareOrgan(Organ):
    name: ClassVar[str] = "care"
