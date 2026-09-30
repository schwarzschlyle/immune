from __future__ import annotations

from typing import ClassVar

from immune.core.conversation import Conversation, Reply, ToolCall
from immune.profiling.capabilities import ToolCapabilities
from immune.profiling.profile import SiteProfile
from immune.reflexes.findings import Finding


class Organ:
    name: ClassVar[str]

    def inspect_request(self, site: str, conversation: Conversation, profile: SiteProfile) -> list[Finding]:
        return []

    def inspect_reply(self, conversation: Conversation, reply: Reply, profile: SiteProfile) -> list[Finding]:
        return []

    def inspect_call(self, call: ToolCall, capabilities: ToolCapabilities, profile: SiteProfile) -> list[Finding]:
        return []


class OrganSet:
    def __init__(self, organs: list[Organ]) -> None:
        self._organs = {organ.name: organ for organ in organs}

    def active(self, profile: SiteProfile) -> list[Organ]:
        return [organ for name, organ in self._organs.items() if name in profile.organs]

    def inspect_request(self, site: str, conversation: Conversation, profile: SiteProfile) -> list[Finding]:
        return [
            finding for organ in self.active(profile) for finding in organ.inspect_request(site, conversation, profile)
        ]

    def inspect_reply(self, conversation: Conversation, reply: Reply, profile: SiteProfile) -> list[Finding]:
        return [
            finding for organ in self.active(profile) for finding in organ.inspect_reply(conversation, reply, profile)
        ]

    def inspect_call(self, call: ToolCall, capabilities: ToolCapabilities, profile: SiteProfile) -> list[Finding]:
        return [
            finding for organ in self.active(profile) for finding in organ.inspect_call(call, capabilities, profile)
        ]
