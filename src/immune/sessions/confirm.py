from __future__ import annotations

import hashlib
import hmac

from immune.core.conversation import Conversation, ToolCall
from immune.sessions.state import ConfirmationBook

_REFERENCE_LENGTH = 10


class ConfirmationSeal:
    def __init__(self, secret: bytes) -> None:
        self._secret = secret

    def reference(self, call: ToolCall, turn: int) -> str:
        message = f"{turn}\n{call.signature}".encode()
        return hmac.new(self._secret, message, hashlib.sha256).hexdigest()[:_REFERENCE_LENGTH]

    def marker(self, call: ToolCall, turn: int) -> str:
        return f"ref {self.reference(call, turn)}"

    def confirmed(self, conversation: Conversation, call: ToolCall) -> bool:
        latest = conversation.latest_user
        if latest is None or latest.turn < 1 or not ConfirmationBook.affirms(latest.text):
            return False
        return self.marker(call, latest.turn - 1) in conversation.assistant_before_latest_user
