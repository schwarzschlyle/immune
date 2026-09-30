from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import immune
from immune.core.conversation import ToolCall
from immune.core.runtime import Runtime
from immune.integrations.local import LocalGuard, LocalOutcome
from immune.types import Action

_SITE = "claude-agent"
_BLOCKING = frozenset({Action.REDIRECT, Action.REFUSE, Action.REWRITE, Action.HANDOFF, Action.END_SESSION})
_PROMPT_HISTORY = 20
HookOutput = Any
HookInput = Mapping[str, Any]

__all__ = ["ClaudeAgentHooks", "hooks"]


class ClaudeAgentHooks:
    def __init__(self, guard: LocalGuard, crisis_message: str) -> None:
        self._guard = guard
        self._crisis_message = crisis_message
        self._prompts: dict[str, list[str]] = defaultdict(list)

    async def user_prompt_submit(self, data: HookInput, tool_use_id: str | None, context: Any) -> HookOutput:
        session = str(data.get("session_id", "claude-agent"))
        self._prompts[session] = [*self._prompts[session], str(data.get("prompt", ""))][-_PROMPT_HISTORY:]
        outcome = await self._screen(session, self._user_messages(session))
        if outcome.verdict.action in _BLOCKING:
            return {"decision": "block", "reason": outcome.verdict.explanation}
        if any(hit.action is Action.AUGMENT and hit.enforced for hit in outcome.verdict.hits):
            return {
                "hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": self._crisis_message}
            }
        return {}

    async def pre_tool_use(self, data: HookInput, tool_use_id: str | None, context: Any) -> HookOutput:
        session = str(data.get("session_id", "claude-agent"))
        call = ToolCall(
            call_id=tool_use_id or "tool", name=str(data.get("tool_name")), arguments=dict(data.get("tool_input") or {})
        )
        outcome = await self._screen(session, self._user_messages(session), tools=[call.name], reply_calls=[call])
        if outcome.reply_calls:
            return {}
        stops = [hit for hit in outcome.verdict.enforced_hits if hit.stage.value == "tool"]
        decision = "ask" if stops and all(hit.action is Action.CONFIRM for hit in stops) else "deny"
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision,
                "permissionDecisionReason": outcome.reply_text or outcome.verdict.explanation,
            }
        }

    async def post_tool_use(self, data: HookInput, tool_use_id: str | None, context: Any) -> HookOutput:
        session = str(data.get("session_id", "claude-agent"))
        name = str(data.get("tool_name"))
        result = data.get("tool_response")
        text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        call_id = tool_use_id or "tool"
        messages = [
            *self._user_messages(session),
            {
                "role": "assistant",
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": call_id, "content": text},
        ]
        outcome = await self._screen(session, messages, tools=[name])
        forwarded = outcome.forwarded_content(len(messages) - 1)
        if forwarded == text:
            return {}
        return {
            "decision": "block",
            "reason": forwarded,
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": forwarded,
                "updatedMCPToolOutput": forwarded,
            },
        }

    def matchers(self) -> dict[str, list[Any]]:
        from claude_agent_sdk import HookMatcher

        return {
            "UserPromptSubmit": [HookMatcher(hooks=[self.user_prompt_submit])],
            "PreToolUse": [HookMatcher(hooks=[self.pre_tool_use])],
            "PostToolUse": [HookMatcher(hooks=[self.post_tool_use])],
        }

    async def _screen(
        self,
        session: str,
        messages: list[dict[str, Any]],
        tools: list[str] | None = None,
        reply_calls: list[ToolCall] | None = None,
    ) -> LocalOutcome:
        with immune.session(f"claude:{session}"), immune.site(_SITE):
            return await self._guard.screen(messages, tools or (), reply_calls=reply_calls or ())

    def _user_messages(self, session: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": prompt} for prompt in self._prompts[session]] or [
            {"role": "user", "content": ""}
        ]


def hooks(runtime: Runtime | None = None) -> dict[str, list[Any]]:
    active = runtime or immune.runtime() or immune.init()
    if active is None:
        return {}
    return ClaudeAgentHooks(LocalGuard(active), active.spec.templates.crisis_augment).matchers()
