from __future__ import annotations

from collections.abc import Mapping, Sequence

from immune.codecs.base import ReplyEdit
from immune.config.spec import Templates
from immune.core.conversation import Reply, Span, ToolCall
from immune.core.decide import Decision
from immune.reflexes.patterns import SpanRemover
from immune.sessions.confirm import ConfirmationSeal
from immune.types import Action, Sink

_SPAN_THREATS = frozenset(
    {"output.exfil_link", "output.unsafe_markup", "output.secret_leak", "output.canary_leak", "output.prompt_copy"}
)
_REPLACING = frozenset({Action.REWRITE, Action.REDIRECT, Action.HANDOFF, Action.END_SESSION})
_TOOL_STOPS = frozenset({Action.DENY, Action.HOLD, Action.CONFIRM})


class ResponseComposer:
    def __init__(self, templates: Templates, seal: ConfirmationSeal, messages: Mapping[str, str] | None = None) -> None:
        self._templates = templates
        self._seal = seal
        self._messages = dict(messages or {})

    def blocking_text(self, decisions: Sequence[Decision], action: Action) -> str:
        for decision in decisions:
            if decision.hit.enforced and decision.hit.action is action and decision.hit.threat in self._messages:
                return self._messages[decision.hit.threat]
        return self.template(action)

    def template(self, action: Action) -> str:
        return {
            Action.REDIRECT: self._templates.redirect,
            Action.REFUSE: self._templates.refuse,
            Action.HANDOFF: self._templates.handoff,
            Action.END_SESSION: self._templates.end_session,
        }.get(action, self._templates.rewrite)

    def compose(
        self,
        reply: Reply,
        output: Sequence[Decision],
        tools: Sequence[Decision],
        sink: Sink,
        calls: Mapping[str, ToolCall],
        interactive: bool,
        augment: bool,
        turn: int = 0,
    ) -> ReplyEdit:
        enforced_output = [decision for decision in output if decision.hit.enforced]
        if sink is Sink.SOFTWARE and any(decision.hit.action is Action.REFUSE for decision in enforced_output):
            return ReplyEdit(
                text=self._templates.refuse,
                refusal=True,
                removed_calls=frozenset(call.call_id for call in reply.tool_calls),
                replaced=True,
            )
        text = self._redacted(reply.text, enforced_output)
        replacing = [
            decision.hit.action
            for decision in enforced_output
            if decision.hit.action in _REPLACING and decision.hit.threat not in _SPAN_THREATS
        ]
        if replacing:
            text = self.blocking_text(enforced_output, Action.most_severe(replacing))
        appended: list[str] = []
        if augment or any(decision.hit.action is Action.AUGMENT for decision in enforced_output):
            appended.append(self._templates.crisis_augment)
            text = f"{text.rstrip()}\n\n{self._templates.crisis_augment}".strip()
        removed, notes = self._tool_outcomes(tools, calls, interactive, turn)
        if notes:
            appended.extend(dict.fromkeys(notes))
            text = "\n\n".join(part for part in (text.strip(), *dict.fromkeys(notes)) if part)
        return ReplyEdit(
            text=text if text != reply.text else None,
            removed_calls=frozenset(removed),
            replaced=bool(replacing),
            appendix="\n\n".join(appended),
        )

    def span_replacements(self, decisions: Sequence[Decision]) -> list[tuple[Span, str]]:
        edits = [
            (decision.assessment.span, self._replacement(decision.hit.threat))
            for decision in decisions
            if decision.assessment.span is not None and decision.hit.threat in _SPAN_THREATS
        ]
        merged = SpanRemover.merge([span for span, _ in edits if span is not None])
        return [
            (span, next(value for edit_span, value in edits if edit_span is not None and span.contains(edit_span)))
            for span in merged
        ]

    def _redacted(self, text: str, decisions: Sequence[Decision]) -> str:
        edits = [
            (decision.assessment.span, self._replacement(decision.hit.threat))
            for decision in decisions
            if decision.assessment.span is not None and decision.hit.threat in _SPAN_THREATS
        ]
        if not edits:
            return text
        merged = SpanRemover.merge([span for span, _ in edits if span is not None])
        for span in reversed(merged):
            replacement = next(
                value for edit_span, value in edits if edit_span is not None and span.contains(edit_span)
            )
            text = text[: span.start] + replacement + text[span.end :]
        return text

    def _replacement(self, threat: str) -> str:
        if threat == "output.exfil_link":
            return self._templates.link_removed
        if threat == "output.unsafe_markup":
            return ""
        return self._templates.redaction

    def _tool_outcomes(
        self, tools: Sequence[Decision], calls: Mapping[str, ToolCall], interactive: bool, turn: int
    ) -> tuple[set[str], list[str]]:
        removed: set[str] = set()
        notes: list[str] = []
        stops: dict[str, Decision] = {}
        for decision in tools:
            subject = decision.assessment.subject
            if subject is None or not decision.hit.enforced or decision.hit.action not in _TOOL_STOPS:
                continue
            if subject not in calls:
                continue
            current = stops.get(subject)
            if current is None or decision.hit.action.severity > current.hit.action.severity:
                stops[subject] = decision
        for subject, decision in stops.items():
            call = calls[subject]
            removed.add(call.call_id)
            notes.append(self._note(call, decision, interactive, turn))
        return removed, notes

    def _note(self, call: ToolCall, decision: Decision, interactive: bool, turn: int) -> str:
        reason = self._messages.get(decision.hit.threat) or self._templates.reason(decision.hit.threat)
        if decision.hit.action is Action.DENY:
            return self._templates.deny_action.format(action=call.describe(), reason=reason)
        if interactive:
            return self._templates.confirm_action.format(
                action=call.describe(), reference=self._seal.marker(call, turn)
            )
        return self._templates.hold_action.format(action=call.describe(), reason=reason)
