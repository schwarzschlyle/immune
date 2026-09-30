from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from immune.config.spec import PanelFacts, QuestionSpec, Spec
from immune.core.conversation import Conversation, Reply, Segment, ToolCall
from immune.sensing.signals import DEFANGED_SUFFIX

if TYPE_CHECKING:
    from immune.core.candidates import Candidate

_CANDIDATES = "candidates"


@dataclass(frozen=True, slots=True)
class StateLimits:
    operator_chars: int = 4000
    message_chars: int = 6000
    item_chars: int = 4000
    turn_chars: int = 800
    turns: int = 4
    context_chars: int = 6000


@dataclass(frozen=True, slots=True)
class ViewedText:
    raw: str
    defanged: str

    @property
    def split(self) -> bool:
        return self.raw.strip() != self.defanged.strip()


class StateBuilder:
    def __init__(self, redact: Callable[[str], str] | None = None, limits: StateLimits | None = None) -> None:
        self._redact = redact or (lambda text: text)
        self._limits = limits or StateLimits()

    def operator(self, template: str, conversation: Conversation) -> dict[str, Any]:
        return {
            "operator_instructions": self._clip(template, self._limits.operator_chars),
            "tools": [
                {"name": tool.name, "description": self._clip(tool.description, 300)} for tool in conversation.tools
            ],
            "response_format": "structured JSON" if conversation.expects_structure else "free text",
        }

    def input(self, conversation: Conversation, message: ViewedText, provided_content: str | None) -> dict[str, Any]:
        state = self._base(conversation)
        state["recent_conversation"] = self._recent(conversation)
        state["untrusted_user_message"] = self._text(message.raw, self._limits.message_chars)
        if message.split:
            state["untrusted_user_message_defanged"] = self._text(message.defanged, self._limits.message_chars)
        if provided_content:
            state["provided_content"] = self._text(provided_content, self._limits.context_chars)
        return state

    def data(self, conversation: Conversation, items: Sequence[tuple[str, Segment, ViewedText]]) -> dict[str, Any]:
        state = self._base(conversation)
        state["user_request"] = self._latest_user(conversation)
        for item_id, segment, text in items:
            state[item_id] = {"source": segment.origin, "untrusted_text": self._text(text.raw, self._limits.item_chars)}
            if text.split:
                state[f"{item_id}_defanged"] = {
                    "source": segment.origin,
                    "untrusted_text": self._text(text.defanged, self._limits.item_chars),
                }
        return state

    def tool(self, conversation: Conversation, calls: Sequence[tuple[str, ToolCall]]) -> dict[str, Any]:
        state = self._base(conversation)
        state["user_request"] = self._latest_user(conversation)
        state["recent_conversation"] = self._recent(conversation)
        state["data_sources_read"] = sorted({segment.origin for segment in conversation.data})
        for call_id, call in calls:
            tool = conversation.tool(call.name)
            state[call_id] = {
                "tool": call.name,
                "tool_description": self._clip(tool.description if tool else "", 300),
                "arguments": self._redact(call.describe()),
            }
        return state

    def output(self, conversation: Conversation, reply: Reply) -> dict[str, Any]:
        state = self._base(conversation)
        state["user_request"] = self._latest_user(conversation)
        if conversation.data:
            context = "\n\n".join(f"[{segment.origin}] {segment.text}" for segment in conversation.data[-8:])
            state["documents_and_tool_results"] = self._text(context, self._limits.context_chars)
        state["untrusted_assistant_output"] = self._text(reply.text, self._limits.message_chars)
        if reply.tool_calls:
            state["assistant_tool_calls"] = [self._redact(call.describe()) for call in reply.tool_calls]
        return state

    def candidates(
        self, conversation: Conversation, reply: Reply | None, candidates: Sequence[Candidate]
    ) -> dict[str, Any]:
        state = self._base(conversation)
        state["user_request"] = self._latest_user(conversation)
        state["recent_conversation"] = self._recent(conversation)
        if reply is not None and reply.text:
            state["assistant_output"] = self._text(reply.text, self._limits.message_chars)
        for candidate in candidates:
            state[candidate.id] = {
                "check": candidate.threat,
                "where": self._text(candidate.where, 300),
                "found": self._text(candidate.finding.evidence, 300),
                "excerpt": self._text(candidate.excerpt, 800),
                **candidate.hints,
            }
        return state

    def _base(self, conversation: Conversation) -> dict[str, Any]:
        return {"operator_task": self._text(conversation.operator_text, self._limits.operator_chars)}

    def _recent(self, conversation: Conversation) -> list[dict[str, str]]:
        return [
            {"role": segment.channel.value, "text": self._text(segment.text, self._limits.turn_chars)}
            for segment in conversation.recent(self._limits.turns + 1)[:-1]
        ]

    def _latest_user(self, conversation: Conversation) -> str:
        latest = conversation.latest_user
        return self._text(latest.text, self._limits.turn_chars * 2) if latest else ""

    def _text(self, text: str, limit: int) -> str:
        return self._clip(self._redact(text), limit)

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        return text if len(text) <= limit else text[: limit - 1] + "…"


class PanelCompiler:
    def __init__(self, spec: Spec) -> None:
        self._spec = spec

    def questions(
        self, panel: str, facts: PanelFacts, split_view: bool = False, field: str = "untrusted_user_message"
    ) -> list[QuestionSpec]:
        compiled: list[QuestionSpec] = []
        for question in self._spec.panel(panel).applicable(facts):
            if split_view and question.views:
                compiled.append(question.model_copy(update={"text": f"Consider only {field}. {question.text}"}))
                compiled.append(
                    question.model_copy(
                        update={
                            "key": f"{question.key}{DEFANGED_SUFFIX}",
                            "text": f"Consider only {field}_defanged. {question.text}",
                        }
                    )
                )
            else:
                compiled.append(question)
        return compiled

    def item_questions(
        self, panel: str, facts: PanelFacts, placeholder: str, item_ids: Iterable[str], split_items: Iterable[str] = ()
    ) -> list[QuestionSpec]:
        split = set(split_items)
        compiled: list[QuestionSpec] = []
        templates = self._spec.panel(panel).applicable(facts)
        for item_id in item_ids:
            for question in templates:
                compiled.append(question.bound(f"{item_id}__{question.key}", **{placeholder: item_id}))
                if question.views and item_id in split:
                    compiled.append(
                        question.bound(
                            f"{item_id}__{question.key}{DEFANGED_SUFFIX}", **{placeholder: f"{item_id}_defanged"}
                        )
                    )
        return compiled

    def candidate_questions(self, candidates: Iterable[Candidate]) -> list[QuestionSpec]:
        """One question per candidate: the candidate panel questions its threat's head listens to."""
        questions = {question.key: question for question in self._spec.panel(_CANDIDATES).questions}
        compiled: list[QuestionSpec] = []
        for candidate in candidates:
            head = self._spec.heads.get(candidate.threat)
            if head is None:
                continue
            for feature in head.features:
                question = questions.get(feature.signal or "")
                if question is not None:
                    compiled.append(question.bound(f"{candidate.id}__{question.key}", candidate=candidate.id))
        return compiled
