from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from immune.errors import ConfigError
from immune.types import Action, Sink, Stage

_ID = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+$")
_KEY = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
_INVARIANT = re.compile(r"^U\d{1,2}$")
_PYTHON = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")
LIBRARY_PREFIX = "immune."
_PLACEHOLDERS = {Stage.DATA: "item", Stage.TOOL: "call"}
_ALLOWED: dict[Sink, frozenset[Action]] = {
    Sink.TEXT: frozenset(
        {Action.ANNOTATE, Action.REDIRECT, Action.REWRITE, Action.REFUSE, Action.HANDOFF, Action.END_SESSION}
    ),
    Sink.SOFTWARE: frozenset({Action.ANNOTATE, Action.REFUSE}),
    Sink.TOOL: frozenset({Action.ANNOTATE, Action.CONFIRM, Action.HOLD, Action.DENY}),
    Sink.DATA: frozenset({Action.ANNOTATE, Action.NEUTRALIZE}),
}
_DEFAULTS: dict[Stage, dict[Sink, Action]] = {
    Stage.INPUT: {Sink.TEXT: Action.REDIRECT, Sink.SOFTWARE: Action.REFUSE},
    Stage.DATA: {Sink.DATA: Action.NEUTRALIZE},
    Stage.TOOL: {Sink.TOOL: Action.HOLD},
    Stage.OUTPUT: {Sink.TEXT: Action.REWRITE, Sink.SOFTWARE: Action.REFUSE},
}
_SINKS_BY_STAGE: dict[Stage, frozenset[Sink]] = {
    Stage.INPUT: frozenset({Sink.TEXT, Sink.SOFTWARE}),
    Stage.DATA: frozenset({Sink.DATA}),
    Stage.TOOL: frozenset({Sink.TOOL}),
    Stage.OUTPUT: frozenset({Sink.TEXT, Sink.SOFTWARE}),
}
DetectorKind = Literal["pattern", "questions", "tool", "python"]
Maturity = Literal["experimental", "stable", "deprecated"]


class VaccineError(ConfigError):
    pass


def describe(error: ValidationError) -> str:
    """A validation error as `field: problem` pairs, for messages that name the field and the fix."""
    problems = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue["loc"]) or "file"
        message = str(issue["msg"]).removeprefix("Value error, ")
        problems.append(f"{location}: {message}")
    return "; ".join(problems)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class VaccineQuestion(_Model):
    key: str
    kind: Literal["noul", "choice"] = "noul"
    text: str
    options: dict[str, str] = Field(default_factory=dict)
    flag: tuple[str, ...] = ()

    @field_validator("key")
    @classmethod
    def _key(cls, value: str) -> str:
        if not _KEY.match(value):
            raise ValueError(
                "use lowercase letters, digits and underscores, starting with a letter (max 41 characters)"
            )
        return value

    @model_validator(mode="after")
    def _choice(self) -> VaccineQuestion:
        if self.kind == "choice":
            if len(self.options) < 2:
                raise ValueError("a choice question needs at least two options")
            if not self.flag or not set(self.flag) <= set(self.options):
                raise ValueError("a choice question needs `flag`: the options that mean the threat is present")
        elif self.options or self.flag:
            raise ValueError("`options` and `flag` only apply to choice questions")
        return self


class VaccineHead(_Model):
    bias: float = 0.0
    weights: dict[str, float] = Field(default_factory=dict)


class ArgumentRule(_Model):
    path: str
    equals: Any = None
    one_of: tuple[Any, ...] = ()
    contains: str | None = None
    greater_than: float | None = None
    less_than: float | None = None

    @model_validator(mode="after")
    def _condition(self) -> ArgumentRule:
        conditions = [self.equals is not None, bool(self.one_of), self.contains is not None]
        conditions += [self.greater_than is not None, self.less_than is not None]
        if sum(conditions) == 0:
            raise ValueError("an argument rule needs one of equals, one_of, contains, greater_than or less_than")
        return self


class Detect(_Model):
    keywords: tuple[str, ...] = ()
    regex: tuple[str, ...] = ()
    questions: tuple[VaccineQuestion, ...] = ()
    head: VaccineHead | None = None
    threshold: float = Field(0.8, gt=0, le=1)
    tool: str | None = None
    argument: ArgumentRule | None = None
    python: str | None = None
    budget_ms: float = Field(50.0, gt=0)
    confirm: Literal["jev"] | None = None

    @property
    def kind(self) -> DetectorKind:
        if self.python is not None:
            return "python"
        if self.questions:
            return "questions"
        if self.keywords or self.regex:
            return "pattern"
        return "tool"

    @model_validator(mode="after")
    def _one_kind(self) -> Detect:
        kinds = [bool(self.keywords or self.regex), bool(self.questions), self.python is not None]
        tool_only = self.tool is not None and not any(kinds)
        if sum(kinds) + int(tool_only) != 1:
            raise ValueError("choose exactly one detector: keywords/regex, questions, tool, or python")
        if self.argument is not None and self.tool is None:
            raise ValueError("`argument` needs `tool`")
        if self.head is not None and not self.questions:
            raise ValueError("`head` only applies to questions")
        if self.confirm is not None and self.questions:
            raise ValueError(
                "`confirm: jev` applies to keywords, regex, tool and python detectors; questions already ask Jev"
            )
        if self.python is not None and not _PYTHON.match(self.python):
            raise ValueError("`python` must look like module.path:function_name")
        if any(not keyword.strip() for keyword in self.keywords):
            raise ValueError("keywords cannot be empty")
        if self.head is not None:
            unknown = set(self.head.weights) - {question.key for question in self.questions}
            if unknown:
                raise ValueError(f"head weights name unknown questions: {', '.join(sorted(unknown))}")
        return self


class Respond(_Model):
    text: Action | None = None
    software: Action | None = None
    tool: Action | None = None
    data: Action | None = None
    message: str | None = None

    def actions(self, stage: Stage) -> dict[Sink, Action]:
        chosen = {sink: getattr(self, sink.value) for sink in Sink if getattr(self, sink.value) is not None}
        return {**_DEFAULTS[stage], **chosen}

    def validate_for(self, stage: Stage) -> None:
        for sink in Sink:
            action = getattr(self, sink.value)
            if action is None:
                continue
            if sink not in _SINKS_BY_STAGE[stage]:
                expected = ", ".join(sorted(item.value for item in _SINKS_BY_STAGE[stage]))
                raise ValueError(f"respond.{sink.value} does not apply to the {stage.value} stage; use {expected}")
            if action not in _ALLOWED[sink]:
                allowed = ", ".join(sorted(item.value for item in _ALLOWED[sink]))
                raise ValueError(f"respond.{sink.value} cannot be {action.value}; use one of {allowed}")


class AppliesTo(_Model):
    sites: tuple[str, ...] = ()
    organs: tuple[str, ...] = ()


class ToolExample(_Model):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class VaccineTests(_Model):
    positives: tuple[str | ToolExample, ...] = ()
    negatives: tuple[str | ToolExample, ...] = ()


class Vaccine(_Model):
    id: str
    version: str = "1.0.0"
    title: str
    description: str = ""
    stage: Stage
    invariant: str = "U0"
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    detect: Detect
    respond: Respond = Field(default_factory=Respond)
    applies_to: AppliesTo = Field(default_factory=AppliesTo)
    enforcement: Literal["observe", "enforce"] = "observe"
    default: Literal["on", "off"] = "on"
    frameworks: tuple[str, ...] = ()
    tests: VaccineTests = Field(default_factory=VaccineTests)
    provenance: dict[str, str] = Field(default_factory=dict)
    maturity: Maturity | None = None
    related: tuple[str, ...] = ()

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError("use a namespaced lowercase id such as acme.no_competitor_mentions")
        return value

    @field_validator("default", mode="before")
    @classmethod
    def _switch(cls, value: object) -> object:
        # YAML 1.1 reads a bare on/off as a boolean.
        if isinstance(value, bool):
            return "on" if value else "off"
        return value

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        if not _VERSION.match(value):
            raise ValueError("use a semantic version such as 1.0.0")
        return value

    @field_validator("invariant")
    @classmethod
    def _invariant(cls, value: str) -> str:
        if not _INVARIANT.match(value):
            raise ValueError("use U0 (custom) or one of U1-U10")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Vaccine:
        if self.stage is Stage.OPERATOR:
            raise ValueError("vaccines apply to the input, data, tool or output stage")
        if self.detect.tool is not None and self.stage is not Stage.TOOL:
            raise ValueError("a tool rule needs `stage: tool`")
        self.respond.validate_for(self.stage)
        self._check_examples()
        placeholder = _PLACEHOLDERS.get(self.stage)
        for question in self.detect.questions:
            if placeholder is not None and f"{{{placeholder}}}" not in question.text:
                raise ValueError(
                    f"question {question.key} must refer to the {placeholder} with {{{placeholder}}} "
                    f"because it runs once per {placeholder}"
                )
            try:
                question.text.format(**({placeholder: "x"} if placeholder else {}))
            except (KeyError, IndexError, ValueError) as error:
                raise ValueError(f"question {question.key} has an unexpected {{placeholder}}: {error}") from error
        return self

    def _check_examples(self) -> None:
        examples = (*self.tests.positives, *self.tests.negatives)
        if self.stage is Stage.TOOL:
            if any(isinstance(example, str) for example in examples):
                raise ValueError("tests for a tool vaccine are tool calls such as {tool: refund_order, arguments: {}}")
        elif any(isinstance(example, ToolExample) for example in examples):
            raise ValueError(f"tests for a {self.stage.value} vaccine are text; tool calls only apply to stage: tool")

    @property
    def library(self) -> bool:
        """Whether the id is in the namespace reserved for the vaccine library that ships with Immune."""
        return self.id.startswith(LIBRARY_PREFIX)

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9_]", "_", self.id)

    def question_key(self, key: str) -> str:
        return f"{self.slug}__{key}"

    @property
    def confirmation(self) -> str:
        """What Jev is asked about each match when ``detect.confirm`` is ``jev``."""
        rule = f"{self.title}: {self.description}" if self.description else self.title
        rule = rule.strip().replace("{", "{{").replace("}", "}}")
        return f"The match described in {{candidate}} is a real case of this rule, not an innocent use: {rule}"
