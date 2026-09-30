"""A guided tour of how Immune works: every feature and setting, with detailed logs in your terminal.

    python examples/08_feature_tour.py                          # the whole tour
    python examples/08_feature_tour.py --list                   # list the demos
    python examples/08_feature_tour.py --only modes,streaming   # run some of them
    python examples/08_feature_tour.py --quiet                  # hide Immune's own log lines
    python examples/08_feature_tour.py --live                   # also run the live Jev chapter
    python examples/08_feature_tour.py --live --only live       # only the live Jev chapter

By default it runs offline: model traffic goes to immune.testing's scripted provider and TypeSafe Jev's answers come
from a scripted sensor, so no API keys or network are needed. --live adds a chapter that sends everyday examples to the
real Jev service (TYPESAFE_API_KEY from the environment or from .env) and logs every exchange: what Jev saw, each
question, each answer, latency and billed tokens. The running example is a burger-ordering bot; the policy it enforces
is staying on task. Every demo checks its own outcome, and the run exits non-zero if any check fails. In the live
chapter, a check that depends on Jev's judgement is reported as a note when Jev answers differently.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib
import importlib.util
import itertools
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar
from unittest import mock

import anthropic
import openai
import yaml
from pydantic import BaseModel

import immune
from immune.config.settings import Settings, SiteSettings, ToolSettings
from immune.config.spec import QuestionSpec, Spec
from immune.core.runtime import Runtime
from immune.sensing.sensor import Sensor
from immune.sensing.signals import SensorReading
from immune.testing import (
    FakeBedrock,
    FakeProvider,
    FakeReply,
    FakeToolCall,
    ImmuneHarness,
    JevWireStub,
    LangSmithRecorder,
    MockSensor,
    RecordingSensor,
    ReplaySensor,
)
from immune.types import Action, Mode, Verdict

SPEC = Spec.default()
MODEL = "gpt-5.5"
OPERATOR = "You are the ordering assistant for Bob's Burgers. Help customers with the menu, orders and delivery."
MENU_ANSWER = "The Classic is $9 and comes with fries. The Double Stack is $12."
OFF_TOPIC = "Can you write me a Python script that sorts a list of numbers?"
OFF_TOPIC_ANSWER = "Sure! numbers = [3, 1, 2]; print(sorted(numbers))"
OFF_TASK = {"off_task": 0.95, "task_fidelity": 0.05}
STORY = (
    "Our burgers are made fresh every morning with local beef and brioche buns from the bakery down the street. "
    "The Classic comes with lettuce, tomato and our house sauce, and the Double Stack adds a second patty. "
)


class Palette:
    CODES: ClassVar[dict[str, str]] = {
        "bold": "1",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "grey": "90",
    }

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        return f"\x1b[{';'.join(self.CODES[style] for style in styles)}m{text}\x1b[0m"


class Terminal:
    WIDTH = 112
    LABEL = 15
    MAX_LINES = 16

    def __init__(self, palette: Palette) -> None:
        self.paint = palette

    def write(self, text: str = "") -> None:
        print(text, flush=True)

    def rule(self, character: str, style: str) -> None:
        self.write(self.paint(character * self.WIDTH, style))

    def chapter(self, title: str) -> None:
        self.write()
        self.write(self.paint(f"■ {title.upper()}", "bold", "blue"))

    def section(self, number: int, total: int, slug: str, title: str, what: str) -> None:
        self.write()
        self.rule("━", "grey")
        counter = self.paint(f"{number:>2}/{total}", "grey")
        self.write(f"{counter}  {self.paint(title, 'bold')}  {self.paint(f'[{slug}]', 'grey')}")
        self.rule("━", "grey")
        self.paragraph(what, "  ")

    def paragraph(self, text: str, prefix: str, style: str | None = None) -> None:
        for line in textwrap.wrap(text, self.WIDTH - len(prefix)):
            self.write(prefix + (self.paint(line, style) if style else line))

    def step(self, text: str) -> None:
        self.write()
        for index, line in enumerate(textwrap.wrap(text, self.WIDTH - 4)):
            marker = self.paint("▸", "cyan") if index == 0 else " "
            self.write(f"  {marker} {self.paint(line, 'bold')}")

    def call(self, text: str) -> None:
        self.write(f"    {self.paint('»', 'magenta')} {self.paint(text, 'magenta')}")

    def field(self, label: str, value: object, style: str | None = None) -> None:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        width = self.WIDTH - 4 - self.LABEL - 1
        lines = [piece for raw in text.splitlines() or [""] for piece in (textwrap.wrap(raw, width) or [""])]
        head = "    " + self.paint(label.ljust(self.LABEL), "grey")
        pad = " " * (4 + self.LABEL)
        for index, line in enumerate(lines[: self.MAX_LINES]):
            self.write((head if index == 0 else pad) + " " + (self.paint(line, style) if style else line))
        if len(lines) > self.MAX_LINES:
            self.write(f"{pad} {self.paint(f'… {len(lines) - self.MAX_LINES} more lines', 'grey')}")

    def block(self, title: str, data: object) -> None:
        self.write(f"    {self.paint(title, 'grey')}")
        dumped = yaml.safe_dump(_plain(data), sort_keys=False, allow_unicode=True, width=self.WIDTH - 8)
        for line in dumped.rstrip().splitlines():
            self.write(f"      {self.paint(line, 'yellow')}")

    def output(self, lines: Sequence[str]) -> None:
        for line in lines:
            self.write(f"      {self.paint('│', 'grey')} {line}")

    def note(self, text: str) -> None:
        self.paragraph(text, "    ℹ ", "grey")

    def check(self, ok: bool, text: str) -> None:
        mark = self.paint("✔", "green", "bold") if ok else self.paint("✘", "red", "bold")
        self.write(f"    {mark} {self.paint(text, 'green' if ok else 'red')}")

    def differ(self, text: str) -> None:
        self.write(f"    {self.paint('⚠', 'yellow', 'bold')} {self.paint(f'Jev judged differently: {text}', 'yellow')}")

    def table(self, rows: Sequence[Mapping[str, object]]) -> None:
        if not rows:
            return
        headers = list(rows[0])
        cells = [[str(row.get(header, "")) for header in headers] for row in rows]
        widths = [max(len(header), *(len(line[index]) for line in cells)) for index, header in enumerate(headers)]
        self.write("      " + self.paint("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)), "bold"))
        self.write("      " + self.paint("  ".join("─" * w for w in widths), "grey"))
        for line in cells:
            self.write("      " + "  ".join(value.ljust(width) for value, width in zip(line, widths, strict=True)))


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class ImmuneLogs(logging.Handler):
    STYLES: ClassVar[dict[int, str]] = {logging.WARNING: "yellow", logging.ERROR: "red"}

    def __init__(self, terminal: Terminal, quiet: bool) -> None:
        super().__init__(logging.INFO)
        self.terminal = terminal
        self.quiet = quiet
        self.records: list[logging.LogRecord] = []

    def install(self) -> None:
        logger = logging.getLogger("immune")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers = [self]

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        if self.quiet:
            return
        paint = self.terminal.paint
        style = self.STYLES.get(record.levelno, "grey")
        message = record.getMessage().removeprefix("immune: ")
        head = paint(f"    immune │ {record.levelname:<7}", style)
        for index, line in enumerate(textwrap.wrap(message, self.terminal.WIDTH - 23) or [""]):
            self.terminal.write((head if index == 0 else " " * 22) + " " + paint(line, style))
        if record.exc_info and record.exc_info[1] is not None:
            error = record.exc_info[1]
            self.terminal.write(" " * 23 + paint(f"{type(error).__name__}: {error}", style))

    def since(self, count: int) -> list[str]:
        return [record.getMessage() for record in self.records[count:]]


class Http:
    @staticmethod
    def for_sdk(sdk: ModuleType) -> ModuleType:
        for base in sdk.DefaultHttpxClient.__mro__:
            root = base.__module__.split(".")[0]
            if root in ("httpx", "httpx2"):
                return importlib.import_module(root)
        return importlib.import_module("httpx")


class Model:
    def __init__(self, script: FakeReply | str | Callable[[Any], FakeReply]) -> None:
        self.said: list[FakeReply] = []
        reply = FakeReply(script) if isinstance(script, str) else script
        self._script = reply if callable(reply) and not isinstance(reply, FakeReply) else (lambda _: reply)
        self.provider = FakeProvider(self._speak)

    def _speak(self, conversation: Any) -> FakeReply:
        reply = self._script(conversation)
        self.said.append(reply)
        return reply

    def openai(self, asynchronous: bool = False, base_url: str | None = None, **http_options: Any) -> Any:
        http = Http.for_sdk(openai)
        options: dict[str, Any] = {"api_key": "sk-demo", "max_retries": 0, "base_url": base_url}
        if asynchronous:
            transport = http.AsyncClient(transport=self.provider.async_transport(http), **http_options)
            return openai.AsyncOpenAI(http_client=transport, **options)
        client = http.Client(transport=self.provider.transport(http), **http_options)
        return openai.OpenAI(http_client=client, **options)

    def anthropic(self) -> Any:
        http = Http.for_sdk(anthropic)
        client = http.Client(transport=self.provider.transport(http))
        return anthropic.Anthropic(api_key="sk-ant-demo", http_client=client, max_retries=0)

    def gemini(self) -> Any:
        genai = importlib.import_module("google.genai")
        types = importlib.import_module("google.genai.types")
        http = importlib.import_module("httpx")
        options = types.HttpOptions(
            httpx_client=http.Client(transport=self.provider.transport(http)),
            httpx_async_client=http.AsyncClient(transport=self.provider.async_transport(http)),
        )
        return genai.Client(api_key="gemini-demo", http_options=options)


class Wire:
    @classmethod
    def request(cls, body: Mapping[str, Any]) -> list[tuple[str, str]]:
        lines = [
            ("system", cls.text(body[key])) for key in ("system", "instructions", "systemInstruction") if body.get(key)
        ]
        for message in body.get("messages", []):
            calls = [
                f"calls {call['function']['name']}({call['function']['arguments']})"
                for call in message.get("tool_calls") or []
            ]
            text = " ".join(part for part in [cls.text(message.get("content")), *calls] if part)
            lines.append((message.get("role", "?"), text))
        entries = body.get("input")
        if isinstance(entries, str):
            lines.append(("user", entries))
        elif isinstance(entries, list):
            lines.extend((item.get("role") or item.get("type", "item"), cls.text(item)) for item in entries)
        lines.extend(
            (content.get("role", "user"), cls.text(content.get("parts"))) for content in body.get("contents", [])
        )
        tools = [cls._tool(tool) for tool in body.get("tools") or body.get("toolConfig", {}).get("tools", [])]
        if tools:
            lines.append(("tools", ", ".join(tools)))
        return lines

    @staticmethod
    def _tool(tool: Mapping[str, Any]) -> str:
        return str(tool.get("name") or tool.get("function", {}).get("name") or tool.get("toolSpec", {}).get("name"))

    @classmethod
    def text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(part for part in (cls.text(item) for item in value) if part)
        if isinstance(value, Mapping):
            kind = value.get("type")
            if kind == "tool_use":
                return f"tool_use {value.get('name')}({json.dumps(value.get('input', {}))})"
            if kind == "tool_result":
                return f"tool_result: {cls.text(value.get('content'))}"
            for key in ("text", "content", "parts"):
                if key in value:
                    return cls.text(value[key])
            return json.dumps(value, default=str)
        return str(value)

    @classmethod
    def delivered(cls, response: Any) -> str:
        if isinstance(response, str):
            return response
        if isinstance(response, Mapping):
            return cls.text(response.get("output", {}).get("message", {}).get("content", []))
        if hasattr(response, "choices"):
            message = response.choices[0].message
            parts = [message.content or "", f"(refusal) {message.refusal}" if message.refusal else ""]
            parts += [
                f"[tool call] {call.function.name}({call.function.arguments})" for call in message.tool_calls or []
            ]
            return " ".join(part for part in parts if part)
        if hasattr(response, "output_text"):
            return str(response.output_text)
        if hasattr(response, "candidates"):
            return str(response.text or "")
        if hasattr(response, "content"):
            return " ".join(
                block.text if block.type == "text" else f"[tool call] {block.name}({json.dumps(block.input)})"
                for block in response.content
            )
        return repr(response)


class VerdictView:
    def __init__(self, terminal: Terminal) -> None:
        self.terminal = terminal

    def show(self, verdict: Verdict | None) -> None:
        t = self.terminal
        if verdict is None:
            t.field("⚖ verdict", "none: Immune did not screen this call", "yellow")
            return
        style = "green" if verdict.action is Action.ALLOW else "red"
        t.field(
            "⚖ verdict",
            f"action={verdict.action.value}  would_action={verdict.would_action.value}  trace={verdict.trace_id}",
            style,
        )
        t.field(
            "  context",
            f"site={verdict.site}  session={verdict.session_id}  taint={verdict.taint.value}  "
            f"session_risk={verdict.session_risk:.2f}",
        )
        sensor = verdict.sensor
        t.field(
            "  sensor",
            f"{sensor.name}  model={sensor.model}  calls={sensor.calls}  latency={sensor.latency_ms:.1f} ms  "
            f"input_tokens={sensor.input_tokens}",
        )
        t.field("  versions", f"spec={verdict.spec_version}  config={verdict.config_hash}")
        for hit in verdict.hits:
            threat = SPEC.threats.get(hit.threat)
            floor = f"  floor {threat.floor.id}" if threat is not None and threat.floor is not None else ""
            state = "ENFORCED" if hit.enforced else "observed"
            t.field(
                "  hit",
                f"{hit.threat}  stage={hit.stage.value}  p={hit.probability:.2f}  action={hit.action.value}  "
                f"{state}  invariant={hit.invariant}{floor}",
                "red" if hit.enforced else "yellow",
            )
            for evidence in hit.evidence:
                t.field("    evidence", evidence, "grey")
        for name, distribution in verdict.echo.items():
            t.field("  echo", f"{name}: " + ", ".join(f"{option}={p:.2f}" for option, p in distribution.items()))
        t.field("  why", verdict.explanation or "-")


class Env:
    def __init__(self, **values: str) -> None:
        self._values = values
        self._saved: dict[str, str | None] = {}

    def __enter__(self) -> Env:
        for key, value in self._values.items():
            self._saved[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc_info: object) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LiveKey:
    NAME = "TYPESAFE_API_KEY"

    @classmethod
    def find(cls) -> tuple[str | None, str]:
        if os.environ.get(cls.NAME):
            return os.environ[cls.NAME], "the environment"
        for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
            if candidate.is_file():
                for line in candidate.read_text(encoding="utf-8").splitlines():
                    name, _, value = line.strip().partition("=")
                    if name.strip() == cls.NAME and value.strip():
                        return value.strip().strip("'\""), str(candidate)
        return None, ""


class LiveLedger:
    def __init__(self) -> None:
        self.requests: list[tuple[int, float, int, bool]] = []


class LiveJev(Sensor):
    name: ClassVar[str] = "jev"
    MODEL = "jev-1.13.0"

    def __init__(self, api_key: str, ledger: LiveLedger) -> None:
        from immune.sensing.jev import JevSensor

        self._inner = JevSensor(self.MODEL, timeout_s=5.0, api_key=api_key)
        self._ledger = ledger
        self._lock = threading.Lock()
        self._pending: list[tuple[Mapping[str, Any], Sequence[QuestionSpec], SensorReading | None, float]] = []
        self.seen: list[Mapping[str, Any]] = []

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        started = time.perf_counter()
        try:
            reading = await self._inner.read(state, questions)
        except BaseException:
            self._record(state, questions, None, (time.perf_counter() - started) * 1000)
            raise
        self._record(state, questions, reading, reading.latency_ms)
        return reading

    def _record(
        self, state: Mapping[str, Any], questions: Sequence[QuestionSpec], reading: SensorReading | None, ms: float
    ) -> None:
        with self._lock:
            self._pending.append((state, questions, reading, ms))
            self.seen.append(state)
            tokens = reading.input_tokens if reading is not None else 0
            self._ledger.requests.append((len(questions), ms, tokens, reading is not None))

    async def close(self) -> None:
        await self._inner.close()

    def reset(self) -> None:
        self._inner.reset()

    def report(self, terminal: Terminal) -> None:
        with self._lock:
            pending, self._pending = self._pending, []
        for state, questions, reading, ms in pending:
            if reading is None:
                terminal.field(
                    "⇄ Jev", f"{len(questions)} questions, no answer after {ms:.0f} ms (timed out)", "yellow"
                )
                continue
            summary = (
                f"{len(questions)} questions, {ms:.0f} ms, {reading.input_tokens} billed tokens, model {reading.model}"
            )
            terminal.field("⇄ Jev", summary, "magenta")
            for key, value in state.items():
                terminal.field(f"  saw {key}", self._clip(Wire.text(value), 150), "grey")
            for question in questions:
                answer = self.answer(reading.get(question.key))
                terminal.field(f"  {question.key}", f"{self._clip(question.text, 72)} → {answer}")

    @staticmethod
    def answer(signal: Any) -> str:
        if signal is None:
            return "no answer"
        if signal.kind == "choice":
            ranked = sorted(signal.distribution.items(), key=lambda item: -item[1])[:3]
            return ", ".join(f"{option} {probability:.2f}" for option, probability in ranked)
        if signal.kind == "score":
            return f"level {signal.value}"
        return f"{signal.probability:.2f}"

    @staticmethod
    def _clip(text: str, limit: int) -> str:
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"


class Tour:
    def __init__(self, terminal: Terminal, logs: ImmuneLogs, work: Path, live_key: str | None = None) -> None:
        self.terminal = terminal
        self.logs = logs
        self.work = work
        self.view = VerdictView(terminal)
        self.checks: list[tuple[str, bool, str]] = []
        self.notes: list[tuple[str, str]] = []
        self.current = "tour"
        self.demo_started = 0
        self.ledger = LiveLedger()
        self._live_key = live_key
        self._jev: LiveJev | None = None
        self._counter = itertools.count(1)

    def jev(self) -> LiveJev:
        if self._live_key is None:
            raise RuntimeError("the live chapter needs --live and TYPESAFE_API_KEY")
        self._jev = LiveJev(self._live_key, self.ledger)
        return self._jev

    def live_report(self) -> None:
        if self._jev is not None:
            self._jev.report(self.terminal)

    def judge(self, ok: object, message: str) -> bool:
        if ok:
            return self.expect(True, message)
        timed_out = any("sensor exceeded" in line for line in self.logs.since(self.demo_started))
        note = f"Jev did not answer in time, so only floor candidates acted: {message}" if timed_out else message
        self.notes.append((self.current, note))
        if timed_out:
            self.terminal.write(f"    {self.terminal.paint('⚠ ' + note, 'yellow')}")
        else:
            self.terminal.differ(message)
        return False

    def state_dir(self, name: str = "state") -> Path:
        return self.work / f"{next(self._counter):03d}-{self.current}-{name}"

    def start(
        self,
        *,
        mode: str | None = None,
        config: Mapping[str, Any] | Path | None = None,
        sensor: Sensor | None = None,
        backend: Any = None,
    ) -> Runtime:
        sensor = sensor if sensor is not None else MockSensor()
        state_dir = self.state_dir()
        arguments = [f"mode={mode!r}" if mode else "", "config=…" if config else ""]
        arguments += [f"sensor={type(sensor).__name__}(…)", f"state_dir='{state_dir.name}'"]
        arguments += [f"backend={type(backend).__name__}(…)"] if backend is not None else []
        self.terminal.call(f"immune.init({', '.join(argument for argument in arguments if argument)})")
        if isinstance(config, Mapping) and config:
            self.terminal.block("config", config)
        loaded = dict(config) if isinstance(config, Mapping) else config
        runtime = immune.init(mode=mode, config=loaded, sensor=sensor, state_dir=state_dir, backend=backend)
        assert runtime is not None
        return runtime

    def stop(self) -> None:
        if immune.runtime() is not None:
            self.terminal.call("immune.shutdown()")
            immune.shutdown()

    @contextlib.contextmanager
    def running(self, **options: Any) -> Iterator[Runtime]:
        runtime = self.start(**options)
        try:
            yield runtime
        finally:
            self.stop()

    def configure(self, **changes: Any) -> Settings:
        self.terminal.call(f"immune.configure({', '.join(f'{key}=…' for key in changes)})")
        self.terminal.block("changes", changes)
        return immune.configure(**changes)

    def exchange(self, model: Model, send: Callable[[Model], Any]) -> tuple[Any, Verdict | None, str]:
        t = self.terminal
        requests, said = len(model.provider.requests), len(model.said)
        response = send(model)
        if len(model.provider.requests) > requests:
            for role, text in Wire.request(model.provider.requests[-1]):
                t.field(f"→ {role}", text)
        for reply in model.said[said:]:
            spoken = [reply.text] if reply.text else []
            spoken += [f"[tool call] {call.name}({json.dumps(dict(call.arguments))})" for call in reply.tool_calls]
            t.field("← model said", " ".join(spoken), "blue")
        self.live_report()
        verdict = immune.verdict(response)
        self.view.show(verdict)
        delivered = Wire.delivered(response)
        t.field("⇐ app got", delivered, "cyan")
        return response, verdict, delivered

    def expect(self, ok: object, message: str) -> bool:
        self.checks.append((self.current, bool(ok), message))
        self.terminal.check(bool(ok), message)
        return bool(ok)

    def cli(self, *arguments: object, lines: int | None = None) -> str:
        self.terminal.call(f"$ immune {' '.join(str(argument) for argument in arguments)}")
        command = [sys.executable, "-c", "import sys; from immune.cli.main import main; sys.exit(main(sys.argv[1:]))"]
        completed = subprocess.run(
            [*command, *map(str, arguments)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "NO_COLOR": "1", "COLUMNS": "140"},
        )
        rows = (completed.stdout + completed.stderr).rstrip().splitlines()
        self.terminal.output(rows[:lines] if lines else rows)
        if lines and len(rows) > lines:
            self.terminal.output([f"… {len(rows) - lines} more lines"])
        return "\n".join(rows)


def chat(client: Any, user: str, operator: str | None = OPERATOR, **options: Any) -> Any:
    messages = [{"role": "system", "content": operator}] if operator else []
    messages.append({"role": "user", "content": user})
    return client.chat.completions.create(model=MODEL, messages=messages, **options)


def asking(user: str, operator: str | None = OPERATOR) -> Callable[[Model], Any]:
    return lambda model: chat(model.openai(), user, operator=operator)


def tools(*names: str) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": name, "description": name.replace("_", " ")}} for name in names]


def after_tool(operator: str, user: str, tool: str, output: str) -> list[dict[str, Any]]:
    call = {"id": "t0", "type": "function", "function": {"name": tool, "arguments": "{}"}}
    return [
        {"role": "system", "content": operator},
        {"role": "user", "content": user},
        {"role": "assistant", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "t0", "content": output},
    ]


def threats(verdict: Verdict | None) -> set[str]:
    return verdict.threats() if verdict else set()


def required(verdict: Verdict | None) -> Verdict:
    if verdict is None:
        raise AssertionError("expected a verdict, but Immune did not screen the call")
    return verdict


def enforced(verdict: Verdict | None) -> set[str]:
    return {hit.threat for hit in verdict.enforced_hits} if verdict else set()


class Demo:
    registry: ClassVar[list[type[Demo]]] = []
    slug: ClassVar[str]
    chapter: ClassVar[str]
    title: ClassVar[str]
    what: ClassVar[str]
    live: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Demo.registry.append(cls)

    def __init__(self, tour: Tour) -> None:
        self.tour = tour
        self.t = tour.terminal

    def run(self) -> None:
        raise NotImplementedError


class InitDefaults(Demo):
    slug = "init"
    chapter = "1 · Getting started"
    title = "immune.init() and what it switches on"
    what = (
        "One call protects every LLM call in the process. The effective settings (all defaults) are printed below, "
        "then the HTTP interception points and SDK adapters that were installed, and any coverage warnings."
    )

    def run(self) -> None:
        self.t.step("Start Immune with no configuration (a scripted sensor stands in for Jev)")
        runtime = self.tour.start()
        self.t.block(
            "effective settings", runtime.settings.model_dump(mode="json", exclude={"sensor": {"api_key", "api_keys"}})
        )
        coverage = immune.coverage()
        for point in coverage.points:
            self.t.field("intercepts", point)
        for adapter in coverage.adapters:
            self.t.field("adapter", adapter)
        self.t.field("warnings", coverage.warnings or "none")
        self.tour.expect(immune.runtime() is runtime, "immune.runtime() returns the active runtime")
        self.tour.expect(runtime.settings.mode is Mode.AUTO, "the default mode is auto")
        self.tour.expect(any("httpx" in point for point in coverage.points), "httpx and httpx2 clients are intercepted")
        self.tour.stop()
        self.tour.expect(
            immune.runtime() is None and not immune.coverage().points, "immune.shutdown() removes every patch"
        )


class Disabled(Demo):
    slug = "disabled"
    chapter = "1 · Getting started"
    title = "IMMUNE_DISABLED: the environment kill switch"
    what = "IMMUNE_DISABLED=1 turns immune.init() into a no-op, so Immune can be switched off without a code change."

    def run(self) -> None:
        with Env(IMMUNE_DISABLED="1"):
            self.t.call("immune.init()  # with IMMUNE_DISABLED=1")
            result = immune.init(sensor=MockSensor(), state_dir=self.tour.state_dir())
        self.t.field("returned", repr(result))
        self.tour.expect(result is None and immune.runtime() is None, "Immune stays off and patches nothing")


class Gateway:
    def __init__(self, http: ModuleType, reply: str) -> None:
        self._http = http
        self._reply = reply

    def handle_request(self, request: Any) -> Any:
        request.read()
        if request.method == "GET":
            return self._http.Response(200, json={"status": "ok"}, request=request)
        choice = {"index": 0, "message": {"role": "assistant", "content": self._reply}, "finish_reason": "stop"}
        body = {"id": "gw-1", "object": "chat.completion", "created": 0, "model": MODEL, "choices": [choice]}
        return self._http.Response(200, json=body, request=request)

    def close(self) -> None:
        return None


class Endpoints(Demo):
    slug = "endpoints"
    chapter = "1 · Getting started"
    title = "Routing: what gets screened, and the endpoints setting"
    what = (
        "Immune screens only requests it recognizes as LLM calls: known provider hosts, plus standard API paths whose "
        "body looks like an LLM request. Everything else passes untouched. A gateway on a custom path is declared "
        "with endpoints: ['<codec>=<path regex>']."
    )

    def run(self) -> None:
        http = Http.for_sdk(openai)
        body = {"model": MODEL, "messages": [{"role": "user", "content": "What's on the menu?"}]}
        url = "https://llm.internal.example/llm/complete"
        with self.tour.running():
            client = http.Client(transport=Gateway(http, MENU_ANSWER))
            self.t.step("GET https://status.example.com/health (not an LLM call)")
            status = client.get("https://status.example.com/health")
            self.t.field("response", status.json())
            self.tour.expect("x-immune-trace" not in status.headers, "non-LLM traffic passes through untouched")
            self.t.step(f"POST {url} (an LLM gateway on a custom path)")
            response = client.post(url, json=body)
            self.tour.expect("x-immune-trace" not in response.headers, "an unknown path is not screened by default")
        self.t.step("The same call with the gateway declared in endpoints")
        with self.tour.running(config={"endpoints": ["openai_chat=^/llm/complete$"]}):
            client = http.Client(transport=Gateway(http, MENU_ANSWER))
            response = client.post(url, json=body)
            trace = response.headers.get("x-immune-trace")
            self.t.field("x-immune-trace", str(trace))
            self.tour.view.show(immune.verdict(trace))
            self.tour.expect(trace is not None, "the gateway call is now screened")


class VerdictAnatomy(Demo):
    slug = "verdict"
    chapter = "1 · Getting started"
    title = "Reading a verdict, and immune.status()"
    what = (
        "Every screened call produces a Verdict: what Immune did (action), what it would have done if everything "
        "were enforced (would_action), each threat found with its evidence, the sensor used, and versions. "
        "immune.status() summarizes every call site."
    )

    def run(self) -> None:
        with self.tour.running():
            model = Model(MENU_ANSWER)
            with immune.site("ordering"):
                response, verdict, _ = self.tour.exchange(
                    model, lambda model: chat(model.openai(), "How much is the Classic?")
                )
            self.tour.expect(verdict is not None and verdict.action is Action.ALLOW, "an on-task question is allowed")
            self.tour.expect(immune.verdict(response) is verdict, "immune.verdict(response) finds the verdict")
            self.t.step("immune.status()")
            for status in immune.status():
                self.t.block(
                    f"site {status.site}", {name: getattr(status, name) for name in status.__dataclass_fields__}
                )


class Modes(Demo):
    slug = "modes"
    chapter = "2 · Modes and live control"
    title = "mode: auto, observe, strict and off (and IMMUNE_MODE)"
    what = (
        "auto enforces the floor rules from day one and promotes detectors per site once they prove safe. "
        "observe reports what it would have done. strict enforces every threat at its threshold. off passes calls "
        "straight through. A customer asks the burger bot to write code; the scripted sensor reads it as off-task."
    )

    def run(self) -> None:
        outcomes = {"auto": False, "observe": False, "strict": True, "off": False}
        for mode, withheld in outcomes.items():
            self.t.step(f"mode: {mode}")
            with self.tour.running(mode=mode, sensor=MockSensor(OFF_TASK)):
                model = Model(OFF_TOPIC_ANSWER)
                _, verdict, delivered = self.tour.exchange(model, lambda model: chat(model.openai(), OFF_TOPIC))
                self.tour.expect(
                    ("sorted" not in delivered) is withheld,
                    f"{mode}: the code is {'withheld' if withheld else 'delivered'}",
                )
                if mode == "off":
                    self.tour.expect(verdict is None, "off: no verdict, because nothing was screened")
                elif mode != "strict":
                    self.tour.expect(
                        "input.off_task" in threats(verdict), f"{mode}: the off-task request is still recorded"
                    )
        self.t.step("IMMUNE_MODE sets the mode from the environment")
        with Env(IMMUNE_MODE="observe"), self.tour.running() as runtime:
            self.t.field("mode", runtime.settings.mode.value)
            self.tour.expect(runtime.settings.mode is Mode.OBSERVE, "IMMUNE_MODE=observe is applied")


class LiveControl(Demo):
    slug = "configure"
    chapter = "2 · Modes and live control"
    title = "immune.configure(): change behavior without a restart"
    what = (
        "configure() deep-merges changes into the live settings; they apply from the next call. Settings that own "
        "resources (state, sensor, state_dir, canary) stay fixed until the next init()."
    )

    def run(self) -> None:
        with self.tour.running(sensor=MockSensor(OFF_TASK)):
            model = Model(OFF_TOPIC_ANSWER)
            client = model.openai()
            with immune.site("ordering"):
                self.t.step("auto mode: the off-task detector is observed")
                _, _, delivered = self.tour.exchange(model, lambda model: chat(client, OFF_TOPIC))
                self.tour.expect("sorted" in delivered, "delivered while observed")
                self.t.step("Enforce it for this site, live")
                self.tour.configure(sites={"ordering": {"enforce": ["input.off_task"]}})
                _, _, delivered = self.tour.exchange(model, lambda model: chat(client, OFF_TOPIC))
                self.tour.expect("sorted" not in delivered, "enforced from the very next call")
                self.t.step("Drop the whole process to observe mode (for example during a false-positive storm)")
                self.tour.configure(mode="observe")
                _, _, delivered = self.tour.exchange(model, lambda model: chat(client, OFF_TOPIC))
                self.tour.expect("sorted" not in delivered, "a site's enforce list still wins over the mode")
            self.t.step("Try to change a fixed setting")
            try:
                self.tour.configure(state={"backend": "sqlite"})
                self.tour.expect(False, "changing state at runtime is refused")
            except immune.ConfigError as error:
                self.t.field("ConfigError", str(error), "yellow")
                self.tour.expect(True, "changing state at runtime is refused with a clear message")


class Detectors(Demo):
    slug = "detectors"
    chapter = "3 · Detectors and sites"
    title = "How a detector turns Jev's answer into a calibrated probability"
    what = (
        "Immune asks Jev short yes/no questions. A head per threat combines the answers into a calibrated "
        "probability, and the threat fires above its threshold. The scripted sensor answers 'is this request "
        "unrelated to the bot's task?' with rising confidence (strict mode, so every hit is enforced)."
    )

    def run(self) -> None:
        rows = []
        for answer in (0.05, 0.5, 0.8, 0.9, 0.99):
            with self.tour.running(mode="strict", sensor=MockSensor({"off_task": answer})):
                response = chat(Model(OFF_TOPIC_ANSWER).openai(), OFF_TOPIC)
                verdict = immune.verdict(response)
                hit = next((hit for hit in verdict.hits if hit.threat == "input.off_task"), None) if verdict else None
                rows.append(
                    {
                        "Jev: off task?": answer,
                        "head probability": f"{hit.probability:.2f}" if hit else "-",
                        "threshold": SPEC.threats["input.off_task"].threshold,
                        "action": verdict.action.value if verdict else "-",
                    }
                )
        self.t.table(rows)
        self.tour.expect(
            rows[0]["action"] == "allow" and rows[-1]["action"] != "allow", "low answers pass, high answers act"
        )
        self.tour.cli("explain", "input.off_task")


class Sites(Demo):
    slug = "sites"
    chapter = "3 · Detectors and sites"
    title = "Call sites: immune.site(), self-profiling and sites.<site>.archetype"
    what = (
        "Each call site gets its own profile, statistics and settings. Name sites with immune.site(); unnamed "
        "calls get an id derived from their system prompt. sites.<site>.archetype pins what Jev would infer."
    )

    def run(self) -> None:
        config = {"sites": {"kitchen": {"archetype": "internal_tool"}}}
        signals: dict[str, float | str] = {"archetype": "customer_service", "talks_to_end_users": 0.95}
        with self.tour.running(config=config, sensor=MockSensor(signals)):
            model = Model("Sure.")
            for site in ("ordering", "kitchen", None):
                with immune.site(site) if site else contextlib.nullcontext():
                    response = chat(model.openai(), "Hi", operator=f"You are the {site or 'loyalty'} assistant.")
                self.t.field("call", f"immune.site({site!r}) -> verdict.site={required(immune.verdict(response)).site}")
            rows = [
                {"site": s.site, "archetype": s.archetype, "organs": ",".join(s.organs) or "-", "calls": s.calls}
                for s in immune.status()
            ]
            self.t.table(rows)
            kitchen = next(row for row in rows if row["site"] == "kitchen")
            self.tour.expect(kitchen["archetype"] == "internal_tool", "the configured archetype wins over profiling")
            self.tour.expect(
                any(str(row["site"]).startswith("site-") for row in rows), "unnamed calls get a derived id"
            )


class SitePolicy(Demo):
    slug = "site-policy"
    chapter = "3 · Detectors and sites"
    title = "sites.<site>.enforce, observe and enabled"
    what = (
        "enforce and observe override the mode for the listed threats at one site. enabled: false switches Immune "
        "off for a site entirely. The same off-topic request goes to three sites in strict mode."
    )

    def run(self) -> None:
        playground = {"observe": ["input.off_task", "output.task_deviation"]}
        config = {"sites": {"playground": playground, "archive": {"enabled": False}}}
        with self.tour.running(mode="strict", config=config, sensor=MockSensor(OFF_TASK)):
            model = Model(OFF_TOPIC_ANSWER)
            for site, delivered_expected in (("ordering", False), ("playground", True), ("archive", True)):
                self.t.step(f"site {site!r}")
                with immune.site(site):
                    _, verdict, delivered = self.tour.exchange(model, lambda model: chat(model.openai(), OFF_TOPIC))
                self.tour.expect(
                    ("sorted" in delivered) is delivered_expected,
                    f"{site}: {'delivered' if delivered_expected else 'withheld'}",
                )
            self.tour.expect(verdict is None, "a disabled site produces no verdict")


class Organs(Demo):
    slug = "organs"
    chapter = "3 · Detectors and sites"
    title = "Organs and sites.<site>.organs"
    what = (
        "Organs are groups of extra checks that switch on when a site's profile calls for them: agent, coding, "
        "pipeline, business and care. The business organ checks that promises about prices, refunds or discounts "
        "are backed by the operator instructions or documents. sites.<site>.organs switches one on explicitly."
    )

    def run(self) -> None:
        promise = {"makes_commitment": 0.96, "commitment_supported": 0.03}
        for organs in ([], ["business"]):
            self.t.step(f"The bot promises a discount nobody authorized, sites.shop.organs: {organs}")
            with self.tour.running(sensor=MockSensor(promise), config={"sites": {"shop": {"organs": organs}}}):
                model = Model("Deal! You get 50% off everything today, guaranteed.")
                with immune.site("shop"):
                    _, verdict, _ = self.tour.exchange(
                        model, lambda model: chat(model.openai(), "Can I get a discount?")
                    )
                checked = "business.unauthorized_commitment" in threats(verdict)
                self.tour.expect(checked is bool(organs), f"the promise is {'checked' if checked else 'not checked'}")


class Echo(Demo):
    slug = "echo"
    chapter = "3 · Detectors and sites"
    title = "Structured outputs: the independent echo (sites.<site>.echo)"
    what = (
        "For JSON-schema replies, Jev reads each field independently from the input. If the model's value disagrees "
        "(agreement below echo.min_agreement), output.echo_disagreement fires. It is observed by default and "
        "enforced with echo.enforce."
    )

    def run(self) -> None:
        class Triage(BaseModel):
            priority: typing.Literal["low", "medium", "high"]
            refund_requested: bool

        reply = FakeReply('{"priority": "high", "refund_requested": false}')
        setups = [
            ("Jev reads 'high', agreeing with the model", "high", None, False),
            ("Jev reads 'low', disagreeing", "low", None, True),
            ("disagreeing, with echo.enforce: true", "low", {"sites": {"triage": {"echo": {"enforce": True}}}}, True),
            (
                "disagreeing, with echo.min_agreement: 0.02",
                "low",
                {"sites": {"triage": {"echo": {"min_agreement": 0.02}}}},
                False,
            ),
        ]
        for label, read, config, fires in setups:
            self.t.step(label)
            sensor = MockSensor({"echo__priority": read, "echo__refund_requested": 0.02})
            with self.tour.running(config=config, sensor=sensor), immune.site("triage"):
                model = Model(reply)
                response, verdict, _ = self.tour.exchange(
                    model,
                    lambda model: model.openai().responses.parse(
                        model=MODEL,
                        instructions="Triage the support ticket. priority is low, medium or high.",
                        input="The checkout page is down for every customer and we are losing sales!",
                        text_format=Triage,
                    ),
                )
                self.t.field("parsed", repr(response.output_parsed))
                self.tour.expect(
                    ("output.echo_disagreement" in threats(verdict)) is fires,
                    f"echo_disagreement {'fires' if fires else 'stays quiet'}",
                )
                if config and "enforce" in json.dumps(config):
                    self.tour.expect(
                        "output.echo_disagreement" in enforced(verdict), "and it is enforced for this site"
                    )


class Sessions(Demo):
    slug = "sessions"
    chapter = "4 · Sessions and tools"
    title = "immune.session(), anonymous sessions and Responses chains"
    what = (
        "Taint, risk and provenance accumulate per session. Name sessions when you can; otherwise Immune derives "
        "one from the transcript, so two customers who both open with 'Hi' never share state. With the Responses "
        "API, previous_response_id chains stay one session, and an unknown chain starts tainted."
    )

    def run(self) -> None:
        with self.tour.running():
            model = Model("We deliver every day until 10pm.")
            self.t.step("A named session and an anonymous one")
            with immune.session("customer-42"):
                named = chat(model.openai(), "Do you deliver on Sundays?")
            anonymous = chat(model.openai(), "Do you deliver on Sundays?")
            self.t.field("named", str(required(immune.verdict(named)).session_id))
            self.t.field("anonymous", str(required(immune.verdict(anonymous)).session_id))
            self.tour.expect(required(immune.verdict(named)).session_id == "customer-42", "the named session is kept")
            client = model.openai()
            self.t.step("A Responses API chain")
            first, v1, _ = self.tour.exchange(
                model, lambda model: client.responses.create(model=MODEL, input="Hi", instructions="You take orders.")
            )
            _, v2, _ = self.tour.exchange(
                model,
                lambda model: client.responses.create(
                    model=MODEL, input="Two Classics please", previous_response_id=first.id
                ),
            )
            self.tour.expect(v1 and v2 and v1.session_id == v2.session_id, "the chain is one session")
            self.t.step("A chain Immune has never seen")
            _, v3, _ = self.tour.exchange(
                model,
                lambda model: client.responses.create(
                    model=MODEL, input="Continue", previous_response_id="resp_unknown"
                ),
            )
            self.tour.expect(v3 is not None and v3.taint.value == "external", "it starts tainted")


class Confirmations(Demo):
    slug = "confirmations"
    chapter = "4 · Sessions and tools"
    title = "Sealed confirmations for irreversible actions"
    what = (
        "When an irreversible action follows content the user didn't write (here, a tool result), Immune asks the "
        "user to confirm instead of silently dropping it. The reference in the question is sealed to that exact "
        "call and turn."
    )

    def run(self) -> None:
        specs = [
            {"name": name, "description": name.replace("_", " "), "input_schema": {"type": "object"}}
            for name in ("read_records", "delete_record")
        ]
        history: list[dict[str, Any]] = [
            {"role": "user", "content": "Clean up duplicate records"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "read_records", "input": {}}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "Record 7 is a duplicate."}],
            },
        ]
        with self.tour.running(sensor=MockSensor({"talks_to_end_users": 0.95, "archetype": "customer_service"})):
            model = Model(FakeReply(tool_calls=[FakeToolCall("delete_record", {"id": 7}, "c1")]))

            def send(messages: list[dict[str, Any]]) -> Any:
                return model.anthropic().messages.create(
                    model="claude-opus-5",
                    max_tokens=512,
                    system="You are a records assistant.",
                    tools=specs,
                    messages=messages,
                )

            self.t.step("Turn 1: the model wants to delete a record it read about")
            first, _, _ = self.tour.exchange(model, lambda model: send(history))
            question = first.content[0].text
            self.tour.expect("please confirm" in question, "Immune asks the user to confirm")
            self.t.step("Turn 2: the user says yes")
            confirmed = [*history, {"role": "assistant", "content": question}, {"role": "user", "content": "yes"}]
            response, _, _ = self.tour.exchange(model, lambda model: send(confirmed))
            self.tour.expect(
                [block.type for block in response.content] == ["tool_use"], "the confirmed call goes through"
            )


class ToolLoops(Demo):
    slug = "tool-loops"
    chapter = "4 · Sessions and tools"
    title = "limits.max_identical_tool_calls and limits.max_tool_calls_per_turn"
    what = "An agent stuck calling the same tool, or making too many calls in one turn, is stopped (floor F6)."

    def run(self) -> None:
        history: list[dict[str, Any]] = [
            {"role": "system", "content": "You are an ordering assistant."},
            {"role": "user", "content": "Where is my order 42?"},
        ]
        for index in range(3):
            call = {
                "id": f"c{index}",
                "type": "function",
                "function": {"name": "check_order_status", "arguments": '{"id": 42}'},
            }
            history += [
                {"role": "assistant", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": f"c{index}", "content": "in transit"},
            ]
        repeat = FakeReply(tool_calls=[FakeToolCall("check_order_status", {"id": 42})])
        for limit, stops in ((3, True), (5, False)):
            self.t.step(f"A 4th identical check_order_status call with max_identical_tool_calls: {limit}")
            with self.tour.running(config={"limits": {"max_identical_tool_calls": limit}}):
                model = Model(repeat)
                _, verdict, _ = self.tour.exchange(
                    model,
                    lambda model: model.openai().chat.completions.create(
                        model=MODEL, messages=history, tools=tools("check_order_status")
                    ),
                )
                self.tour.expect(
                    ("tool.loop" in enforced(verdict)) is stops, f"limit {limit}: {'stopped' if stops else 'allowed'}"
                )
        self.t.step("Five status checks in one turn with max_tool_calls_per_turn: 3")
        many = FakeReply(tool_calls=[FakeToolCall("check_order_status", {"id": order}) for order in range(5)])
        with self.tour.running(config={"limits": {"max_tool_calls_per_turn": 3}}):
            model = Model(many)
            response, _, _ = self.tour.exchange(
                model,
                lambda model: chat(
                    model.openai(),
                    "Check all my orders",
                    operator="You are an ordering assistant.",
                    tools=tools("check_order_status"),
                ),
            )
            count = len(response.choices[0].message.tool_calls or [])
            self.tour.expect(count == 3, f"3 of the 5 calls were delivered ({count})")


class LatentSensor(MockSensor):
    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        await asyncio.sleep(self.delay_s)
        return await super().read(state, questions)


class Streaming(Demo):
    slug = "streaming"
    chapter = "5 · Streaming and providers"
    title = "streaming and sites.<site>.streaming: progressive or buffered"
    what = (
        "Progressive streaming (default) starts releasing text once the input check passes, holding back only a "
        "short tail until the reply has been checked. Buffered holds the whole reply until the output check "
        "finishes, then releases it at once. The scripted sensor takes 300 ms per Jev request, like a real one."
    )
    _questions = itertools.count(1)

    def stream(self, site: str | None = None) -> list[tuple[float, str]]:
        model = Model(STORY)
        started = time.perf_counter()
        chunks: list[tuple[float, str]] = []
        question = f"Tell me about the burgers ({site or 'any site'}, {next(self._questions)})"
        with immune.site(site) if site else contextlib.nullcontext():
            chunks.extend(
                (time.perf_counter() - started, chunk.choices[0].delta.content)
                for chunk in chat(model.openai(), question, stream=True)
                if chunk.choices and chunk.choices[0].delta.content
            )
        for moment, text in chunks[:6]:
            self.t.field(f"  +{moment * 1000:6.1f} ms", repr(text))
        if len(chunks) > 6:
            self.t.field("  …", f"{len(chunks) - 6} more chunks, the last at +{chunks[-1][0] * 1000:.1f} ms")
        self.t.field("first token", f"{chunks[0][0] * 1000:.0f} ms ({len(chunks)} chunks)")
        return chunks

    def run(self) -> None:
        config = {"sites": {"receipts": {"streaming": "buffered"}}}
        with self.tour.running(config=config, sensor=LatentSensor(0.3)):
            self.t.step("Site 'menu' (progressive, the default)")
            progressive = self.stream("menu")
            self.t.step("Site 'receipts' (buffered)")
            buffered = self.stream("receipts")
            self.tour.expect(
                buffered[0][0] > progressive[0][0] + 0.15, "progressive text arrives well before buffered text"
            )
            self.tour.expect(buffered[-1][0] - buffered[0][0] < 0.05, "buffered text is released all at once")
        self.t.step("streaming: buffered for the whole process")
        with self.tour.running(config={"streaming": "buffered"}, sensor=LatentSensor(0.3)):
            everywhere = self.stream()
            self.tour.expect(everywhere[0][0] > progressive[0][0] + 0.15, "every site now waits for the output check")


class Providers(Demo):
    slug = "providers"
    chapter = "5 · Streaming and providers"
    title = "Every SDK after init: OpenAI, Responses, Anthropic, Gemini, OpenAI-compatible and async"
    what = (
        "immune.init() patches the HTTP layer shared by the OpenAI, Anthropic and Google SDKs (and anything built on "
        "httpx or httpx2), so every client gets a verdict without code changes."
    )

    def run(self) -> None:
        types = importlib.import_module("google.genai.types")
        config = types.GenerateContentConfig(
            system_instruction=OPERATOR,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        with self.tour.running():
            model = Model(MENU_ANSWER)
            calls: list[tuple[str, Callable[[Model], Any]]] = [
                ("OpenAI Chat Completions", lambda model: chat(model.openai(), "What's on the menu?")),
                (
                    "OpenAI Responses",
                    lambda model: model.openai().responses.create(
                        model=MODEL, instructions=OPERATOR, input="What's on the menu?"
                    ),
                ),
                (
                    "Anthropic Messages",
                    lambda model: model.anthropic().messages.create(
                        model="claude-opus-5",
                        max_tokens=512,
                        system=OPERATOR,
                        messages=[{"role": "user", "content": "What's on the menu?"}],
                    ),
                ),
                (
                    "Gemini (google-genai)",
                    lambda model: model.gemini().models.generate_content(
                        model="gemini-3-flash", contents="What's on the menu?", config=config
                    ),
                ),
                (
                    "OpenAI-compatible server (vLLM, Ollama, …)",
                    lambda model: chat(model.openai(base_url="http://localhost:8000/v1"), "What's on the menu?"),
                ),
            ]
            for label, call in calls:
                self.t.step(label)
                _, verdict, _ = self.tour.exchange(model, call)
                self.tour.expect(verdict is not None, f"{label}: screened")
            self.t.step("AsyncOpenAI: 20 concurrent calls")

            async def burst() -> list[Any]:
                client = model.openai(asynchronous=True)
                return await asyncio.gather(*(chat(client, f"What's on the menu? ({index})") for index in range(20)))

            started = time.perf_counter()
            replies = asyncio.run(burst())
            self.t.field("elapsed", f"{(time.perf_counter() - started) * 1000:.0f} ms for {len(replies)} calls")
            self.tour.expect(all(immune.verdict(reply) for reply in replies), "every async call has its own verdict")


class Bedrock(Demo):
    slug = "bedrock"
    chapter = "5 · Streaming and providers"
    title = "Amazon Bedrock through boto3: Converse and ConverseStream"
    what = "Immune hooks botocore's event system for bedrock-runtime. FakeBedrock answers like the real service."

    def run(self) -> None:
        with self.tour.running():
            bedrock = FakeBedrock(FakeReply(MENU_ANSWER))
            client = bedrock.client()
            request = {
                "modelId": "anthropic.claude-opus-5-v1:0",
                "system": [{"text": OPERATOR}],
                "messages": [{"role": "user", "content": [{"text": "What's on the menu?"}]}],
            }
            self.t.step("client.converse(...)")
            response = client.converse(**request)
            for role, text in Wire.request(bedrock.last_request):
                self.t.field(f"→ {role}", text)
            verdict = immune.verdict()
            self.tour.view.show(verdict)
            self.t.field("⇐ app got", Wire.delivered(response), "cyan")
            self.tour.expect(verdict is not None, "Converse is screened")
            self.t.step("client.converse_stream(...)")
            streamed = "".join(
                event["contentBlockDelta"]["delta"].get("text", "")
                for event in client.converse_stream(**request)["stream"]
                if "contentBlockDelta" in event
            )
            self.t.field("⇐ app got", streamed, "cyan")
            self.tour.expect(streamed == MENU_ANSWER, "ConverseStream is screened and delivered intact")


class Protect(Demo):
    slug = "protect"
    chapter = "5 · Streaming and providers"
    title = "immune.protect(client) and immune.protected()"
    what = (
        "protect() wraps one client's transport in place, keeping its proxies, timeouts, limits, certificates, "
        "headers and event hooks, and leaves the rest of the process alone. protected() is a context manager that "
        "protects the whole process inside a block."
    )

    def run(self) -> None:
        http = Http.for_sdk(openai)
        seen: list[str] = []
        model = Model(MENU_ANSWER)
        http_client = http.Client(
            transport=model.provider.transport(http),
            timeout=12.5,
            headers={"x-team": "ordering"},
            event_hooks={"request": [lambda request: seen.append(f"{request.method} {request.url.path}")]},
        )
        self.t.call("client = immune.protect(OpenAI(http_client=http_client), sensor=MockSensor(), state_dir=…)")
        client = immune.protect(
            openai.OpenAI(api_key="sk-demo", http_client=http_client, max_retries=0),
            sensor=MockSensor(),
            state_dir=self.tour.state_dir(),
        )
        _, verdict, _ = self.tour.exchange(model, lambda model: chat(client, "What's on the menu?"))
        self.t.field("event hook saw", seen)
        self.t.field("client kept", f"timeout={http_client.timeout.read}  x-team={http_client.headers.get('x-team')}")
        self.tour.expect(
            verdict is not None and seen and http_client.timeout.read == 12.5,
            "screened, with the client's settings kept",
        )
        self.tour.expect(not immune.coverage().points, "no process-wide patch was installed")
        self.tour.stop()
        self.t.step("with immune.protected(...):")
        self.t.call("with immune.protected(sensor=MockSensor(), state_dir=…):")
        with immune.protected(sensor=MockSensor(), state_dir=self.tour.state_dir()):
            _, verdict, _ = self.tour.exchange(model, lambda model: chat(model.openai(), "What's on the menu?"))
            self.tour.expect(verdict is not None, "inside the block, every client is protected")
        self.tour.expect(immune.runtime() is None, "after the block, Immune is shut down")


class OnBlock(Demo):
    slug = "on-block"
    chapter = "6 · Failure handling and the sensor"
    title = "on_block: respond or raise"
    what = (
        "respond (default) returns a normal SDK response carrying a safe reply, so apps that don't know about Immune "
        "keep working. raise throws immune.Blocked, which subclasses the SDK's own error type and carries the "
        "verdict."
    )

    def run(self) -> None:
        self.t.step("on_block: respond (strict mode, off-topic request)")
        with self.tour.running(mode="strict", sensor=MockSensor(OFF_TASK)):
            model = Model(OFF_TOPIC_ANSWER)
            _, _, delivered = self.tour.exchange(model, lambda model: chat(model.openai(), OFF_TOPIC))
            self.tour.expect(delivered == SPEC.templates.redirect, "the app gets the redirect template")
        self.t.step("on_block: raise")
        with self.tour.running(mode="strict", sensor=MockSensor(OFF_TASK), config={"on_block": "raise"}):
            try:
                chat(Model(OFF_TOPIC_ANSWER).openai(), OFF_TOPIC)
                self.tour.expect(False, "immune.Blocked is raised")
            except immune.Blocked as blocked:
                sdk_error = isinstance(blocked, openai.OpenAIError)
                self.t.field("raised", f"{type(blocked).__name__}; an openai.OpenAIError: {sdk_error}")
                self.tour.view.show(blocked.verdict)
                self.tour.expect(
                    isinstance(blocked, openai.OpenAIError), "immune.Blocked arrives as an SDK error with the verdict"
                )
        self.t.note(
            "Input screening runs while the model call is in flight, so the model was called; its reply was discarded."
        )


class InternalError(Demo):
    slug = "internal-error"
    chapter = "6 · Failure handling and the sensor"
    title = "on_internal_error: pass or block"
    what = (
        "If Immune itself fails, pass (default) lets the call through with a degraded verdict, and block refuses it. "
        "A bug is simulated by making one pipeline step raise."
    )

    def run(self) -> None:
        for choice in ("pass", "block"):
            self.t.step(f"on_internal_error: {choice}")
            with self.tour.running(config={"on_internal_error": choice}) as runtime:
                model = Model(MENU_ANSWER)
                self.t.call("mock.patch.object(runtime.pipeline, 'gate', side_effect=RuntimeError('simulated bug'))")
                with mock.patch.object(runtime.pipeline, "gate", side_effect=RuntimeError("simulated bug")):
                    _, verdict, delivered = self.tour.exchange(
                        model, lambda model: chat(model.openai(), "How much is the Classic?")
                    )
                if choice == "pass":
                    self.tour.expect(
                        MENU_ANSWER in delivered and required(verdict).action is Action.ALLOW, "the call passes through"
                    )
                else:
                    self.tour.expect(
                        required(verdict).action is Action.REFUSE and MENU_ANSWER not in delivered,
                        "the call is refused",
                    )


class SlowSensor(Sensor):
    name: ClassVar[str] = "slow"

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        await asyncio.sleep(3)
        return SensorReading.empty(self.name)


class SensorDeadline(Demo):
    slug = "sensor-deadline"
    chapter = "6 · Failure handling and the sensor"
    title = "sensor.timeout_ms and sensor.deadline_margin_ms"
    what = (
        "Each Jev round trip gets timeout_ms, and screening as a whole gets timeout_ms + deadline_margin_ms. A slow or "
        "hung sensor ends in a degraded verdict (floor candidates act without Jev), never in a hung request."
    )

    def run(self) -> None:
        with self.tour.running(sensor=SlowSensor(), config={"sensor": {"timeout_ms": 200, "deadline_margin_ms": 100}}):
            model = Model(MENU_ANSWER)
            started = time.perf_counter()
            _, verdict, _ = self.tour.exchange(model, lambda model: chat(model.openai(), "What's on the menu?"))
            elapsed = time.perf_counter() - started
            self.t.field("elapsed", f"{elapsed:.2f} s (the sensor would take 3 s)")
            self.tour.expect(
                elapsed < 1.5 and required(verdict).sensor.name == "tier0_only", "finished within budget, without Jev"
            )


class SensorBreaker(Demo):
    slug = "sensor-breaker"
    chapter = "6 · Failure handling and the sensor"
    title = "sensor.breaker_error_rate, breaker_window_s and breaker_cooldown_s"
    what = (
        "When Jev keeps failing, the circuit breaker stops calling it for the cooldown instead of adding latency to "
        "every call."
    )

    def run(self) -> None:
        config = {"sensor": {"breaker_error_rate": 0.5, "breaker_window_s": 60, "breaker_cooldown_s": 5}}
        with self.tour.running(sensor=MockSensor(fail=True), config=config) as runtime:
            model = Model(MENU_ANSWER)
            states = []
            for call in range(4):
                chat(model.openai(), f"What's on the menu? ({call})")
                states.append(runtime.sensor.breaker.is_open)
                self.t.field(
                    f"call {call}", f"breaker open={states[-1]}  sensor={required(immune.verdict()).sensor.name}"
                )
            self.tour.expect(not states[0] and states[-1], "the breaker opens after repeated failures")


class SensorOutage(Demo):
    slug = "sensor-outage"
    chapter = "6 · Failure handling and the sensor"
    title = "sensor.on_outage: act_on_candidates or pass"
    what = (
        "Reflexes only nominate candidates; Jev decides. While Jev is unreachable, act_on_candidates (the default) "
        "acts on floor candidates without Jev, so keys are still redacted and calls still held. pass lets them through "
        "until Jev is back."
    )

    def run(self) -> None:
        results = {}
        for choice in ("act_on_candidates", "pass"):
            with self.tour.running(sensor=MockSensor(fail=True), config={"sensor": {"on_outage": choice}}):
                model = Model(LEAKED_KEY)
                reply = chat(model.openai(), "How do I connect to the printer?").choices[0].message.content
                results[choice] = reply
                self.t.field(choice, reply)
        self.tour.expect("[redacted]" in results["act_on_candidates"], "act_on_candidates redacts without Jev")
        self.tour.expect(results["pass"] == LEAKED_KEY, "pass leaves the reply alone until Jev is back")


class SensorQuota(Demo):
    slug = "sensor-quota"
    chapter = "6 · Failure handling and the sensor"
    title = "sensor.requests_per_minute and sensor.api_keys"
    what = (
        "The quota governor spreads Jev requests across your keys and sheds the least important first (questions "
        "for observed detectors, then enforced ones). Floor questions wait briefly for capacity."
    )

    def run(self) -> None:
        for keys in (["key-a"], ["key-a", "key-b", "key-c"]):
            self.t.step(f"requests_per_minute: 6 with api_keys: {keys}")
            with self.tour.running(config={"sensor": {"requests_per_minute": 6, "api_keys": keys}}) as runtime:
                model = Model("We close at 10pm.")
                for call in range(4):
                    chat(model.openai(), f"What time do you close? ({call})")
                    verdict = required(immune.verdict())
                    self.t.field(f"call {call}", f"sensor={verdict.sensor.name}  jev_calls={verdict.sensor.calls}")
                self.t.field("admitted", dict(runtime.quota.admitted))
                self.t.field("shed", dict(runtime.quota.shed))
                admitted = sum(runtime.quota.admitted.values())
                self.tour.expect(admitted >= len(keys), f"{len(keys)} key(s): {admitted} requests admitted")


class SensorCoalesce(Demo):
    slug = "sensor-coalesce"
    chapter = "6 · Failure handling and the sensor"
    title = "sensor.coalesce"
    what = "coalesce: true merges the input questions and the first document's questions into one Jev request."

    def run(self) -> None:
        messages = after_tool(
            "You are an ordering assistant.",
            "Any news on my delivery?",
            "read_order_notes",
            "Delivery moved to Friday.",
        )
        counts = {}
        for coalesce in (False, True):
            sensor = MockSensor()
            with self.tour.running(sensor=sensor, config={"sensor": {"coalesce": coalesce}}):
                Model("It moved to Friday.").openai().chat.completions.create(model=MODEL, messages=messages)
            counts[coalesce] = len(sensor.calls)
            self.t.field(
                f"coalesce={coalesce}",
                f"{len(sensor.calls)} requests; questions per request {[len(q) for _, q in sensor.calls]}",
            )
        self.tour.expect(counts[True] < counts[False], "coalescing saves a round trip")


class JevWire(Demo):
    slug = "jev-wire"
    chapter = "6 · Failure handling and the sensor"
    title = "The real Jev client, offline: sensor.model, sensor.api_key and cassettes"
    what = (
        "JevWireStub answers in Jev's wire format, so the production JevSensor runs end to end without a network. "
        "RecordingSensor saves readings to a cassette and ReplaySensor plays them back in tests. In production, set "
        "TYPESAFE_API_KEY (or sensor.api_key / sensor.api_keys) and sensor.model."
    )

    def run(self) -> None:
        from immune.sensing.jev import JevSensor

        stub = JevWireStub(model="jev-1.13.0")
        jev = JevSensor("jev-1.13.0", timeout_s=5.0, api_key="ts-demo", transport=stub.transport())
        cassette = self.tour.state_dir("cassette") / "readings.jsonl"
        cassette.parent.mkdir(parents=True)
        self.t.step("JevSensor over the wire stub, recording a cassette")
        with self.tour.running(sensor=RecordingSensor(jev, cassette)):
            model = Model("Yes, we have gluten-free buns.")
            _, recorded, _ = self.tour.exchange(
                model, lambda model: chat(model.openai(), "Do you have gluten-free buns?")
            )
        first = stub.requests[0]
        self.t.field(
            "wire request",
            f"model={first['model']}  questions={len(first['questions'])}  state={sorted(first['state'])}",
        )
        key, question = next(iter(first["questions"].items()))
        self.t.field("a question", f"{key}: {json.dumps(question)}")
        self.t.field("cassette", f"{len(cassette.read_text().splitlines())} readings in {cassette.name}")
        self.t.step("The same call replayed from the cassette")
        with self.tour.running(sensor=ReplaySensor(cassette)):
            model = Model("Yes, we have gluten-free buns.")
            _, replayed, _ = self.tour.exchange(
                model, lambda model: chat(model.openai(), "Do you have gluten-free buns?")
            )
        self.tour.expect(
            recorded is not None and recorded.sensor.model == "jev-1.13.0",
            "the production client spoke Jev's wire format",
        )
        self.tour.expect(
            replayed is not None and replayed.action is required(recorded).action, "the replay reproduces the verdict"
        )


class PrivacyRedaction(Demo):
    slug = "privacy-redaction"
    chapter = "7 · Privacy"
    title = "privacy.redact_before_sensor"
    what = "Personal data is masked before anything is sent to Jev. It is on by default."

    def run(self) -> None:
        message = "Please email my receipt to sam@example.com"
        for redact in (True, False):
            self.t.step(f"redact_before_sensor: {str(redact).lower()}")
            sensor = MockSensor()
            with self.tour.running(sensor=sensor, config={"privacy": {"redact_before_sensor": redact}}):
                chat(Model("Done, it's on its way.").openai(), message)
            seen = next(state for state, _ in sensor.calls if "untrusted_user_message" in state)[
                "untrusted_user_message"
            ]
            self.t.field("Jev saw", seen)
            self.tour.expect(
                ("sam@example.com" in seen) is not redact, f"the address was {'masked' if redact else 'sent as is'}"
            )


class PrivacyLogs(Demo):
    slug = "privacy-log"
    chapter = "7 · Privacy"
    title = "privacy.log: off, redacted or full"
    what = (
        "Controls Immune's per-call log line and the evidence inside verdicts: off logs nothing, redacted (default) "
        "logs ids and threat names with masked evidence, and full keeps raw evidence for debugging."
    )

    def run(self) -> None:
        for choice in ("off", "redacted", "full"):
            self.t.step(f"privacy.log: {choice}")
            with self.tour.running(config={"privacy": {"log": choice}}, sensor=MockSensor(OFF_TASK)):
                count = len(self.tour.logs.records)
                chat(Model(OFF_TOPIC_ANSWER).openai(), OFF_TOPIC)
                lines = [line for line in self.tour.logs.since(count) if "site=" in line]
            self.t.field("verdict log lines", len(lines))
            self.tour.expect(bool(lines) is (choice != "off"), f"{choice}: {'a' if lines else 'no'} per-call log line")


class PrivacyProfiling(Demo):
    slug = "privacy-profiling"
    chapter = "7 · Privacy"
    title = "privacy.profiling: jev or local"
    what = "Profiling a call site sends its system prompt to Jev (jev, default). local profiles it inside your process."

    def run(self) -> None:
        for choice in ("jev", "local"):
            sensor = MockSensor()
            with self.tour.running(sensor=sensor, config={"privacy": {"profiling": choice}}):
                chat(Model("Hi!").openai(), "Hi")
            profiled = any("archetype" in questions for _, questions in sensor.calls)
            self.t.field(f"profiling={choice}", f"Jev asked about the system prompt: {profiled}")
            self.tour.expect(
                profiled is (choice == "jev"), f"{choice}: the system prompt {'was' if profiled else 'was not'} sent"
            )


class PrivacyVerdictLog(Demo):
    slug = "privacy-verdict-log"
    chapter = "7 · Privacy"
    title = "privacy.verdict_log, and state file permissions"
    what = "A durable JSONL verdict log is opt-in. State files are created readable only by their owner."

    def run(self) -> None:
        path = self.tour.state_dir("audit") / "verdicts.jsonl"
        path.parent.mkdir(parents=True)
        with self.tour.running(config={"privacy": {"verdict_log": str(path)}}) as runtime:
            model = Model(MENU_ANSWER)
            chat(model.openai(), "What's on the menu?")
            chat(model.openai(), "Do you deliver?")
            state_dir = runtime.settings.resolved_state_dir()
        lines = path.read_text().splitlines()
        self.t.field("verdict log", f"{len(lines)} lines in {path.name}")
        self.t.field("first line", lines[0])
        self.tour.expect(len(lines) == 2, "one line per verdict")
        if os.name == "posix":
            modes = sorted({oct(item.stat().st_mode & 0o777) for item in state_dir.rglob("*") if item.is_file()})
            self.t.field("file modes", modes)
            self.tour.expect(modes == ["0o600"], "state files are owner-only")


class StateLocal(Demo):
    slug = "state-local"
    chapter = "8 · State"
    title = "state.backend: local, state_dir and IMMUNE_HOME"
    what = "By default state lives in files under ~/.immune. Move it with state_dir or the IMMUNE_HOME variable."

    def run(self) -> None:
        home = self.tour.state_dir("home")
        self.t.call(f"immune.init()  # with IMMUNE_HOME=…/{home.name}")
        with Env(IMMUNE_HOME=str(home)):
            runtime = immune.init(sensor=MockSensor())
            assert runtime is not None
            with immune.site("ordering"):
                chat(Model(MENU_ANSWER).openai(), "What's on the menu?")
            immune.shutdown()
        self.t.field("state dir", str(runtime.settings.resolved_state_dir()))
        files = sorted(item for item in home.rglob("*") if item.is_file())
        for item in files:
            self.t.field("  file", f"{item.relative_to(home)} ({item.stat().st_size} bytes)")
        self.tour.expect(bool(files), "state was written under IMMUNE_HOME")


class StateSqlite(Demo):
    slug = "state-sqlite"
    chapter = "8 · State"
    title = "state.backend: sqlite and state.path"
    what = (
        "Several worker processes on one host can share a SQLite file: sessions, provenance, site profiles and "
        "statistics. Two harnesses stand in for two workers; a third reads the combined site statistics."
    )

    def run(self) -> None:
        shared = {"state": {"backend": "sqlite", "path": str(self.tour.state_dir("shared") / "immune.db")}}
        self.t.block("config (every worker)", shared)
        for worker in ("worker-a", "worker-b"):
            harness = ImmuneHarness(self.tour.state_dir(worker), config=shared)
            with immune.site("ordering"):
                for call in range(3):
                    harness.openai().chat.completions.create(
                        model=MODEL, messages=[{"role": "user", "content": f"Hi {call}"}]
                    )
            self.t.field(worker, "made 3 calls at site 'ordering'")
            harness.close()
        reader = ImmuneHarness(self.tour.state_dir("reader"), config=shared)
        calls = {status.site: status.calls for status in reader.runtime.status()}
        reader.close()
        self.t.field("reader sees", calls)
        self.tour.expect(calls.get("ordering") == 6, "statistics from both workers add up")


class StateRedis(Demo):
    slug = "state-redis"
    chapter = "8 · State"
    title = "state.backend: redis, state.url and state.key_prefix"
    what = "Use Redis for a fleet. fakeredis stands in for a server here; in production set state.url."

    def run(self) -> None:
        if importlib.util.find_spec("fakeredis") is None:
            self.t.note("pip install fakeredis to run this demo")
            return
        import fakeredis

        from immune.state.redis import RedisBackend

        server = fakeredis.FakeRedis(server=fakeredis.FakeServer())
        self.t.call("RedisBackend(redis_client, prefix='bobs:immune:')  # what state.url and state.key_prefix build")
        with self.tour.running(backend=RedisBackend(server, prefix="bobs:immune:")), immune.site("ordering"):
            chat(Model(MENU_ANSWER).openai(), "What's on the menu?")
        keys = sorted(key.decode() if isinstance(key, bytes) else str(key) for key in server.keys("*"))
        for key in keys[:8]:
            self.t.field("  key", key)
        self.tour.expect(
            bool(keys) and all(key.startswith("bobs:immune:") for key in keys), "every key uses the prefix"
        )


class StateFailover(Demo):
    slug = "state-failover"
    chapter = "8 · State"
    title = "state.failover_cooldown_s: when Redis goes away"
    what = (
        "If the shared backend fails, Immune falls back to in-memory state for failover_cooldown_s and keeps screening."
    )

    def run(self) -> None:
        config = {"state": {"backend": "redis", "url": "redis://127.0.0.1:1/0", "failover_cooldown_s": 10}}
        with self.tour.running(config=config):
            model = Model(MENU_ANSWER)
            _, verdict, _ = self.tour.exchange(model, lambda model: chat(model.openai(), "What's on the menu?"))
            self.tour.expect(verdict is not None, "screening continued on in-memory state")


class Promotion(Demo):
    slug = "promotion"
    chapter = "9 · Operations"
    title = "promotion: enabled, min_calls, min_days and max_firing_rate"
    what = (
        "In auto mode a detector starts as observed at each site. Once it has screened min_calls calls over min_days "
        "with a firing-rate upper bound below max_firing_rate, it is promoted and enforced there. The thresholds are "
        "lowered here so it happens in seconds."
    )

    def run(self) -> None:
        for enabled in (True, False):
            config = {"promotion": {"enabled": enabled, "min_calls": 20, "min_days": 0, "max_firing_rate": 0.2}}
            self.t.step(
                f"22 on-topic questions, then one off-topic request (promotion.enabled: {str(enabled).lower()})"
            )
            sensor = MockSensor({"off_task": 0.02})
            with self.tour.running(config=config, sensor=sensor) as runtime, immune.site("ordering"):
                for call in range(22):
                    chat(Model(MENU_ANSWER).openai(), f"What's on the menu? ({call})")
                decision = runtime.ledger.decision("ordering", "input.off_task")
                self.t.field(
                    "decision", f"promoted={runtime.ledger.promoted('ordering', 'input.off_task')}  ({decision.reason})"
                )
                sensor.script(**OFF_TASK)
                model = Model(OFF_TOPIC_ANSWER)
                _, verdict, _ = self.tour.exchange(model, lambda model: chat(model.openai(), OFF_TOPIC))
                self.tour.expect(
                    ("input.off_task" in enforced(verdict)) is enabled,
                    f"the detector is {'enforced' if enabled else 'still observed'}",
                )


class Callbacks(Demo):
    slug = "on-verdict"
    chapter = "9 · Operations"
    title = "immune.on_verdict(): stream verdicts to your SIEM or analytics"
    what = "Callbacks run on a background thread, so a slow or failing callback never delays or breaks a request."

    def run(self) -> None:
        with self.tour.running(sensor=MockSensor(OFF_TASK)):
            collected: list[Verdict] = []

            def broken(verdict: Verdict) -> None:
                raise ValueError("this callback is broken")

            self.t.call("unsubscribe = immune.on_verdict(collected.append)")
            unsubscribe = immune.on_verdict(collected.append)
            immune.on_verdict(broken)
            model = Model(MENU_ANSWER)
            for question in ("What's on the menu?", "Do you deliver?", OFF_TOPIC):
                chat(model.openai(), question)
            time.sleep(0.3)
            unsubscribe()
            rows = [
                {
                    "trace": v.trace_id[:8],
                    "site": v.site,
                    "action": v.action.value,
                    "threats": ",".join(sorted(v.threats())) or "-",
                }
                for v in collected
            ]
            self.t.table(rows)
            self.tour.expect(len(collected) == 3, "every verdict reached the callback, despite the broken one")


class Alerts(Demo):
    slug = "alerts"
    chapter = "9 · Operations"
    title = "telemetry.alerts and immune.on_alert()"
    what = (
        "Immune tracks each site's firing rate per threat and alerts when it jumps far above the baseline (tested with "
        "a Wilson bound), or above absolute_rate when there is no baseline yet. cooldown_s stops repeat alerts."
    )

    def run(self) -> None:
        config = {"telemetry": {"alerts": {"min_fired": 3, "absolute_rate": 0.1, "cooldown_s": 3600, "window_s": 600}}}
        sensor = MockSensor({"off_task": 0.02})
        with self.tour.running(config=config, sensor=sensor):
            raised: list[Any] = []
            immune.on_alert(raised.append)
            with immune.site("ordering"):
                for call in range(6):
                    chat(Model(MENU_ANSWER).openai(), f"What's on the menu? ({call})")
                sensor.script(**OFF_TASK)
                for call in range(8):
                    chat(Model(OFF_TOPIC_ANSWER).openai(), f"{OFF_TOPIC} ({call})")
            time.sleep(0.2)
            for alert in raised:
                self.t.field("alert", alert.message, "yellow")
            alerted = sorted(alert.threat for alert in raised)
            self.tour.expect(
                alerted == ["input.off_task", "output.task_deviation"],
                "one alert per threat, with no repeats during the cooldown",
            )
        with self.tour.running(config={"telemetry": {"alerts": {"enabled": False}}}, sensor=MockSensor(OFF_TASK)):
            raised = []
            immune.on_alert(raised.append)
            for call in range(8):
                chat(Model(OFF_TOPIC_ANSWER).openai(), f"{OFF_TOPIC} ({call})")
            self.tour.expect(not raised, "alerts.enabled: false raises none")


class OpenTelemetry(Demo):
    slug = "opentelemetry"
    chapter = "9 · Operations"
    title = "telemetry.opentelemetry: spans and metrics"
    what = (
        "When your app configures an OpenTelemetry SDK, Immune emits an immune.screen span per call (nested under "
        "the LLM client span when there is one) and metrics for calls, interventions and sensor latency."
    )

    def run(self) -> None:
        try:
            from opentelemetry import metrics, trace
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import InMemoryMetricReader
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        except ImportError:
            self.t.note("pip install opentelemetry-sdk to run this demo")
            return
        spans, reader = InMemorySpanExporter(), InMemoryMetricReader()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(spans))
        trace.set_tracer_provider(provider)
        metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
        for enabled in (True, False):
            spans.clear()
            self.t.step(f"telemetry.opentelemetry: {str(enabled).lower()}")
            with self.tour.running(config={"telemetry": {"opentelemetry": enabled}}, sensor=MockSensor(OFF_TASK)):
                chat(Model(OFF_TOPIC_ANSWER).openai(), OFF_TOPIC)
            finished = spans.get_finished_spans()
            for span in finished:
                self.t.field("span", span.name, "bold")
                for key, value in sorted((span.attributes or {}).items()):
                    self.t.field(f"  {key}", str(value))
            self.tour.expect(bool(finished) is enabled, f"{'a span was' if enabled else 'no span was'} emitted")
        names = sorted(
            {
                metric.name
                for resource in getattr(reader.get_metrics_data(), "resource_metrics", [])
                for scope in resource.scope_metrics
                for metric in scope.metrics
            }
        )
        self.t.field("metrics", names)


class Cli(Demo):
    slug = "cli"
    chapter = "9 · Operations"
    title = "immune.feedback() and the immune command line"
    what = (
        "Label verdicts as correct, false_positive or missed; labels feed per-site calibration. The CLI inspects the "
        "catalog, the state directory and your configuration, and replays scenarios."
    )

    def run(self) -> None:
        with self.tour.running(sensor=MockSensor(OFF_TASK)) as runtime:
            with immune.site("ordering"):
                response = chat(Model(OFF_TOPIC_ANSWER).openai(), OFF_TOPIC)
            verdict = required(immune.verdict(response))
            self.t.call(f"immune.feedback('{verdict.trace_id}', 'correct', note='customers ask for code a lot')")
            immune.feedback(verdict.trace_id, "correct", note="customers ask for code a lot")
            state = runtime.settings.resolved_state_dir()
        generated = self.tour.state_dir("generated") / "immune.yaml"
        generated.parent.mkdir(parents=True)
        self.tour.cli("version")
        self.tour.cli("threats", "--stage", "input")
        self.tour.cli("status", "--state-dir", state)
        self.tour.cli("posture", "--state-dir", state)
        exported = self.tour.cli("export", "labels", "--state-dir", state)
        self.tour.cli("calibrate", "--state-dir", state)
        self.tour.cli("promote", "--state-dir", state)
        self.tour.cli("init", "--state-dir", state, "--output", generated)
        self.tour.cli(
            "test", OFF_TOPIC, "--operator", OPERATOR, "--signal", "off_task=0.95", "--reply", OFF_TOPIC_ANSWER
        )
        self.tour.cli("doctor", "--state-dir", state)
        self.tour.expect(verdict.trace_id in exported, "the label was exported")
        self.tour.expect(generated.exists(), "immune init wrote a starter immune.yaml from the observed sites")


class HeadsArtifact(Demo):
    slug = "heads-artifact"
    chapter = "9 · Operations"
    title = "heads.artifact: detectors trained on your labeled data"
    what = (
        "The evidence engine runs labeled examples through the real pipeline, fits each detector's weights, "
        "calibration and threshold, and writes an artifact pinned to the spec version and Jev model. A toy dataset "
        "stands in: placeholder requests with labels, and scripted answers that are noisy but informative."
    )

    def run(self) -> None:
        from immune.evidence import Example, FeatureCollector, HeadTrainer, ModelCard
        from immune.evidence.collect import FeatureRow

        splits = ("train", "train", "train", "dev", "test")
        examples = [
            Example(
                id=f"ex-{index}",
                source="tour",
                split=splits[index % len(splits)],
                operator=OPERATOR,
                user=f"Sample request {index}",
                reply="OK.",
                labels={"input.off_task": index % 2 == 0},
            )
            for index in range(200)
        ]

        def scripted(example: Example) -> MockSensor:
            noise = random.Random(example.id).uniform(-0.25, 0.25)
            center = 0.8 if example.labels["input.off_task"] else 0.2
            return MockSensor({"off_task": min(0.99, max(0.01, center + noise))})

        self.t.step("Collect features and train")
        features = [row for row in FeatureCollector(scripted).collect(examples) if isinstance(row, FeatureRow)]
        artifact, reports = HeadTrainer(SPEC, "jev-1.13.0").train(features)
        for report in reports:
            rows = f"train={report.rows.get('train', 0)}  test={report.rows.get('test', 0)}"
            self.t.field(report.threat, f"{rows}  metrics={dict(report.metrics)}  threshold={report.threshold}")
        path = self.tour.state_dir("heads") / "heads.json"
        path.parent.mkdir(parents=True)
        artifact.save(path)
        self.t.field("model card", ModelCard.render(artifact, reports))
        self.t.step("Load it with heads.artifact")
        with self.tour.running(config={"heads": {"artifact": str(path)}}) as runtime:
            trained = runtime.spec.heads["input.off_task"].features[0].weight
        provisional = SPEC.heads["input.off_task"].features[0].weight
        self.t.field("weight", f"trained {trained:.2f}, provisional {provisional:.2f}")
        self.tour.expect(trained != provisional, "the trained head is in use")
        self.t.step("The same artifact with a different sensor.model")
        count = len(self.tour.logs.records)
        with self.tour.running(config={"heads": {"artifact": str(path)}, "sensor": {"model": "jev-2.0.0"}}) as runtime:
            fallback = runtime.spec.heads["input.off_task"].features[0].weight
        self.tour.expect(
            fallback == provisional and any("ignoring trained heads" in line for line in self.tour.logs.since(count)),
            "a mismatched artifact is refused with a warning",
        )


LEAKED_KEY = "Use key AKIAABCDEFGHIJKLMNOP to connect to the printer."


class LangSmith(Demo):
    slug = "langsmith"
    chapter = "9 · Operations"
    title = "telemetry.langsmith: possible threats in your LangSmith traces"
    what = (
        "With LANGSMITH_TRACING=true and LANGSMITH_API_KEY set (and pip install 'immune-ai[langsmith]'), every call "
        "with a possible threat becomes a run in LangSmith, nested under your app's own trace, with tags, metadata, "
        "one child run per Jev request, and feedback you can filter on. Clean calls are never sent. Here "
        "immune.testing.LangSmithRecorder stands in for the LangSmith API."
    )

    def run(self) -> None:
        try:
            langsmith = importlib.import_module("langsmith")
        except ImportError:
            self.t.note("pip install 'immune-ai[langsmith]' to run this demo")
            return
        self.t.call("with LangSmithRecorder() as recorder: ...")
        with LangSmithRecorder() as recorder, self.tour.running():
            self.t.step("A clean call: nothing is sent")
            self.tour.exchange(Model(MENU_ANSWER), asking("What's on the menu?"))

            @langsmith.traceable(client=recorder, name="ordering-agent")
            def agent() -> None:
                self.tour.exchange(Model(LEAKED_KEY), asking("How do I connect to the printer?"))

            self.t.step("A call with a threat, inside the app's traced function")
            with langsmith.tracing_context(enabled=True):
                agent()
        runs = recorder.immune_runs()
        parent = next((run for run in recorder.runs if run["name"] == "ordering-agent"), None)
        for run in runs:
            self.t.field("run", run["name"], "bold")
            self.t.field("  tags", run["tags"])
            metadata = run["extra"]["metadata"]
            self.t.field("  metadata", {key: metadata[key] for key in ("site", "action", "spec_version", "jev_model")})
            self.t.field("  inputs", run["inputs"])
            self.t.field("  threats", [threat["id"] for threat in run["outputs"]["threats"]])
            for child in recorder.children(run):
                self.t.field("  child run", f"{child['name']} ({len(child['inputs']['questions'])} questions)")
        for item in recorder.feedback:
            self.t.field("feedback", f"{item['key']} score={item['score']} value={item['value']}")
        self.tour.expect(len(runs) == 1, "only the call with a possible threat was sent")
        self.tour.expect(
            bool(runs) and parent is not None and runs[0]["parent_run_id"] == parent["id"],
            "Immune's run is nested under the app's run",
        )
        self.tour.expect(
            parent is not None and bool(recorder.feedback_on(parent["id"], "immune.blocked")),
            "immune.blocked and immune.threat feedback landed on the app's run",
        )
        self.tour.expect("AKIAABCDEFGHIJKLMNOP" not in json.dumps(recorder.runs, default=str), "inputs were masked")


COMPETITOR = "bobs.no_competitor_mentions"
COMPETITOR_VACCINE: dict[str, Any] = {
    "id": COMPETITOR,
    "version": "1.0.0",
    "title": "The reply names a competitor",
    "stage": "output",
    "detect": {"keywords": ["Burger Palace", "McRival"]},
    "respond": {"message": "I can only talk about Bob's Burgers products."},
    "tests": {"positives": ["You might prefer the Burger Palace deal."], "negatives": [MENU_ANSWER]},
}
RIVAL_ANSWER = "Honestly, Burger Palace has a better deal this week."
VIP_MODULE = """\
from immune.vaccines import VaccineContext


def touches_vip_account(context: VaccineContext) -> str | None:
    account = str(context.arguments.get("account", ""))
    return f"account {account} is a VIP account" if account.startswith("vip-") else None
"""


def write_vaccine(folder: Path, document: Mapping[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{document['id']}.yaml"
    path.write_text(yaml.safe_dump(dict(document), sort_keys=False), encoding="utf-8")
    return path


class Vaccines(Demo):
    slug = "vaccines"
    chapter = "10 · Vaccines"
    title = "Vaccines: teach Immune a new threat in one YAML file"
    what = (
        "A vaccine adds one protection: keywords, a bounded regex, Jev questions, a tool rule or a Python function. "
        "It compiles into the same spec as the built-in threats, so it gets modes, sites, verdicts and telemetry. "
        "New vaccines start observed; enforce them once they have run quietly. They load from vaccines.paths, "
        "immune.init(vaccines=[...]) or the immune.vaccines entry point group."
    )

    def run(self) -> None:
        folder = self.tour.state_dir("vaccines")
        path = write_vaccine(folder, COMPETITOR_VACCINE)
        self.t.step(f"The vaccine ({path.name})")
        self.t.output(path.read_text(encoding="utf-8").rstrip().splitlines())
        self.t.step("Observed first: the reply goes through, and the verdict records the hit")
        with self.tour.running(config={"vaccines": {"paths": [str(folder)]}}) as runtime:
            self.t.field("spec digest", runtime.spec.digest)
            _, verdict, delivered = self.tour.exchange(Model(RIVAL_ANSWER), asking("Any deals this week?"))
            self.tour.expect(COMPETITOR in threats(verdict) and delivered == RIVAL_ANSWER, "observed, not enforced")
            self.tour.expect("vaccines-" in runtime.spec.digest, "the spec digest records the active vaccines")
        write_vaccine(folder, {**COMPETITOR_VACCINE, "enforcement": "enforce"})
        self.t.step("enforcement: enforce, with the vaccine's own message")
        with self.tour.running(config={"vaccines": {"paths": [str(folder)]}}):
            _, verdict, delivered = self.tour.exchange(Model(RIVAL_ANSWER), asking("Any deals this week?"))
            self.tour.expect(delivered == COMPETITOR_VACCINE["respond"]["message"], "the vaccine's message replaced it")
            self.t.step("Switch it off live")
            self.tour.configure(vaccines={"disabled": [COMPETITOR]})
            _, verdict, delivered = self.tour.exchange(Model(RIVAL_ANSWER), asking("Any deals this week?"))
            self.tour.expect(delivered == RIVAL_ANSWER, "vaccines.disabled took effect on the next call")
        config = folder.parent / f"{folder.name}-immune.yaml"
        config.write_text(yaml.safe_dump({"vaccines": {"paths": [str(folder)]}}), encoding="utf-8")
        self.t.step("The vaccine commands")
        self.tour.cli("vaccines", "list", "--custom", "--config", config)
        tested = self.tour.cli("vaccines", "test", folder)
        self.tour.expect("all passed" in tested, "immune vaccines test ran the file's examples")
        trial = self.tour.cli("vaccines", "trial", path, "--max-rate", "0.05")
        self.tour.expect("fired on 0 of" in trial, "the trial found no everyday replies it would fire on")
        made = self.tour.cli(
            "vaccinate",
            "bobs.no_delivery_promises",
            "--stage",
            "output",
            "--regex",
            r"\bdelivered (?:with)?in \d{1,2} minutes\b",
            "--positive",
            "Your order will be delivered in 15 minutes, guaranteed.",
            "--negative",
            "Delivery usually takes 30 to 40 minutes.",
            "--dir",
            folder,
        )
        self.tour.expect(
            (folder / "bobs.no_delivery_promises.yaml").exists(), "immune vaccinate wrote a tested vaccine"
        )
        self.tour.expect("enforcement: observe" in made, "it starts observed")


class VaccineKinds(Demo):
    slug = "vaccine-kinds"
    chapter = "10 · Vaccines"
    title = "Vaccine detectors: Jev questions, tool rules and Python functions"
    what = (
        "Question vaccines join the Jev panel for their stage, so they add tokens but no round trip. Tool rules check "
        "a tool's name and arguments. Python vaccines call your function with the stage, site, text or tool call; "
        "one that raises is skipped with a warning, and one slower than budget_ms is logged. On sites that talk to "
        "users, hold asks the user to confirm; use deny for actions the user must not approve."
    )

    def run(self) -> None:
        folder = self.tour.state_dir("kinds")
        questions = write_vaccine(
            folder,
            {
                "id": "bobs.no_allergy_guarantees",
                "title": "The reply guarantees a dish is allergen-free",
                "stage": "output",
                "detect": {
                    "questions": [
                        {"key": "guarantees", "text": "The assistant output guarantees that food is allergen-free."}
                    ],
                    "threshold": 0.8,
                },
                "applies_to": {"sites": ["ordering"]},
                "enforcement": "enforce",
            },
        )
        refunds = write_vaccine(
            folder,
            {
                "id": "bobs.refund_over_limit",
                "title": "Refunds over $50 need the customer's confirmation",
                "stage": "tool",
                "detect": {"tool": "refund_order", "argument": {"path": "amount", "greater_than": 50}},
                "respond": {"tool": "confirm"},
                "enforcement": "enforce",
            },
        )
        (folder / "bobs_vaccines.py").write_text(VIP_MODULE, encoding="utf-8")
        sys.path.insert(0, str(folder))
        vip = write_vaccine(
            folder,
            {
                "id": "bobs.vip_accounts",
                "title": "Changes to VIP accounts",
                "stage": "tool",
                "detect": {"python": "bobs_vaccines:touches_vip_account"},
                "respond": {"tool": "deny", "message": "VIP accounts are handled by the account team"},
                "enforcement": "enforce",
            },
        )
        try:
            self.t.step("A question vaccine at the ordering site (scripted Jev answer: yes)")
            sensor = MockSensor({"bobs_no_allergy_guarantees__guarantees": 0.95})
            with self.tour.running(config={"vaccines": {"paths": [str(questions)]}}, sensor=sensor):
                with immune.site("ordering"):
                    _, verdict, _ = self.tour.exchange(
                        Model("The Garden Stack is 100% nut-free, guaranteed."), asking("Is it safe for nut allergies?")
                    )
                self.tour.expect("bobs.no_allergy_guarantees" in enforced(verdict), "the question vaccine fired")
                asked = [keys for _, keys in sensor.calls if "bobs_no_allergy_guarantees__guarantees" in keys]
                self.tour.expect(len(asked) == 1, "its question joined the existing output request")
            self.t.step("A tool rule and a Python vaccine")
            calls = [
                FakeToolCall("refund_order", {"order": 7, "amount": 80}),
                FakeToolCall("update_account", {"account": "vip-42"}),
            ]
            model = Model(FakeReply(tool_calls=calls))
            paths = [str(refunds), str(vip)]
            with self.tour.running(
                config={"vaccines": {"paths": paths}}, sensor=MockSensor({"talks_to_end_users": 0.95})
            ):
                _, verdict, delivered = self.tour.exchange(
                    model,
                    lambda m: chat(
                        m.openai(), "Refund order 7 and upgrade vip-42", tools=tools("refund_order", "update_account")
                    ),
                )
                self.tour.expect(
                    {"bobs.refund_over_limit", "bobs.vip_accounts"} <= enforced(verdict),
                    "the refund waits for the customer's confirmation and the VIP change is denied",
                )
                self.tour.expect(
                    "VIP accounts are handled by the account team" in delivered, "with the vaccine's message"
                )
        finally:
            sys.path.remove(str(folder))


class Switches(Demo):
    slug = "switches"
    chapter = "10 · Vaccines"
    title = "Switching protections on and off: globally, per site and live"
    what = (
        "vaccines.disabled and vaccines.enabled take threat ids or globs, for built-in threats and vaccines alike. "
        "Site settings win over global ones. Turning off a floor protection (F1–F10) needs allow_floor_changes: true."
    )

    def run(self) -> None:
        config = {
            "vaccines": {"disabled": ["input.off_task"]},
            "sites": {"kiosk": {"vaccines": {"enabled": ["input.off_task"]}}},
        }
        with self.tour.running(config=config, sensor=MockSensor(OFF_TASK)):
            _, verdict, _ = self.tour.exchange(Model(OFF_TOPIC_ANSWER), asking(OFF_TOPIC))
            self.tour.expect("input.off_task" not in threats(verdict), "off everywhere by default")
            with immune.site("kiosk"):
                _, verdict, _ = self.tour.exchange(Model(OFF_TOPIC_ANSWER), asking(OFF_TOPIC))
            self.tour.expect("input.off_task" in threats(verdict), "back on at the kiosk site")
        self.t.step("The floor is protected")
        try:
            self.tour.start(config={"vaccines": {"disabled": ["output.secret_leak"]}})
        except immune.ConfigError as error:
            self.t.field("error", str(error), "red")
            self.tour.expect("allow_floor_changes" in str(error), "disabling F3 without permission is refused")
        else:
            self.tour.expect(False, "disabling F3 without permission is refused")
        finally:
            self.tour.stop()
        path = self.tour.state_dir("switches") / "immune.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        listed = self.tour.cli("vaccines", "list", "--off", "--site", "kiosk", "--config", path)
        self.tour.expect("input.off_task" not in listed, "immune vaccines list explains each switch per site")
        listed = self.tour.cli("vaccines", "list", "--off", "--config", path)
        self.tour.expect("vaccines.disabled 'input.off_task'" in listed, "and names the setting that decided it")


REFERENCE_CONFIG = """\
# Every Immune setting. Values are the defaults unless a comment says otherwise.
mode: auto                      # auto | observe | strict | off
on_block: respond               # respond (a safe reply) | raise (immune.Blocked)
on_internal_error: pass         # pass (fail open) | block (refuse)
streaming: progressive          # progressive | buffered
canary: true                    # add a reference token to system prompts
state_dir: null                 # default: IMMUNE_HOME or ~/.immune
endpoints:                      # extra LLM routes, <codec>=<path regex> (default: none)
  - openai_chat=^/llm/complete$
sensor:
  model: jev-1.13.0
  timeout_ms: 800
  deadline_margin_ms: 250
  api_key: null                 # prefer the TYPESAFE_API_KEY environment variable
  api_keys: []                  # several keys to spread the load
  requests_per_minute: 1200
  coalesce: false
  breaker_error_rate: 0.2
  breaker_window_s: 60
  breaker_cooldown_s: 30
  on_outage: act_on_candidates  # act_on_candidates | pass: floor candidates while Jev is unreachable
privacy:
  redact_before_sensor: true
  log: redacted                 # off | redacted | full
  profiling: jev                # jev | local
  verdict_log: null             # path of a JSONL audit log
promotion:
  enabled: true
  max_firing_rate: 0.001
  min_calls: 5000
  min_days: 14
limits:
  max_identical_tool_calls: 3
  max_tool_calls_per_turn: 25
  session_ttl_s: 86400
  max_sessions: 50000
state:
  backend: local                # local | sqlite | redis
  path: null                    # the SQLite file
  url: null                     # redis://host:6379/0
  key_prefix: "immune:"
  failover_cooldown_s: 30
telemetry:
  opentelemetry: true
  alerts:
    enabled: true
    window_s: 900
    ratio: 5.0
    min_fired: 5
    absolute_rate: 0.05
    cooldown_s: 3600
  langsmith:
    enabled: auto                # auto (on with LANGSMITH_TRACING + LANGSMITH_API_KEY) | true | false
    project: null                # default: LANGSMITH_PROJECT
    inputs: masked               # masked | none | raw (raw also needs privacy.log: full)
    jev_runs: true               # a child run per Jev request
    feedback: true               # immune.blocked and immune.threat feedback
    mark_blocked_as_error: false
heads:
  artifact: null                # written by `immune evidence fit`
vaccines:
  paths: []                     # vaccine files or directories, e.g. [vaccines/]
  entry_points: true            # also load the immune.vaccines entry point group
  disabled: []                  # threat ids or globs to switch off, e.g. [output.claims_human]
  enabled: []                   # vaccines that declare default: off
  allow_floor_changes: false    # must be true to disable any F1–F10 protection
sites:                          # per call site, named with immune.site("ordering") (default: none)
  ordering:
    enabled: true
    archetype: customer_service
    user_facing: true
    organs: [business]
    enforce: [input.off_task, output.task_deviation]
    observe: []
    streaming: progressive
    allowed_destinations: [bobsburgers.example]
    echo:
      enforce: false
      min_agreement: 0.2
    vaccines:                   # site switches win over the global ones
      disabled: []
      enabled: []
    tools:
      place_order:
        egress: false
        writes_state: true
        executes: false
        reads_private: false
        irreversible: false
        allowed_destinations: []
"""


class ConfigFile(Demo):
    slug = "config-file"
    chapter = "11 · Configuration"
    title = "immune.yaml: every setting, validated and loaded"
    what = (
        "Settings come from immune.yaml in the working directory (or the file named by IMMUNE_CONFIG), the arguments "
        "to immune.init(), and IMMUNE_MODE / IMMUNE_HOME / IMMUNE_DISABLED / TYPESAFE_API_KEY. A JSON Schema gives "
        "editors completion. This reference file sets every option."
    )

    def run(self) -> None:
        path = self.tour.state_dir("config") / "immune.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(REFERENCE_CONFIG, encoding="utf-8")
        self.t.step(f"The reference file ({path.name})")
        self.t.output(REFERENCE_CONFIG.rstrip().splitlines())
        missing = list(self._missing(Settings, yaml.safe_load(REFERENCE_CONFIG)))
        self.t.field("not covered", missing or "none")
        self.tour.expect(not missing, "the reference file sets every configuration option")
        self.tour.cli("config", "validate", path)
        self.t.step("A typo is reported with its path")
        broken = path.with_name("broken.yaml")
        broken.write_text(REFERENCE_CONFIG.replace("privacy:", "privcy:"), encoding="utf-8")
        output = self.tour.cli("config", "validate", broken)
        self.tour.expect("privcy" in output, "validation names the unknown key")
        self.tour.cli("config", "schema", lines=8)
        self.t.step("Load it with immune.init(config=path)")
        with self.tour.running(config=path) as runtime:
            self.t.field("mode", runtime.settings.mode.value)
            self.t.field("sites", list(runtime.settings.sites))
            self.t.field("config hash", runtime.settings.digest)
            self.tour.expect(
                getattr(runtime.settings.site("ordering"), "enforce", ())
                == ("input.off_task", "output.task_deviation"),
                "every value was applied",
            )
        self.t.step("Or point IMMUNE_CONFIG at it")
        with Env(IMMUNE_CONFIG=str(path)), self.tour.running() as runtime:
            self.tour.expect("ordering" in runtime.settings.sites, "IMMUNE_CONFIG was read")

    def _missing(self, model: type[BaseModel], data: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
        for name, field in model.model_fields.items():
            if name not in data:
                yield f"{prefix}{name}"
                continue
            value = data[name]
            nested = self._nested(field.annotation)
            if nested is not None and isinstance(value, Mapping):
                entries = value.values() if nested in (SiteSettings, ToolSettings) else [value]
                for entry in entries:
                    yield from self._missing(nested, entry, f"{prefix}{name}.")

    @staticmethod
    def _nested(annotation: Any) -> type[BaseModel] | None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation
        arguments = typing.get_args(annotation)
        if arguments and isinstance(arguments[-1], type) and issubclass(arguments[-1], BaseModel):
            return arguments[-1]
        return None


LIVE_CHAPTER = "12 · Live Jev (--live)"
TICKET = "The checkout page is down for every customer and we are losing sales!"
MENU_OPERATOR = (
    "You are the ordering assistant for Bob's Burgers. The Classic costs $9 and the Double Stack costs $12. "
    "Never offer discounts."
)


def live_config(**extra: Any) -> dict[str, Any]:
    return {"sensor": {"timeout_ms": 10_000}, **extra}


class LiveConnection(Demo):
    slug = "live-connection"
    chapter = LIVE_CHAPTER
    live = True
    title = "Connecting to Jev: immune doctor --live"
    what = "Checks the key, the SDK and the interception setup, then makes one real Jev call to measure latency."

    def run(self) -> None:
        output = self.tour.cli("doctor", "--live", "--state-dir", self.tour.state_dir())
        row = next((line for line in output.splitlines() if line.startswith("jev")), "")
        self.tour.expect(" yes " in f" {row} ", "the live Jev call succeeded")


class LiveVerdicts(Demo):
    slug = "live-verdicts"
    chapter = LIVE_CHAPTER
    live = True
    title = "Real Jev answers: an on-topic question, an off-topic request, and the reading cache"
    what = (
        "The burger bot gets an on-topic question and an off-topic request. Each Jev exchange is logged: what Jev "
        "saw (already masked), every question it was asked and its answer. Repeating an identical call is then "
        "served from Immune's reading cache without a new Jev request."
    )

    def run(self) -> None:
        with self.tour.running(config=live_config(), sensor=self.tour.jev()), immune.site("ordering"):
            self.t.step("An on-topic question")
            model = Model("The Classic costs $9.")
            _, verdict, _ = self.tour.exchange(model, asking("How much is the Classic?", MENU_OPERATOR))
            self.tour.judge(not threats(verdict), "the on-topic question raised nothing")
            self.t.step("An off-topic request (observed in auto mode)")
            model = Model(OFF_TOPIC_ANSWER)
            _, verdict, _ = self.tour.exchange(model, asking(OFF_TOPIC, MENU_OPERATOR))
            self.tour.judge("input.off_task" in threats(verdict), "Jev recognized the request as off-task")
            self.t.step("Enforce the off-task detectors for this site and repeat the identical request")
            self.tour.configure(sites={"ordering": {"enforce": ["input.off_task", "output.task_deviation"]}})
            before = len(self.tour.ledger.requests)
            _, verdict, delivered = self.tour.exchange(model, asking(OFF_TOPIC, MENU_OPERATOR))
            self.tour.expect(len(self.tour.ledger.requests) == before, "the repeat was answered from the reading cache")
            self.tour.judge("sorted" not in delivered, "the code is withheld once the detectors are enforced")


class LiveProfiling(Demo):
    slug = "live-profiling"
    chapter = LIVE_CHAPTER
    live = True
    title = "Real Jev profiling of three system prompts"
    what = (
        "On the first call at a site, Jev reads the system prompt and Immune builds the site's profile: what kind "
        "of assistant it is, whether it faces the public, and which organs to switch on."
    )

    def run(self) -> None:
        operators = {
            "ordering": OPERATOR,
            "invoices": "Extract the invoice number, total and due date from the document. Reply only with JSON.",
            "coding": "You are a coding assistant working in the user's repository. Explain bugs and propose fixes.",
        }
        expected = {"ordering": "customer_service", "invoices": "extract_or_classify", "coding": "code_assistant"}
        with self.tour.running(config=live_config(), sensor=self.tour.jev()):
            model = Model("OK.")
            for site, operator in operators.items():
                self.t.step(f"First call at site {site!r}")
                with immune.site(site):
                    self.tour.exchange(model, asking("Hello", operator))
            statuses = {status.site: status for status in immune.status()}
            rows = [
                {"site": site, "archetype": status.archetype, "organs": ", ".join(status.organs) or "-"}
                for site, status in statuses.items()
            ]
            self.t.table(rows)
            for site, archetype in expected.items():
                found = statuses[site].archetype if site in statuses else None
                self.tour.judge(found == archetype, f"{site} was profiled as {archetype} ({found})")


class LiveEcho(Demo):
    slug = "live-echo"
    chapter = LIVE_CHAPTER
    live = True
    title = "Real Jev echo: reading a structured field independently"
    what = (
        "For a JSON-schema reply, Jev reads the priority from the ticket itself. When the model's field agrees, "
        "nothing fires; when it disagrees, output.echo_disagreement is reported."
    )

    def run(self) -> None:
        class Triage(BaseModel):
            priority: typing.Literal["low", "medium", "high"]

        with self.tour.running(config=live_config(), sensor=self.tour.jev()), immune.site("triage"):
            for priority, disagrees in (("high", False), ("low", True)):
                self.t.step(f"The model answers priority={priority!r}")
                model = Model(FakeReply(json.dumps({"priority": priority})))
                _, verdict, _ = self.tour.exchange(
                    model,
                    lambda model: model.openai().responses.parse(
                        model=MODEL,
                        instructions="Triage the support ticket. priority is low, medium or high.",
                        input=TICKET,
                        text_format=Triage,
                    ),
                )
                fired = "output.echo_disagreement" in threats(verdict)
                self.tour.judge(
                    fired is disagrees,
                    f"priority={priority!r}: echo_disagreement {'fires' if disagrees else 'stays quiet'}",
                )


class LiveBusiness(Demo):
    slug = "live-business"
    chapter = LIVE_CHAPTER
    live = True
    title = "Real Jev reply checks: a backed price versus an invented discount"
    what = (
        "With the business organ on, Jev checks whether promises in a reply are backed by the operator "
        "instructions. A price from the menu is backed; a discount the instructions forbid is not."
    )

    def run(self) -> None:
        cases = [
            ("a price from the menu", "How much is the Classic?", "The Classic costs $9.", False),
            ("an invented discount", "Any deals today?", "Great news: everything is 50% off today, guaranteed!", True),
        ]
        config = live_config(sites={"ordering": {"organs": ["business"]}})
        with self.tour.running(config=config, sensor=self.tour.jev()), immune.site("ordering"):
            for label, question, answer, flagged in cases:
                self.t.step(f"The reply contains {label}")
                model = Model(answer)
                _, verdict, _ = self.tour.exchange(model, asking(question, MENU_OPERATOR))
                found = "business.unauthorized_commitment" in threats(verdict)
                self.tour.judge(found is flagged, f"{label}: {'flagged' if flagged else 'not flagged'}")


class LivePrivacy(Demo):
    slug = "live-privacy"
    chapter = LIVE_CHAPTER
    live = True
    title = "What leaves your process: masking before Jev"
    what = (
        "Personal data is replaced by typed placeholders before the request is sent. The log shows exactly what "
        "Jev saw."
    )

    def run(self) -> None:
        jev = self.tour.jev()
        with self.tour.running(config=live_config(), sensor=jev), immune.site("ordering"):
            model = Model("Done! Your receipt is on its way.")
            self.tour.exchange(model, lambda model: chat(model.openai(), "Please email my receipt to sam@example.com"))
        sent = json.dumps(jev.seen, default=str)
        self.tour.expect("sam@example.com" not in sent, "the email address never reached Jev")
        self.tour.expect("[EMAIL]" in sent, "Jev saw a typed placeholder instead")


class LiveStreaming(Demo):
    slug = "live-streaming"
    chapter = LIVE_CHAPTER
    live = True
    title = "Real Jev latency and streaming: progressive versus buffered"
    what = (
        "With real Jev latency, progressive streaming releases text once the input check passes, while buffered "
        "waits for the output check too."
    )

    def run(self) -> None:
        first: dict[str, float] = {}
        for mode in ("progressive", "buffered"):
            self.t.step(f"streaming: {mode}")
            with self.tour.running(config=live_config(streaming=mode), sensor=self.tour.jev()):
                started = time.perf_counter()
                moments = [
                    time.perf_counter() - started
                    for chunk in chat(Model(STORY).openai(), f"Tell me about the burgers ({mode})", stream=True)
                    if chunk.choices and chunk.choices[0].delta.content
                ]
                self.tour.live_report()
            first[mode] = moments[0]
            self.t.field(
                "first token", f"{moments[0] * 1000:.0f} ms, last {moments[-1] * 1000:.0f} ms, {len(moments)} chunks"
            )
        self.tour.judge(
            first["buffered"] > first["progressive"] + 0.1, "progressive text arrived well before buffered text"
        )


class LiveSummary(Demo):
    slug = "live-summary"
    chapter = LIVE_CHAPTER
    live = True
    title = "What the live chapter cost: requests, latency and billed tokens"
    what = "Every real Jev request made during this run, summarized."

    def run(self) -> None:
        requests = self.tour.ledger.requests
        if not requests:
            self.tour.expect(False, "live requests were recorded")
            return
        answered = sorted(latency for _, latency, _, completed in requests if completed)
        self.t.table(
            [
                {
                    "requests": len(requests),
                    "answered": len(answered),
                    "timed out": len(requests) - len(answered),
                    "questions": sum(questions for questions, _, _, _ in requests),
                    "latency p50": f"{answered[len(answered) // 2]:.0f} ms" if answered else "-",
                    "latency max": f"{answered[-1]:.0f} ms" if answered else "-",
                    "over 800 ms": sum(latency > 800 for latency in answered),
                    "billed tokens": sum(tokens for _, _, tokens, _ in requests),
                }
            ]
        )
        self.t.note(
            "The production default is sensor.timeout_ms: 800. Requests slower than that continue without Jev "
            "(sensor.on_outage); this chapter uses 10 s so that every answer can be shown."
        )
        self.tour.expect(bool(answered), f"{len(answered)} of {len(requests)} live Jev requests answered")


class TourRunner:
    def __init__(self, tour: Tour, demos: Sequence[type[Demo]]) -> None:
        self.tour = tour
        self.demos = demos

    def run(self) -> int:
        t = self.tour.terminal
        started = time.perf_counter()
        chapter = ""
        for number, demo in enumerate(self.demos, start=1):
            if demo.chapter != chapter:
                chapter = demo.chapter
                t.chapter(chapter)
            self.tour.current = demo.slug
            self.tour.demo_started = len(self.tour.logs.records)
            t.section(number, len(self.demos), demo.slug, demo.title, demo.what)
            began = time.perf_counter()
            try:
                demo(self.tour).run()
            except Exception as error:
                t.write(t.paint("".join(traceback.format_exception(error)).rstrip(), "red"))
                self.tour.expect(False, f"the demo crashed: {type(error).__name__}: {error}")
            finally:
                if immune.runtime() is not None:
                    immune.shutdown()
            t.write(t.paint(f"    ({time.perf_counter() - began:.2f} s)", "grey"))
        return self.summary(time.perf_counter() - started)

    def summary(self, elapsed: float) -> int:
        t = self.tour.terminal
        t.write()
        t.rule("═", "cyan")
        rows = []
        for demo in self.demos:
            results = [ok for slug, ok, _ in self.tour.checks if slug == demo.slug]
            rows.append(
                {
                    "demo": demo.slug,
                    "title": demo.title[:70],
                    "checks": f"{sum(results)}/{len(results)}",
                    "result": "pass" if all(results) else "FAIL",
                }
            )
        t.table(rows)
        failed = [(slug, message) for slug, ok, message in self.tour.checks if not ok]
        passed = len(self.tour.checks) - len(failed)
        t.write()
        verdict = (
            t.paint("ALL CHECKS PASSED", "green", "bold")
            if not failed
            else t.paint(f"{len(failed)} CHECKS FAILED", "red", "bold")
        )
        t.write(f"  {verdict}  {passed}/{len(self.tour.checks)} checks in {len(self.demos)} demos, {elapsed:.1f} s")
        for slug, message in failed:
            t.write(t.paint(f"    ✘ [{slug}] {message}", "red"))
        if self.tour.notes:
            t.write(t.paint(f"  {len(self.tour.notes)} Jev judgements differed from the expected answer:", "yellow"))
            for slug, message in self.tour.notes:
                t.write(t.paint(f"    ⚠ [{slug}] {message}", "yellow"))
        t.rule("═", "cyan")
        return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="A guided tour of Immune's features and settings, with detailed logs.")
    parser.add_argument("--list", action="store_true", help="list the demos and exit")
    parser.add_argument("--only", help="comma-separated demo names to run")
    parser.add_argument("--quiet", action="store_true", help="hide Immune's own log lines")
    parser.add_argument("--no-color", action="store_true", help="plain output")
    parser.add_argument("--keep", action="store_true", help="keep the scratch state directory")
    parser.add_argument("--live", action="store_true", help="also run the live Jev chapter (TYPESAFE_API_KEY or .env)")
    args = parser.parse_args()
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="replace")
    colors = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    terminal = Terminal(Palette(colors))
    if args.list:
        for demo in Demo.registry:
            terminal.write(f"  {demo.slug:<22} {demo.chapter:<36} {demo.title}")
        return 0
    live_key, key_source = LiveKey.find() if args.live else (None, "")
    if args.live and live_key is None:
        parser.error("--live needs TYPESAFE_API_KEY in the environment or in .env")
    if live_key is not None:
        os.environ[LiveKey.NAME] = live_key
    wanted = {name.strip() for name in args.only.split(",")} if args.only else None
    unknown = (wanted or set()) - {demo.slug for demo in Demo.registry} - {"live"}
    if unknown:
        parser.error(f"unknown demo(s): {', '.join(sorted(unknown))}; see --list")
    demos = [
        demo
        for demo in Demo.registry
        if (args.live or not demo.live) and (wanted is None or demo.slug in wanted or ("live" in wanted and demo.live))
    ]
    if not demos:
        parser.error("nothing to run; live demos need --live")
    work = Path(tempfile.mkdtemp(prefix="immune-tour-"))
    logs = ImmuneLogs(terminal, args.quiet)
    logs.install()
    terminal.rule("═", "cyan")
    terminal.write(
        terminal.paint(f"  Immune {immune.__version__}: a tour of every feature and setting", "bold", "cyan")
    )
    terminal.write(f"  openai {openai.__version__}, anthropic {anthropic.__version__}, Python {sys.version.split()[0]}")
    terminal.write(f"  scripted model; scripted Jev answers offline; scratch state in {work}")
    if live_key is not None:
        terminal.write(f"  live Jev chapter enabled: key from {key_source} ({len(live_key)} characters)")
    terminal.rule("═", "cyan")
    try:
        return TourRunner(Tour(terminal, logs, work, live_key), demos).run()
    finally:
        if args.keep:
            terminal.write(f"  state kept in {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
