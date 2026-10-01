from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from immune.config.settings import VaccineSettings
from immune.testing import FakeReply, ImmuneHarness
from immune.types import Stage
from immune.vaccines import LibraryCatalog, VaccineLoader

_VOCABULARY = (
    "order delivery burger fries menu table booking weekend receipt refund store opening hours pickup address "
    "invoice account support ticket shipping package tracking update schedule meeting summary report quarter "
    "customer product feature question answer kitchen recipe garden weather travel museum library"
)
_WORDS = _VOCABULARY.split()
_OPERATOR = "You are a helpful assistant for a local business. Answer briefly."
_REPLY = " ".join(_WORDS[index % len(_WORDS)] for index in range(60)) + "."
_STREAMED_REPLY = " ".join(_WORDS[index % len(_WORDS)] for index in range(400)) + "."
_PAYLOADS = {"1kb": (1_024, 120), "32kb": (32_768, 40), "256kb": (262_144, 10)}
_CONCURRENCY = (1, 16, 64)
_CONCURRENT_CALLS = 256
_WARMUP = 5
_UNMETERED = {"sensor": {"requests_per_minute": None}}
_VACCINES = 50
_LIBRARY = 200


class Text:
    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)

    def of_size(self, size: int) -> str:
        words: list[str] = []
        length = 0
        while length < size:
            word = self._random.choice(_WORDS)
            words.append(word)
            length += len(word) + 1
        return " ".join(words)[:size]


@dataclass(slots=True)
class Timings:
    samples: list[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        self.samples.append(seconds * 1000)

    def percentile(self, fraction: float) -> float:
        ordered = sorted(self.samples)
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    @property
    def median(self) -> float:
        return statistics.median(self.samples)


class Target:
    def __init__(self, mode: str, reply: str, config: dict[str, Any] | None = None) -> None:
        self._scratch = tempfile.TemporaryDirectory()
        settings = {**_UNMETERED, **(config or {})}
        self.harness = ImmuneHarness(Path(self._scratch.name), script=FakeReply(reply), mode=mode, config=settings)
        self.client = self.harness.openai()

    def ask(self, user: str) -> None:
        self.client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "system", "content": _OPERATOR}, {"role": "user", "content": user}],
        )

    def first_token(self, user: str) -> float:
        started = time.perf_counter()
        stream = self.client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "system", "content": _OPERATOR}, {"role": "user", "content": user}],
            stream=True,
        )
        first = 0.0
        for chunk in stream:
            if not first and chunk.choices and chunk.choices[0].delta.content:
                first = time.perf_counter() - started
        return first

    def close(self) -> None:
        self.harness.close()
        self._scratch.cleanup()


@contextmanager
def targets(reply: str) -> Iterator[tuple[Target, Target]]:
    protected, bare = Target("auto", reply), Target("off", reply)
    try:
        yield protected, bare
    finally:
        protected.close()
        bare.close()


class Benchmark:
    def __init__(self, seed: int, scale: float) -> None:
        self._text = Text(seed)
        self._scale = scale

    def run(self) -> dict[str, float]:
        results: dict[str, float] = {}
        with targets(_REPLY) as (protected, bare):
            for name, (size, rounds) in _PAYLOADS.items():
                results.update(self._payload(name, size, self._rounds(rounds), protected, bare))
            for threads in _CONCURRENCY:
                results.update(self._concurrency(threads, protected, bare))
        with targets(_STREAMED_REPLY) as (protected, bare):
            results.update(self._streaming(self._rounds(40), protected, bare))
        results.update(self._vaccines(self._rounds(200)))
        results.update(self._library(self._rounds(120)))
        return {name: round(value, 3) for name, value in results.items()}

    def _rounds(self, rounds: int) -> int:
        return max(3, int(rounds * self._scale))

    def _payload(self, name: str, size: int, rounds: int, protected: Target, bare: Target) -> dict[str, float]:
        timings = {"protected": Timings(), "bare": Timings()}
        for index in range(_WARMUP + rounds):
            user = self._text.of_size(size)
            for label, target in (("protected", protected), ("bare", bare)):
                started = time.perf_counter()
                target.ask(user)
                if index >= _WARMUP:
                    timings[label].add(time.perf_counter() - started)
        return {
            f"payload.{name}.overhead_ms.p50": timings["protected"].median - timings["bare"].median,
            f"payload.{name}.protected_ms.p95": timings["protected"].percentile(0.95),
        }

    def _vaccines(self, rounds: int) -> dict[str, float]:
        with tempfile.TemporaryDirectory(prefix="immune-bench-vaccines-") as scratch:
            folder = Path(scratch)
            for index in range(_VACCINES):
                detect = (
                    {"keywords": [f"product{index}x", f"brand{index}y"]}
                    if index % 2
                    else {"regex": [rf"\bCODE{index}-\d{{4,8}}\b"]}
                )
                document = {"id": f"bench.rule_{index}", "title": f"Rule {index}", "stage": "output", "detect": detect}
                (folder / f"rule_{index}.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
            settings = VaccineSettings(paths=(folder,), entry_points=False, library=False)
            reflexes = VaccineLoader(settings).load().reflexes()
        timings = Timings()
        for index in range(_WARMUP + rounds):
            text = self._text.of_size(1_024)
            started = time.perf_counter()
            reflexes.text_findings(Stage.OUTPUT, text, "bench", frozenset())
            if index >= _WARMUP:
                timings.add(time.perf_counter() - started)
        return {f"vaccines.{_VACCINES}.text_ms.p50": timings.median}

    def _library(self, rounds: int) -> dict[str, float]:
        """What a large vaccine library costs while every vaccine in it is switched off: it should be almost nothing."""
        with tempfile.TemporaryDirectory(prefix="immune-bench-library-") as scratch:
            bundle = _library_bundle(Path(scratch), _LIBRARY)
            configs: dict[str, dict[str, Any]] = {
                "library": {"vaccines": {"library": str(bundle), "entry_points": False}},
                "none": {"vaccines": {"library": False, "entry_points": False}},
            }
            starts = {label: Timings() for label in configs}
            for _ in range(max(3, rounds // 10)):
                for label, config in configs.items():
                    started = time.perf_counter()
                    Target("auto", _REPLY, config).close()
                    starts[label].add(time.perf_counter() - started)
            calls = {label: Timings() for label in configs}
            library, none = Target("auto", _REPLY, configs["library"]), Target("auto", _REPLY, configs["none"])
            try:
                for index in range(_WARMUP + rounds):
                    user = self._text.of_size(1_024)
                    for label, target in (("library", library), ("none", none)):
                        started = time.perf_counter()
                        target.ask(user)
                        if index >= _WARMUP:
                            calls[label].add(time.perf_counter() - started)
            finally:
                library.close()
                none.close()
        return {
            f"library.{_LIBRARY}.init_overhead_ms": starts["library"].median - starts["none"].median,
            f"library.{_LIBRARY}.off.overhead_ms.p50": calls["library"].median - calls["none"].median,
        }

    def _concurrency(self, threads: int, protected: Target, bare: Target) -> dict[str, float]:
        users = [self._text.of_size(1_024) for _ in range(_CONCURRENT_CALLS)]
        rates = {
            label: self._throughput(target.ask, users, threads)
            for label, target in (("protected", protected), ("bare", bare))
        }
        return {
            f"concurrency.{threads}.calls_per_s": rates["protected"],
            f"concurrency.{threads}.throughput_ratio": rates["protected"] / rates["bare"],
        }

    @staticmethod
    def _throughput(ask: Callable[[str], None], users: list[str], threads: int) -> float:
        started = time.perf_counter()
        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(ask, users))
        return len(users) / (time.perf_counter() - started)

    def _streaming(self, rounds: int, protected: Target, bare: Target) -> dict[str, float]:
        timings = {"protected": Timings(), "bare": Timings()}
        for index in range(_WARMUP + rounds):
            user = self._text.of_size(512)
            for label, target in (("protected", protected), ("bare", bare)):
                seconds = target.first_token(user)
                if index >= _WARMUP:
                    timings[label].add(seconds)
        return {
            "streaming.ttft_ms.p50": timings["protected"].median,
            "streaming.ttft_overhead_ms.p50": timings["protected"].median - timings["bare"].median,
        }


def _library_bundle(folder: Path, count: int) -> Path:
    """A synthetic library of question, keyword and regex vaccines across stages, all switched off."""
    for index in range(count):
        kind = index % 3
        stage = ("output", "input", "data")[index % 3]
        if kind == 0:
            subject = {"output": "The assistant output", "input": "The user message", "data": "The {item}"}[stage]
            detect: dict[str, Any] = {"questions": [{"key": "present", "text": f"{subject} is about topic {index}."}]}
        elif kind == 1:
            detect = {"keywords": [f"product{index}x", f"brand{index}y"]}
        else:
            detect = {"regex": [rf"\bCODE{index}-\d{{4,8}}\b"]}
        document = {
            "id": f"immune.bench.rule_{index}",
            "title": f"Rule {index}",
            "stage": stage,
            "maturity": "experimental",
            "default": "off",
            "detect": detect,
            "provenance": {"owner": "@bench"},
        }
        (folder / f"rule_{index}.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    bundle = folder / "bundle.json"
    bundle.write_text(json.dumps(LibraryCatalog.from_sources(folder).bundle(folder, {})), encoding="utf-8")
    return bundle


@dataclass(frozen=True, slots=True)
class Budget:
    name: str
    limit: float
    kind: str

    def violated(self, measured: float, tolerance: float) -> bool:
        if self.kind == "max":
            return measured > self.limit * (1 + tolerance)
        return measured < self.limit * (1 - tolerance)


class Budgets:
    def __init__(self, path: Path) -> None:
        self._path = path
        data = json.loads(path.read_text(encoding="utf-8"))
        self.tolerance = float(data["tolerance"])
        self.entries = [
            Budget(name, float(limit), kind) for kind in ("max", "min") for name, limit in data.get(kind, {}).items()
        ]

    def check(self, measured: dict[str, float]) -> list[str]:
        problems = []
        for budget in self.entries:
            value = measured.get(budget.name)
            if value is None:
                problems.append(f"{budget.name}: not measured")
            elif budget.violated(value, self.tolerance):
                sign = "<=" if budget.kind == "max" else ">="
                limit = f"{sign} {budget.limit:.3f}, {self.tolerance:.0%} slack"
                problems.append(f"{budget.name}: {value:.3f} (budget {limit})")
        return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="measure immune's overhead against an unprotected client")
    parser.add_argument("--check", type=Path, help="budgets file; exit 1 when a budget is exceeded")
    parser.add_argument("--out", type=Path, help="write the measurements as JSON")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scale", type=float, default=1.0, help="multiply the number of rounds")
    args = parser.parse_args()
    measured = Benchmark(args.seed, args.scale).run()
    width = max(len(name) for name in measured)
    for name, value in measured.items():
        print(f"{name.ljust(width)}  {value:10.3f}")
    if args.out is not None:
        report: dict[str, Any] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "results": measured,
        }
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.check is None:
        return 0
    problems = Budgets(args.check).check(measured)
    for problem in problems:
        print(f"over budget: {problem}", file=sys.stderr)
    print("all budgets met" if not problems else f"{len(problems)} budgets exceeded")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
