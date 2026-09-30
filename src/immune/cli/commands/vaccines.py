from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from immune.cli.commands.base import Command
from immune.cli.console import Console
from immune.config.loader import SettingsLoader
from immune.config.settings import Settings, VaccineSettings
from immune.config.spec import Spec
from immune.config.yaml_io import load_yaml
from immune.errors import ConfigError
from immune.sensing.sensor import Sensor
from immune.types import Stage
from immune.vaccines import LoadedVaccine, Switchboard, VaccineBundle, VaccineError, VaccineLoader
from immune.vaccines.lab import (
    Probe,
    ProbeResult,
    SelfSamples,
    SensorFactory,
    TrialReport,
    VaccineLab,
    VaccineReport,
)
from immune.vaccines.model import Vaccine

KINDS = ("keywords", "regex", "questions", "tool", "python")
_STAGES = [stage.value for stage in Stage if stage is not Stage.OPERATOR]
_DEFAULT_ACTION = {
    "input": "text: redirect",
    "data": "data: neutralize",
    "tool": "tool: hold",
    "output": "text: rewrite",
}
_SUBJECT = {"input": "The user message", "data": "The {item}", "tool": "The {call}", "output": "The assistant output"}
_ARGUMENT = re.compile(r"^(?P<path>[\w.]+)\s*(?P<op>>|<|=|~)\s*(?P<value>.+)$")
_OPERATORS = {">": "greater_than", "<": "less_than", "=": "equals", "~": "contains"}


class VaccinesCommand(Command):
    name = "vaccines"
    help = "create, list, test and trial vaccines (custom protections)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        actions = parser.add_subparsers(dest="action", required=True)
        new = actions.add_parser("new", help="scaffold a vaccine file to edit")
        new.add_argument("id", help="namespaced id, such as acme.no_competitor_mentions")
        new.add_argument("--kind", choices=KINDS, default="keywords", help="how it detects (default: keywords)")
        new.add_argument("--stage", choices=_STAGES, help="where it runs (default: output, or tool for tool/python)")
        new.add_argument("--title", help="one-line description (default: from the id)")
        new.add_argument("--dir", type=Path, default=Path("vaccines"), help="directory to write to (default: vaccines)")
        new.add_argument("--force", action="store_true", help="overwrite an existing file")

        listing = actions.add_parser("list", help="every built-in threat and vaccine, and whether it is on")
        listing.add_argument("--site", help="show the state at this site")
        listing.add_argument("--custom", action="store_true", help="only vaccines")
        listing.add_argument("--off", action="store_true", help="only protections that are off")
        _config_argument(listing)

        test = actions.add_parser("test", help="run each vaccine's positive and negative examples")
        test.add_argument("paths", nargs="*", type=Path, help="vaccine files or directories (default: vaccines.paths)")
        _lab_arguments(test)
        _config_argument(test)

        trial = actions.add_parser("trial", help="estimate how often a vaccine fires on everyday traffic")
        trial.add_argument("vaccine", help="a vaccine id from vaccines.paths, or a vaccine file")
        trial.add_argument("--corpus", type=Path, action="append", default=[], help=_CORPUS_HELP)
        trial.add_argument("--no-self", action="store_true", help="skip the packaged everyday samples")
        trial.add_argument("--limit", type=int, default=200, help="screen at most this many samples (default: 200)")
        trial.add_argument("--max-rate", type=float, help="exit 1 when the firing rate is above this, such as 0.01")
        trial.add_argument("--show", type=int, default=10, help="samples that fired to print (default: 10)")
        _lab_arguments(trial)
        _config_argument(trial)

    def run(self, args: argparse.Namespace, console: Console) -> int:
        if args.action == "new":
            return self._new(args, console)
        if args.action == "list":
            return self._list(args, console)
        if args.action == "test":
            return self._test(args, console)
        return self._trial(args, console)

    @staticmethod
    def _new(args: argparse.Namespace, console: Console) -> int:
        stage = args.stage or ("tool" if args.kind in ("tool", "python") else "output")
        text = VaccineTemplate(args.id, args.kind, stage, args.title).render()
        path = args.dir / f"{args.id}.yaml"
        if path.exists() and not args.force:
            raise ConfigError(f"{path} already exists; pass --force to overwrite it")
        _validate(text, path)
        args.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        console.line(f"wrote {path}")
        console.line("next: replace the placeholders, then run")
        console.line(f"  immune vaccines test {path}")
        console.line(f"  immune vaccines trial {path}")
        console.line(f"and add {args.dir}/ to vaccines.paths in immune.yaml (or immune.init(vaccines=[...])).")
        return 0

    @staticmethod
    def _list(args: argparse.Namespace, console: Console) -> int:
        settings = SettingsLoader().load(args.config)
        bundle = VaccineLoader(settings.vaccines).load()
        spec = bundle.compile(Spec.default())
        switchboard = Switchboard(settings.vaccines, bundle.default_off())
        site = settings.site(args.site) if args.site else None
        if args.site and site is None:
            console.line(f"note: no settings for site {args.site!r}; showing the global state")
        rows = []
        for threat in spec.threats.values():
            loaded = bundle.get(threat.id)
            if args.custom and loaded is None:
                continue
            off, why = switchboard.explain(threat.id, site, f"sites.{args.site}")
            if args.off and not off:
                continue
            source = f"vaccine {loaded.vaccine.version}" if loaded is not None else "built-in"
            floor = threat.floor.id if threat.floor is not None else ""
            rows.append((threat.id, source, threat.stage.value, threat.detector, floor, "off" if off else "on", why))
        console.table(("protection", "source", "stage", "detector", "floor", "state", "why"), rows)
        observed = Switchboard.floor_observed(settings, spec)
        if observed:
            console.line("\nfloor protections only observed at a site (sites.<site>.observe):")
            for entry in observed:
                console.line(f"  {entry}")
        removals = Switchboard.floor_removals(settings, spec)
        if removals:
            allowed = "allowed by" if settings.vaccines.allow_floor_changes else "REFUSED at startup without"
            console.line(f"\nfloor protections switched off ({allowed} vaccines.allow_floor_changes):")
            for removal in removals:
                console.line(f"  {removal}")
        console.line(f"\n{len(rows)} protections, {len(bundle.vaccines)} vaccines loaded")
        return 0

    @staticmethod
    def _test(args: argparse.Namespace, console: Console) -> int:
        settings = SettingsLoader().load(args.config)
        if args.paths:
            bundle = VaccineLoader(VaccineSettings(paths=tuple(args.paths), entry_points=False)).load()
        else:
            bundle = VaccineLoader(settings.vaccines).load()
        if not bundle.vaccines:
            console.line("no vaccines found: pass a path or set vaccines.paths in immune.yaml")
            return 1
        lab = VaccineLab(live=_live(settings) if args.live else None, site=args.site)
        reports = [lab.test(loaded) for loaded in bundle.vaccines]
        rows = [row for report in reports for row in _report_rows(report)]
        console.table(("vaccine", "expect", "example", "result", "detail"), rows)
        for report in reports:
            for note in report.notes:
                console.line(f"note: {report.vaccine}: {note}")
        failed = [report.vaccine for report in reports if not report.passed]
        examples = sum(len(report.results) for report in reports)
        if failed:
            console.line(f"\n{len(failed)} of {len(reports)} vaccines failed: {', '.join(failed)}")
            return 1
        console.line(f"\n{len(reports)} vaccines, {examples} examples: all passed")
        return 0

    @staticmethod
    def _trial(args: argparse.Namespace, console: Console) -> int:
        settings = SettingsLoader().load(args.config)
        loaded = _resolve(args.vaccine, settings)
        probes, sources = _samples(loaded.vaccine, args.corpus, include_self=not args.no_self)
        if not probes:
            raise ConfigError("no samples to trial: drop --no-self or pass --corpus")
        probes = probes[: max(args.limit, 1)]
        lab = VaccineLab(live=_live(settings) if args.live else None, site=args.site)
        report = lab.trial(loaded, probes, sources)
        _print_trial(report, console, args.show)
        if args.max_rate is not None and report.rate > args.max_rate:
            console.line(f"firing rate {report.rate:.2%} is above --max-rate {args.max_rate:.2%}")
            return 1
        return 0


class VaccinateCommand(Command):
    name = "vaccinate"
    help = "build a tested vaccine from examples, trial it, and write it observed-first"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("id", help="namespaced id, such as acme.no_competitor_mentions")
        parser.add_argument("--stage", choices=_STAGES, required=True, help="where it runs")
        parser.add_argument("--title", help="one-line description (default: from the id)")
        detect = parser.add_argument_group("detector (choose one kind)")
        detect.add_argument("--keyword", action="append", default=[], help="whole-word, case-insensitive (repeatable)")
        detect.add_argument("--regex", action="append", default=[], help="bounded regular expression (repeatable)")
        detect.add_argument("--question", action="append", default=[], help="yes/no statement for Jev (repeatable)")
        detect.add_argument("--tool", help="tool name or glob, for stage tool")
        detect.add_argument("--argument", help="tool argument rule: path>N, path<N, path=value or path~text")
        detect.add_argument("--python", help="module.path:function")
        detect.add_argument("--threshold", type=float, default=0.8, help="for questions (default: 0.8)")
        parser.add_argument("--positive", action="append", default=[], help=_EXAMPLE_HELP.format("fire"))
        parser.add_argument("--negative", action="append", default=[], help=_EXAMPLE_HELP.format("not fire"))
        parser.add_argument("--message", help="what users see when it is enforced")
        parser.add_argument("--site", dest="sites", action="append", default=[], help="site pattern (repeatable)")
        parser.add_argument(
            "--dir", type=Path, default=Path("vaccines"), help="directory to write to (default: vaccines)"
        )
        parser.add_argument("--corpus", type=Path, action="append", default=[], help=_CORPUS_HELP)
        parser.add_argument("--live", action="store_true", help="judge question vaccines with Jev (uses tokens)")
        parser.add_argument("--force", action="store_true", help="overwrite an existing file")
        _config_argument(parser)

    def run(self, args: argparse.Namespace, console: Console) -> int:
        document = self.document(args)
        target = args.dir / f"{args.id}.yaml"
        if target.exists() and not args.force:
            raise ConfigError(f"{target} already exists; pass --force to overwrite it")
        settings = SettingsLoader().load(args.config)
        header = f"# Created by `immune vaccinate` on {datetime.now(UTC):%Y-%m-%d}. It starts observed.\n"
        text = header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=120)
        lab = VaccineLab(live=_live(settings) if args.live else None)
        with tempfile.TemporaryDirectory(prefix="immune-vaccinate-") as scratch:
            draft = Path(scratch) / target.name
            draft.write_text(text, encoding="utf-8")
            loaded = VaccineLoader(VaccineSettings(entry_points=False)).load_file(draft)
            report = lab.test(loaded)
            console.heading(f"vaccinate {args.id}")
            console.table(("vaccine", "expect", "example", "result", "detail"), _report_rows(report))
            for note in report.notes:
                console.line(f"note: {note}")
            if not report.passed:
                console.line("\nthe vaccine did not pass its own examples, so nothing was written")
                return 1
            trial = self._trial(lab, loaded, args.corpus, console)
        args.dir.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        console.line(f"\nwrote {target} (enforcement: observe)")
        if trial is not None and trial.fired:
            console.line("review the samples that fired above before enforcing it")
        console.line(f"next: add {args.dir}/ to vaccines.paths, watch its observed hits, then set enforcement: enforce")
        return 0

    @staticmethod
    def document(args: argparse.Namespace) -> dict[str, Any]:
        stage = Stage(args.stage)
        document: dict[str, Any] = {
            "id": args.id,
            "version": "1.0.0",
            "title": args.title or _title(args.id),
            "stage": stage.value,
            "detect": _detect(args, stage),
        }
        if args.message:
            document["respond"] = {"message": args.message}
        if args.sites:
            document["applies_to"] = {"sites": list(args.sites)}
        document["enforcement"] = "observe"
        positives = [_example(item, stage) for item in args.positive]
        negatives = [_example(item, stage) for item in args.negative]
        if not positives:
            raise ConfigError("give at least one --positive example that the vaccine must catch")
        document["tests"] = {"positives": positives, "negatives": negatives}
        document["provenance"] = {"created_by": "immune vaccinate"}
        return document

    @staticmethod
    def _trial(lab: VaccineLab, loaded: LoadedVaccine, corpus: Sequence[Path], console: Console) -> TrialReport | None:
        if loaded.vaccine.detect.kind == "questions" and not lab.live:
            console.line("trial skipped: question vaccines need Jev to judge samples (run with --live)")
            return None
        probes, sources = _samples(loaded.vaccine, corpus, include_self=True)
        report = lab.trial(loaded, probes, sources)
        console.line()
        _print_trial(report, console, show=5)
        return report


class VaccineTemplate:
    def __init__(self, vaccine_id: str, kind: str, stage: str, title: str | None) -> None:
        if kind == "tool" and stage != "tool":
            raise ConfigError("a tool rule runs at stage tool; drop --stage or use --stage tool")
        self._id = vaccine_id
        self._kind = kind
        self._stage = stage
        self._title = title or _title(vaccine_id)

    def render(self) -> str:
        detect, tests = getattr(self, f"_{self._kind}")()
        subject = _SUBJECT[self._stage]
        return "\n".join(
            [
                f"# {self._title}",
                "# A vaccine adds one protection to Immune. Guide: docs/guides/vaccines.md",
                f"id: {self._id}",
                "version: 1.0.0",
                f"title: {json.dumps(self._title)}",
                f"stage: {self._stage}".ljust(33) + "# input | data | tool | output",
                detect.strip("\n"),
                "respond:                         # what happens when it is enforced",
                f"  {_DEFAULT_ACTION[self._stage]}".ljust(33) + "# the default here; the guide lists the others",
                '  message: "Sorry, I can\'t help with that here."   # optional: what users see instead',
                "applies_to:",
                '  sites: []                      # e.g. ["support*"]; empty means every site',
                "enforcement: observe             # observe first; enforce once the trial and observed hits look right",
                'default: "on"                    # "off": only where vaccines.enabled lists it',
                tests.strip("\n"),
                "",
            ]
        ).replace("{subject}", subject)

    def _keywords(self) -> tuple[str, str]:
        detect = """
detect:
  keywords: ["replace me"]       # whole words, case-insensitive
"""
        return detect, self._text_tests('"an example that should fire: replace me"')

    def _regex(self) -> tuple[str, str]:
        detect = """
detect:
  regex: ['\\bREF-\\d{4,8}\\b']     # bounded repeats only: {1,50} instead of + or *
"""
        return detect, self._text_tests('"an example that should fire: REF-123456"')

    def _questions(self) -> tuple[str, str]:
        detect = """
detect:
  questions:
    - key: replace_me
      text: "{subject} does the thing this vaccine should catch (replace me)."
  head: {bias: 0.0, weights: {replace_me: 1.0}}   # optional; weights per question
  threshold: 0.8                 # fires when the calibrated probability reaches this
"""
        if self._stage == "tool":
            return detect, self._tool_tests()
        return detect, self._text_tests('"an example that should fire (replace me)"')

    def _tool(self) -> tuple[str, str]:
        detect = """
detect:
  tool: replace_me_tool          # tool name; globs such as "refund_*" work
  argument: {path: amount, greater_than: 100}   # optional: equals, one_of, contains, greater_than, less_than
"""
        return detect, self._tool_tests()

    def _python(self) -> tuple[str, str]:
        function = re.sub(r"[^a-z0-9_]", "_", self._id.split(".")[-1])
        detect = f"""
detect:
  python: your_package.vaccines:{function}   # module.path:function, importable where the app runs
  budget_ms: 50                  # slower calls are logged; exceptions skip the vaccine for that call
# def {function}(context: immune.vaccines.VaccineContext) -> bool | str | list[str] | None:
#     # context: vaccine, stage, site, text, tool, arguments, tainted
#     # return True or evidence text (or a list of it) when the threat is present, else None
#     ...
"""
        if self._stage == "tool":
            return detect, self._tool_tests()
        return detect, self._text_tests('"an example that should fire (replace me)"')

    @staticmethod
    def _text_tests(positive: str) -> str:
        return f"""
tests:                           # run with: immune vaccines test
  positives: [{positive}]
  negatives: ["an everyday example that should not fire"]
"""

    @staticmethod
    def _tool_tests() -> str:
        return """
tests:                           # run with: immune vaccines test
  positives: [{tool: replace_me_tool, arguments: {amount: 250}}]
  negatives: [{tool: replace_me_tool, arguments: {amount: 20}}]
"""


_CORPUS_HELP = "your own samples: JSON Lines ({'text': ...} or {'tool': ..., 'arguments': ...}) or text lines"
_EXAMPLE_HELP = "an example that should {} (repeatable); for tool vaccines: 'tool_name {{\"arg\": 1}}'"


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="immune.yaml to read (default: IMMUNE_CONFIG or ./immune.yaml)")


def _lab_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--live", action="store_true", help="judge question vaccines with Jev (uses tokens)")
    parser.add_argument("--site", help="site name to run as (default: one that matches applies_to.sites)")


def _live(settings: Settings) -> SensorFactory:
    from immune.sensing.jev import JevSensor

    def factory() -> Sensor:
        return JevSensor(settings.sensor.model, settings.sensor.timeout_s * 5, api_key=settings.sensor.api_key)

    return factory


def _validate(text: str, path: Path) -> None:
    try:
        Vaccine.model_validate(load_yaml(text))
    except ValueError as error:
        raise VaccineError(f"{path}: the template does not validate: {error}") from error


def _resolve(reference: str, settings: Settings) -> LoadedVaccine:
    candidate = Path(reference)
    if candidate.is_file():
        return VaccineLoader(VaccineSettings(entry_points=False)).load_file(candidate)
    bundle: VaccineBundle = VaccineLoader(settings.vaccines).load()
    loaded = bundle.get(reference)
    if loaded is None:
        known = ", ".join(bundle.ids) or "none loaded; set vaccines.paths or pass a file"
        raise VaccineError(f"no vaccine {reference!r} (known: {known})")
    return loaded


def _samples(vaccine: Vaccine, corpus: Sequence[Path], include_self: bool) -> tuple[list[Probe], list[str]]:
    samples = SelfSamples()
    probes: list[Probe] = []
    sources: list[str] = []
    if include_self:
        packaged = samples.packaged(vaccine.stage)
        probes += packaged
        sources.append(f"{len(packaged)} everyday {vaccine.stage.value} samples")
    for path in corpus:
        loaded = samples.corpus(path)
        probes += loaded
        sources.append(f"{len(loaded)} from {path}")
    return probes, sources


def _report_rows(report: VaccineReport) -> list[tuple[str, str, str, str, str]]:
    if not report.results:
        return [(report.vaccine, "-", "-", "FAIL", "no tests")]
    return [
        (
            report.vaccine,
            "fire" if result.expected else "quiet",
            result.probe.label,
            "pass" if result.passed else "FAIL",
            _detail(result),
        )
        for result in report.results
    ]


def _detail(result: ProbeResult) -> str:
    if not result.fired:
        return "did not fire"
    probability = f"p={result.probability:.2f}" if result.probability is not None else ""
    evidence = result.evidence[0] if result.evidence else ""
    return "fired " + " ".join(part for part in (probability, evidence) if part)


def _print_trial(report: TrialReport, console: Console, show: int) -> None:
    fired = report.fired
    console.line(f"trial {report.vaccine} ({report.sensor}): {', '.join(report.sources)}")
    console.line(f"fired on {len(fired)} of {report.screened} samples ({report.rate:.1%})")
    if report.input_tokens:
        console.line(f"Jev tokens: {report.input_tokens:,}")
    if fired and show > 0:
        console.table(("sample", "detail"), [(result.probe.label, _detail(result)) for result in fired[:show]])


def _title(vaccine_id: str) -> str:
    words = vaccine_id.rsplit(".", maxsplit=1)[-1].replace("_", " ").replace("-", " ").strip()
    return words[:1].upper() + words[1:]


def _detect(args: argparse.Namespace, stage: Stage) -> dict[str, Any]:
    chosen = [
        name
        for name, present in (
            ("keywords/regex", bool(args.keyword or args.regex)),
            ("question", bool(args.question)),
            ("tool", args.tool is not None),
            ("python", args.python is not None),
        )
        if present
    ]
    if len(chosen) != 1:
        raise ConfigError("choose one detector: --keyword/--regex, --question, --tool or --python")
    if args.argument and args.tool is None:
        raise ConfigError("--argument needs --tool")
    if args.keyword or args.regex:
        detect: dict[str, Any] = {}
        if args.keyword:
            detect["keywords"] = list(args.keyword)
        if args.regex:
            detect["regex"] = list(args.regex)
        return detect
    if args.question:
        questions = [{"key": f"q{index}", "text": text} for index, text in enumerate(args.question, start=1)]
        return {"questions": questions, "threshold": args.threshold}
    if args.python:
        return {"python": args.python}
    if stage is not Stage.TOOL:
        raise ConfigError("--tool needs --stage tool")
    detect = {"tool": args.tool}
    if args.argument:
        detect["argument"] = _argument(args.argument)
    return detect


def _argument(expression: str) -> dict[str, Any]:
    match = _ARGUMENT.match(expression.strip())
    if match is None:
        raise ConfigError(f"--argument {expression!r}: use path>N, path<N, path=value or path~text")
    operator, raw = match["op"], match["value"].strip()
    value: Any = raw
    if operator in "<>":
        try:
            value = float(raw)
        except ValueError as error:
            raise ConfigError(f"--argument {expression!r}: {raw!r} is not a number") from error
    elif operator == "=":
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
    return {"path": match["path"], _OPERATORS[operator]: value}


def _example(item: str, stage: Stage) -> str | dict[str, Any]:
    if stage is not Stage.TOOL:
        return item
    name, _, arguments = item.strip().partition(" ")
    try:
        parsed = json.loads(arguments) if arguments.strip() else {}
    except json.JSONDecodeError as error:
        raise ConfigError(f"tool example {item!r}: arguments must be JSON ({error.msg})") from error
    if not isinstance(parsed, dict):
        raise ConfigError(f"tool example {item!r}: arguments must be a JSON object")
    return {"tool": name, "arguments": parsed}
