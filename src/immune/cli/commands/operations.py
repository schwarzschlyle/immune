from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import yaml

from immune.cli.commands.base import StateCommand
from immune.cli.console import Console
from immune.config.spec import Spec
from immune.errors import ConfigError
from immune.profiling.posture import PostureAssessor
from immune.profiling.registry import Site, SiteRegistry
from immune.reflexes import ReflexSuite
from immune.telemetry.labels import LabelStore
from immune.types import Verdict

Prompt = Callable[[str], str]


class PostureCommand(StateCommand):
    name = "posture"
    help = "list configuration risks for every observed call site"

    def run(self, args: argparse.Namespace, console: Console) -> int:
        assessor = PostureAssessor(ReflexSuite(Spec.default()).secrets)
        rows = [
            (site.name, issue.code, issue.message)
            for site in SiteRegistry(self.store(args)).sites()
            for issue in assessor.assess_site(site)
        ]
        if not rows:
            console.line("no posture issues found")
            return 0
        console.table(("site", "issue", "detail"), rows)
        return 1


class InitCommand(StateCommand):
    name = "init"
    help = "write an immune.yaml that pins what immune inferred about each call site"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--output", type=Path, default=Path("immune.yaml"))
        parser.add_argument("--force", action="store_true", help="overwrite an existing file")

    def run(self, args: argparse.Namespace, console: Console) -> int:
        if args.output.exists() and not args.force:
            raise ConfigError(f"{args.output} already exists; pass --force to overwrite it")
        sites = SiteRegistry(self.store(args)).sites()
        document = {"mode": "auto", "sites": {site.name: self._site(site) for site in sites}}
        args.output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        console.line(f"wrote {args.output} with {len(sites)} call sites; review it and pin what is right")
        return 0

    @staticmethod
    def _site(site: Site) -> dict[str, Any]:
        profile = site.profile
        entry: dict[str, Any] = {"archetype": profile.archetype, "user_facing": profile.user_facing}
        if profile.organs:
            entry["organs"] = sorted(profile.organs)
        tools = {
            name: dict.fromkeys(sorted(capabilities.names), True)
            for name, capabilities in sorted(profile.tools.items())
            if capabilities.names
        }
        if tools:
            entry["tools"] = tools
            entry["allowed_destinations"] = []
        return entry


class VerdictLog:
    def __init__(self, path: Path | None) -> None:
        if path is None:
            raise ConfigError("set privacy.verdict_log in immune.yaml to record verdicts for review and export")
        self._path = path

    def verdicts(self) -> Iterator[Verdict]:
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield Verdict.from_dict(json.loads(line))


class ExportCommand(StateCommand):
    name = "export"
    help = "export labels or recorded verdicts as JSONL or CSV"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("what", choices=["labels", "verdicts"])
        parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
        parser.add_argument("--output", type=Path, help="defaults to standard output")

    def run(self, args: argparse.Namespace, console: Console) -> int:
        records = list(self._records(args))
        rendered = self._jsonl(records) if args.format == "jsonl" else self._csv(records)
        if args.output is None:
            console.write(rendered)
            return 0
        args.output.write_text(rendered, encoding="utf-8")
        console.line(f"wrote {len(records)} {args.what} to {args.output}")
        return 0

    def _records(self, args: argparse.Namespace) -> Iterator[dict[str, Any]]:
        if args.what == "labels":
            yield from self.store(args).read_lines("labels")
            return
        for verdict in VerdictLog(self.settings(args).privacy.verdict_log).verdicts():
            yield verdict.to_dict()

    @staticmethod
    def _jsonl(records: list[dict[str, Any]]) -> str:
        return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)

    @staticmethod
    def _csv(records: list[dict[str, Any]]) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=sorted({key for record in records for key in record}))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in record.items()}
            )
        return buffer.getvalue()


class LabelCommand(StateCommand):
    name = "label"
    help = "review observed hits from the verdict log and record whether they were right"

    def __init__(self, prompt: Prompt = input) -> None:
        self._prompt = prompt

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--limit", type=int, default=50)

    def run(self, args: argparse.Namespace, console: Console) -> int:
        settings = self.settings(args)
        labels = LabelStore(self.store(args))
        labeled = {record.get("trace_id") for record in self.store(args).read_lines("labels")}
        pending = [
            verdict
            for verdict in VerdictLog(settings.privacy.verdict_log).verdicts()
            if verdict.hits and verdict.trace_id not in labeled
        ][: args.limit]
        if not pending:
            console.line("nothing to review")
            return 0
        recorded = 0
        for verdict in pending:
            for hit in verdict.hits:
                console.line(f"\n{verdict.trace_id}  {verdict.site}  {hit.threat}  p={hit.probability:.2f}")
                console.line(f"  {'; '.join(hit.evidence) or verdict.explanation}")
                answer = self._prompt("  [c]orrect  [f]alse positive  [s]kip  [q]uit > ").strip().lower()
                if answer == "q":
                    console.line(f"\nrecorded {recorded} labels")
                    return 0
                label = {"c": "correct", "f": "false_positive"}.get(answer)
                if label is not None:
                    labels.add(verdict, label, threat=hit.threat)
                    recorded += 1
        console.line(f"\nrecorded {recorded} labels")
        return 0
