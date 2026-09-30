from __future__ import annotations

import argparse

from immune.cli.commands.base import Command
from immune.cli.console import Console
from immune.config.spec import Spec
from immune.types import Stage


class VersionCommand(Command):
    name = "version"
    help = "print the installed version"

    def run(self, args: argparse.Namespace, console: Console) -> int:
        from immune import __version__

        spec = Spec.default()
        console.line(f"immune-ai {__version__} (spec {spec.version}, {spec.digest})")
        return 0


class ThreatsCommand(Command):
    name = "threats"
    help = "list the threat catalog"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--stage", choices=[stage.value for stage in Stage])

    def run(self, args: argparse.Namespace, console: Console) -> int:
        spec = Spec.default()
        rows = [
            (
                threat.id,
                threat.stage.value,
                threat.invariant,
                threat.detector,
                threat.floor.id if threat.floor else "",
                threat.organ or "",
                ",".join(threat.frameworks),
            )
            for threat in spec.threats.values()
            if args.stage is None or threat.stage.value == args.stage
        ]
        console.table(("threat", "stage", "invariant", "detector", "floor", "organ", "frameworks"), rows)
        return 0


class ExplainCommand(Command):
    name = "explain"
    help = "show how a threat is detected and handled"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("threat")

    def run(self, args: argparse.Namespace, console: Console) -> int:
        spec = Spec.default()
        threat = spec.threat(args.threat)
        invariant = spec.invariants[threat.invariant]
        console.heading(threat.id)
        console.line(f"invariant   {threat.invariant} {invariant.name}: {invariant.statement}")
        console.line(f"stage       {threat.stage.value}   detector {threat.detector}   severity {threat.severity}")
        console.line(f"threshold   {threat.threshold}   floor {threat.floor.id if threat.floor else 'no (observed)'}")
        console.line(
            f"actions     {', '.join(f'{sink.value}->{action.value}' for sink, action in threat.actions.items())}"
        )
        console.line(f"frameworks  {', '.join(threat.frameworks) or '-'}")
        if threat.detector == "candidate":
            console.line("decided     by Jev: a reflex nominates each candidate; Jev answers the question below")
        head = spec.heads.get(threat.id)
        if head is None:
            console.line("head        deterministic reflex")
            return 0
        questions = {question.key: question for panel in spec.panels.values() for question in panel.questions}
        console.line("head        logistic over atomic questions:")
        for feature in head.features:
            if feature.finding:
                console.line(f"  +{feature.weight:.1f} if reflex {feature.finding} fired")
                continue
            assert feature.signal is not None
            question = questions.get(feature.signal)
            detail = f" [{', '.join(feature.options)}]" if feature.options else ""
            console.line(f"  {feature.weight:+.1f} x {feature.transform}({feature.signal}{detail}, {feature.reduce})")
            if question is not None:
                console.line(f"      {question.kind}: {question.text}")
        return 0
