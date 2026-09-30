from __future__ import annotations

import argparse

from immune.cli.commands.base import Command, missing_sdk
from immune.cli.console import Console
from immune.cli.render import VerdictView
from immune.config.loader import SettingsLoader
from immune.sensing.jev import JevSensor
from immune.sensing.sensor import Sensor
from immune.testing.scenarios import Scenario, ScenarioLibrary, ScenarioResult, ScenarioRunner, SensorFactory


class ReplayCommand(Command):
    name = "replay"
    help = "replay incident, chain and benign scenarios offline"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("scenarios", nargs="*", help="scenario ids or YAML files (default: all)")
        parser.add_argument("--kind", choices=["incident", "chain", "threat", "benign"])
        parser.add_argument("--live", action="store_true", help="use Jev instead of scripted signals")

    def run(self, args: argparse.Namespace, console: Console) -> int:
        library = ScenarioLibrary()
        chosen = [library.find(reference) for reference in args.scenarios] or library.all()
        chosen = [scenario for scenario in chosen if args.kind is None or scenario.kind == args.kind]
        runner = ScenarioRunner(sensor_factory=_live_factory() if args.live else None)
        results: list[ScenarioResult] = []
        skipped: dict[str, list[str]] = {}
        for scenario in chosen:
            try:
                results.append(runner.run(scenario))
            except ModuleNotFoundError as error:
                package = missing_sdk(error)
                if package is None:
                    raise
                skipped.setdefault(package, []).append(scenario.id)
        console.table(
            ("scenario", "result", "details"),
            [(result.scenario.id, "pass" if result.passed else "FAIL", self._details(result)) for result in results],
        )
        failed = sum(not result.passed for result in results)
        console.line(f"\n{len(results) - failed}/{len(results)} scenarios passed")
        for package, ids in skipped.items():
            console.line(
                f"skipped {len(ids)} that need the {package} package (pip install {package}): {', '.join(ids)}"
            )
        if failed:
            return 1
        return 2 if skipped and not results else 0

    @staticmethod
    def _details(result: ScenarioResult) -> str:
        if not result.passed:
            return "; ".join(result.failures)
        verdict = result.turns[-1].verdict
        return verdict.explanation if verdict else ""


class TestCommand(Command):
    name = "test"
    help = "screen one message (and optionally a draft reply) and print the verdict"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("message")
        parser.add_argument("--reply", default="OK.", help="draft assistant reply to screen")
        parser.add_argument("--operator", default="You are a helpful assistant.", help="operator instructions")
        parser.add_argument("--data", action="append", default=[], help="untrusted data item (repeatable)")
        parser.add_argument(
            "--signal",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="scripted sensor answer, e.g. override=0.97 or crisis=suicide_or_self_harm",
        )
        parser.add_argument("--live", action="store_true", help="use Jev instead of scripted signals")

    def run(self, args: argparse.Namespace, console: Console) -> int:
        scenario = Scenario.model_validate(
            {
                "id": "cli.test",
                "title": "immune test",
                "operator": args.operator,
                "signals": dict(self._signal(item) for item in args.signal),
                "turns": [
                    {
                        "user": args.message,
                        "reply": {"text": args.reply},
                        "data": [{"text": text} for text in args.data],
                    }
                ],
            }
        )
        result = ScenarioRunner(sensor_factory=_live_factory() if args.live else None).run(scenario)
        turn = result.turns[0]
        console.line(f"reply: {turn.reply}")
        if turn.verdict is not None:
            VerdictView(turn.verdict).print(console)
        return 0

    @staticmethod
    def _signal(item: str) -> tuple[str, float | str]:
        key, _, value = item.partition("=")
        try:
            return key, float(value)
        except ValueError:
            return key, value


def _live_factory() -> SensorFactory:
    settings = SettingsLoader().load()

    def factory(_: Scenario) -> Sensor:
        return JevSensor(settings.sensor.model, settings.sensor.timeout_s * 5)

    return factory
