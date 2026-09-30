from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import platform
import sys
from collections import Counter
from pathlib import Path

from immune.cli.commands.base import StateCommand
from immune.cli.console import Console
from immune.config.loader import SettingsLoader
from immune.config.settings import Settings
from immune.config.spec import Spec
from immune.errors import ImmuneError
from immune.intercept.transport import HttpLibrary
from immune.sensing.jev import JevSensor
from immune.vaccines import Switchboard, VaccineLoader

_MAX_CUSTOM_QUESTIONS = 10
_SDKS = ("typesafe-sdk", "openai", "anthropic", "google-genai", "httpx2", "httpx", "litellm", "langchain-core")


class DoctorCommand(StateCommand):
    name = "doctor"
    help = "check configuration, dependencies and connectivity"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--live", action="store_true", help="call Jev once to measure latency")

    def run(self, args: argparse.Namespace, console: Console) -> int:
        checks: list[tuple[str, bool, str]] = []
        checks.append(("python", sys.version_info >= (3, 11), platform.python_version()))
        try:
            spec = Spec.default()
            checks.append(("spec", True, f"{spec.version} ({len(spec.threats)} threats)"))
        except ImmuneError as error:
            checks.append(("spec", False, str(error)))
        checks.append(
            (
                "TYPESAFE_API_KEY",
                bool(os.environ.get("TYPESAFE_API_KEY")),
                "set" if os.environ.get("TYPESAFE_API_KEY") else "missing: only floor candidates act, without Jev",
            )
        )
        checks.extend((package, True, self._version(package)) for package in _SDKS)
        libraries = [name for name in ("httpx2", "httpx") if importlib.util.find_spec(name) is not None]
        resolvable = [name for name in libraries if HttpLibrary.named(name).resolves_transports]
        detail = ", ".join(f"{name} (every transport)" if name in resolvable else name for name in libraries)
        checks.append(("interception", bool(resolvable), detail or "no httpx library installed"))
        settings = self.settings(args)
        checks.extend(self._vaccines(settings))
        if settings.state.backend == "local":
            directory = settings.resolved_state_dir()
            checks.append(("state", self._writable(directory), str(directory)))
        else:
            checks.append(self._shared_state(args))
        if args.live:
            checks.append(self._ping())
        console.table(("check", "ok", "detail"), [(name, "yes" if ok else "NO", detail) for name, ok, detail in checks])
        return 0 if all(ok for _, ok, _ in checks) else 1

    @staticmethod
    def _vaccines(settings: Settings) -> list[tuple[str, bool, str]]:
        try:
            bundle = VaccineLoader(settings.vaccines).load()
            spec = bundle.compile(Spec.default())
            Switchboard.validate(settings, spec)
        except ImmuneError as error:
            return [("vaccines", False, str(error))]
        checks = [
            ("vaccines", True, f"{len(bundle.vaccines)} loaded" + (f": {', '.join(bundle.ids)}" if bundle.ids else ""))
        ]
        questions = Counter(item.vaccine.stage.value for item in bundle.vaccines for _ in item.vaccine.detect.questions)
        checks.extend(
            (f"vaccines ({stage})", True, f"warning: {count} custom questions add Jev tokens to every {stage} request")
            for stage, count in sorted(questions.items())
            if count > _MAX_CUSTOM_QUESTIONS
        )
        checks.extend(
            ("floor", True, f"warning: only observed at {entry}")
            for entry in Switchboard.floor_observed(settings, spec)
        )
        return checks

    @staticmethod
    def _version(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    def _shared_state(self, args: argparse.Namespace) -> tuple[str, bool, str]:
        store = self.store(args)
        try:
            store.backend.put("doctor", b"ok", ttl_s=5)
            reachable = store.backend.get("doctor") == b"ok" and not getattr(store.backend, "degraded", False)
        except ImmuneError as error:
            return ("state", False, str(error))
        return ("state", reachable, store.location)

    @staticmethod
    def _writable(directory: Path) -> bool:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / ".doctor"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            return False
        return True

    @staticmethod
    def _ping() -> tuple[str, bool, str]:
        import asyncio

        from immune.config.spec import QuestionSpec

        settings = SettingsLoader().load()
        sensor = JevSensor(settings.sensor.model, timeout_s=10.0)
        question = QuestionSpec(key="ping", kind="noul", text="The text is a greeting.")
        try:
            reading = asyncio.run(sensor.read({"text": "hello"}, [question]))
        except ImmuneError as error:
            return ("jev", False, str(error))
        return ("jev", True, f"{reading.model} in {reading.latency_ms:.0f} ms")
