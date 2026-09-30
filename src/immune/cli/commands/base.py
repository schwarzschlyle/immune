from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from immune.cli.console import Console
from immune.config.loader import SettingsLoader
from immune.config.settings import Settings
from immune.state import StateBackends
from immune.telemetry.state import StateStore

# Provider SDKs that commands drive (replay and test send scenarios through them), and the package to install for each.
_SDKS = {"openai": "openai", "anthropic": "anthropic", "google": "google-genai", "boto3": "boto3", "botocore": "boto3"}


def missing_sdk(error: ModuleNotFoundError) -> str | None:
    """The package to install when `error` is a missing provider SDK, otherwise None."""
    return _SDKS.get((error.name or "").partition(".")[0])


class Command(ABC):
    name: ClassVar[str]
    help: ClassVar[str]

    def configure(self, parser: argparse.ArgumentParser) -> None:
        return None

    @abstractmethod
    def run(self, args: argparse.Namespace, console: Console) -> int: ...


class StateCommand(Command, ABC):
    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--state-dir", type=Path, help="immune state directory (default: IMMUNE_HOME or ~/.immune)")

    @staticmethod
    def settings(args: argparse.Namespace) -> Settings:
        return SettingsLoader().load(state_dir=args.state_dir)

    @classmethod
    def store(cls, args: argparse.Namespace) -> StateStore:
        settings = cls.settings(args)
        return StateStore(StateBackends.open(settings), str(settings.resolved_state_dir()))
