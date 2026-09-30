from __future__ import annotations

import argparse
import json
from pathlib import Path

from immune.cli.commands.base import Command
from immune.cli.console import Console
from immune.config.loader import SettingsLoader
from immune.config.settings import Settings


class ConfigCommand(Command):
    name = "config"
    help = "validate an immune.yaml file or print its JSON Schema"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        actions = parser.add_subparsers(dest="action", required=True)
        validate = actions.add_parser("validate", help="check a configuration file")
        validate.add_argument("path", nargs="?", type=Path, help="defaults to IMMUNE_CONFIG or ./immune.yaml")
        actions.add_parser("schema", help="print the JSON Schema for immune.yaml")

    def run(self, args: argparse.Namespace, console: Console) -> int:
        if args.action == "schema":
            console.line(json.dumps(ConfigurationSchema.document(), indent=2))
            return 0
        settings = SettingsLoader().load(args.path)
        console.line(f"valid configuration (mode={settings.mode.value}, {len(settings.sites)} sites)")
        return 0


class ConfigurationSchema:
    ID = "https://raw.githubusercontent.com/schwarzschlyle/immune/main/schema/immune.schema.json"

    @classmethod
    def document(cls) -> dict[str, object]:
        schema = Settings.model_json_schema(mode="validation")
        return {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": cls.ID, **schema}
