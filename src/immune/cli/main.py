from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from immune.cli.commands import COMMANDS, Command
from immune.cli.commands.base import missing_sdk
from immune.cli.console import Console
from immune.errors import ImmuneError


class CommandLine:
    def __init__(self, commands: Sequence[Command] = COMMANDS, console: Console | None = None) -> None:
        self._commands = {command.name: command for command in commands}
        self._console = console or Console()

    def parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="immune", description="A general immune system for LLM applications.")
        parser.add_argument("-v", "--verbose", action="store_true", help="show immune log messages")
        subparsers = parser.add_subparsers(dest="command", required=True)
        for command in self._commands.values():
            command.configure(subparsers.add_parser(command.name, help=command.help, description=command.help))
        return parser

    def run(self, argv: Sequence[str] | None = None) -> int:
        args = self.parser().parse_args(argv)
        logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
        try:
            return self._commands[args.command].run(args, self._console)
        except ImmuneError as error:
            self._console.line(f"error: {error}")
            return 2
        except ModuleNotFoundError as error:
            package = missing_sdk(error)
            if package is None:
                raise
            self._console.line(f"error: this command needs the {package} package: pip install {package}")
            return 2


def main(argv: Sequence[str] | None = None) -> int:
    return CommandLine().run(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
