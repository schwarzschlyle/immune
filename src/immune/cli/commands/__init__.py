from immune.cli.commands.base import Command, StateCommand
from immune.cli.commands.catalog import ExplainCommand, ThreatsCommand, VersionCommand
from immune.cli.commands.config import ConfigCommand
from immune.cli.commands.evidence import EvidenceCommand
from immune.cli.commands.health import DoctorCommand
from immune.cli.commands.operations import ExportCommand, InitCommand, LabelCommand, PostureCommand
from immune.cli.commands.scenarios import ReplayCommand, TestCommand
from immune.cli.commands.state import CalibrateCommand, PromoteCommand, StatusCommand
from immune.cli.commands.vaccines import VaccinateCommand, VaccinesCommand

COMMANDS: tuple[Command, ...] = (
    TestCommand(),
    ReplayCommand(),
    ThreatsCommand(),
    ExplainCommand(),
    StatusCommand(),
    PostureCommand(),
    PromoteCommand(),
    LabelCommand(),
    CalibrateCommand(),
    EvidenceCommand(),
    VaccinesCommand(),
    VaccinateCommand(),
    ExportCommand(),
    InitCommand(),
    ConfigCommand(),
    DoctorCommand(),
    VersionCommand(),
)

__all__ = ["COMMANDS", "Command", "StateCommand"]
