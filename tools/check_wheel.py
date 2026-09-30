from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from email.parser import Parser
from pathlib import Path

REQUIRED_IN_WHEEL = (
    "immune/py.typed",
    "immune/spec/threats.yaml",
    "immune/spec/heads.yaml",
    "immune/spec/templates.yaml",
    "immune/spec/endpoints.yaml",
    "immune/spec/version.txt",
)
FORBIDDEN_PREFIXES = ("tests/", "tools/", "bench/", "docs/")
REQUIRED_IN_SDIST = ("tests/", "scenarios/", "docs/", "LICENSE", "NOTICE", "README.md")
SMOKE = "import immune, tempfile; immune.init(state_dir=tempfile.mkdtemp()); print(immune.__version__)"


class WheelCheck:
    def __init__(self, dist: Path) -> None:
        self.wheel = next(dist.glob("*.whl"))
        self.sdist = next(dist.glob("*.tar.gz"))
        self.problems: list[str] = []

    def run(self, smoke: bool) -> list[str]:
        self._wheel_contents()
        self._sdist_contents()
        if smoke:
            self._smoke()
        return self.problems

    def _wheel_contents(self) -> None:
        with zipfile.ZipFile(self.wheel) as archive:
            names = set(archive.namelist())
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
            entry_points = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
            entries = archive.read(entry_points).decode("utf-8")
        self.problems.extend(f"wheel is missing {name}" for name in REQUIRED_IN_WHEEL if name not in names)
        self.problems.extend(f"wheel ships {name}" for name in sorted(names) if name.startswith(FORBIDDEN_PREFIXES))
        if metadata.get("Requires-Python") != ">=3.11":
            self.problems.append(f"unexpected Requires-Python {metadata.get('Requires-Python')}")
        if "immune = immune.cli.main:main" not in entries or "immune.testing.pytest_plugin" not in entries:
            self.problems.append("wheel is missing the immune console script or pytest plugin entry point")

    def _sdist_contents(self) -> None:
        with tarfile.open(self.sdist) as archive:
            names = [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]
        for required in REQUIRED_IN_SDIST:
            if not any(name == required or name.startswith(required) for name in names):
                self.problems.append(f"sdist is missing {required}")

    def _smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory) / "env"
            venv.create(environment, with_pip=True)
            python = environment / ("Scripts" if sys.platform == "win32" else "bin") / "python"
            subprocess.run([str(python), "-m", "pip", "install", "-q", str(self.wheel)], check=True)
            result = subprocess.run(
                [str(python), "-c", SMOKE], capture_output=True, text=True, cwd=directory, check=False
            )
            if result.returncode != 0:
                self.problems.append(f"installed wheel failed to initialize: {result.stderr.strip()[-400:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="check the built wheel and sdist")
    parser.add_argument("dist", type=Path)
    parser.add_argument("--no-smoke", action="store_true", help="skip installing the wheel into a clean environment")
    args = parser.parse_args()
    problems = WheelCheck(args.dist).run(smoke=not args.no_smoke)
    for problem in problems:
        print(f"problem: {problem}")
    print("distribution looks good" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
