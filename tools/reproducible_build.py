from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commit_time() -> str:
    try:
        output = subprocess.run(
            ["git", "log", "-1", "--format=%ct"], capture_output=True, text=True, check=True, cwd=ROOT
        )
    except (OSError, subprocess.CalledProcessError):
        return "1767225600"
    return output.stdout.strip() or "1767225600"


def build(destination: Path, epoch: str) -> dict[str, str]:
    environment = {**os.environ, "SOURCE_DATE_EPOCH": epoch}
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(destination)],
        check=True,
        cwd=ROOT,
        env=environment,
        capture_output=True,
    )
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(destination.iterdir())}


def main() -> int:
    epoch = commit_time()
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        one, two = build(Path(first), epoch), build(Path(second), epoch)
    for name in sorted(one.keys() | two.keys()):
        status = "identical" if one.get(name) == two.get(name) else "DIFFERENT"
        print(f"{status}  {name}  {one.get(name, '-')[:16]}")
    return 0 if one == two else 1


if __name__ == "__main__":
    raise SystemExit(main())
