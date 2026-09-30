from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

TOUR = Path(__file__).resolve().parents[2] / "examples" / "08_feature_tour.py"


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOUR), *arguments], capture_output=True, text=True, timeout=600, check=False
    )


def test_the_tour_lists_its_demos() -> None:
    listed = run("--list")
    assert listed.returncode == 0, listed.stderr
    assert "config-file" in listed.stdout


@pytest.mark.skipif(os.environ.get("IMMUNE_RUN_TOUR") != "1", reason="set IMMUNE_RUN_TOUR=1 to run the tour (~45 s)")
def test_every_check_in_the_tour_passes() -> None:
    completed = run("--quiet", "--no-color")
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-2000:]
    assert "ALL CHECKS PASSED" in completed.stdout
