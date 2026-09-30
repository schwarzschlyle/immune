from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip(
    "fastapi", reason="the showcase's own requirements aren't installed (fastapi, uvicorn); CI's docs job installs them"
)

SHOWCASE = Path(__file__).resolve().parents[2] / "examples" / "showcase"
SECRETS = ("OPENAI_API_KEY", "TYPESAFE_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING")


def offline_environment(tmp_path: Path) -> dict[str, str]:
    environment = {name: value for name, value in os.environ.items() if name not in SECRETS}
    environment.update(DEMO_SPEND_FILE=str(tmp_path / "spend.json"), NO_COLOR="1")
    return environment


def test_the_showcase_tests_pass(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=SHOWCASE,
        env=offline_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-2000:]


@pytest.mark.skipif(os.environ.get("IMMUNE_RUN_TOUR") != "1", reason="set IMMUNE_RUN_TOUR=1 to run the showcase demo")
def test_the_offline_showcase_demo_passes(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "showcase", "--offline", "demo"],
        cwd=SHOWCASE,
        env=offline_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-2000:]
    assert "checks passed" in completed.stdout
