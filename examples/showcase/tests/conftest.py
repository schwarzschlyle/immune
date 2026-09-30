from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from showcase import budget
from showcase.app import Platform
from showcase.config import Options


@pytest.fixture
def platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Platform]:
    """An offline platform: scripted model and scripted Jev answers, no keys, no spend."""
    monkeypatch.setattr(budget, "SPEND_FILE", tmp_path / "spend.json")
    monkeypatch.setenv("DEMO_SPEND_FILE", str(tmp_path / "spend.json"))
    running = Platform(Options.detect(offline=True))
    yield running
    running.close()
