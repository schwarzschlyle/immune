from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from immune.sensing.offline import MockSensor
from immune.testing.harness import ImmuneHarness

HarnessFactory = Callable[..., ImmuneHarness]


@pytest.fixture
def immune_mock_sensor() -> MockSensor:
    return MockSensor()


@pytest.fixture
def immune_harness(tmp_path: Path) -> Iterator[HarnessFactory]:
    created: list[ImmuneHarness] = []

    def factory(**options: Any) -> ImmuneHarness:
        harness = ImmuneHarness(tmp_path / f"immune-state-{len(created)}", **options)
        created.append(harness)
        return harness

    yield factory
    for harness in created:
        harness.close()
