from __future__ import annotations

import pytest

from immune.testing.scenarios import Scenario, ScenarioLibrary, ScenarioRunner

SCENARIOS = ScenarioLibrary().all()


def test_the_library_covers_incidents_chains_and_benign_twins() -> None:
    kinds = {scenario.kind for scenario in SCENARIOS}
    assert {"incident", "chain", "benign"} <= kinds


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[scenario.id for scenario in SCENARIOS])
def test_scenario(scenario: Scenario) -> None:
    result = ScenarioRunner().run(scenario)
    assert result.passed, "\n".join(result.failures)
