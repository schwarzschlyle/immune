from immune.testing.scenarios import ScenarioLibrary, ScenarioRunner

runner = ScenarioRunner()
for scenario in ScenarioLibrary().all():
    result = runner.run(scenario)
    verdict = result.turns[-1].verdict
    status = "pass" if result.passed else "FAIL"
    print(f"{status:4}  {scenario.id:32}  {verdict.explanation if verdict else ''}")
    print(f"      {scenario.title}")
