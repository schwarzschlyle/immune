from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx2
import openai
import pytest
import yaml

import immune
from immune.cli.console import Console
from immune.cli.main import CommandLine
from immune.config.settings import VaccineSettings
from immune.config.spec import PanelFacts, QuestionSpec, Spec, ThreatSpec
from immune.core.decide import EnforcementPolicy
from immune.heads.promotion import PromotionPolicy
from immune.sensing.signals import SensorReading
from immune.telemetry.stats import PromotionLedger
from immune.testing import FakeProvider, FakeReply, ImmuneHarness, MockSensor
from immune.types import Mode, Stage
from immune.vaccines import LibraryCatalog, VaccineError, VaccineLoader
from immune.vaccines.catalog import library_problems
from tests.conftest import OPERATOR

Factory = Callable[..., ImmuneHarness]
GATES_RECALL = 0.8
CALLS: list[str] = []

ASKS: dict[str, Any] = {
    "id": "immune.test.asks",
    "title": "The reply says hello",
    "stage": "output",
    "maturity": "experimental",
    "default": "off",
    "detect": {"questions": [{"key": "says_hello", "text": "The assistant output says hello."}]},
    "provenance": {"owner": "@maintainer"},
}
COUNTS: dict[str, Any] = {
    "id": "immune.test.counts",
    "title": "Counts its calls",
    "stage": "output",
    "maturity": "experimental",
    "default": "off",
    "detect": {"python": "tests.integration.test_vaccine_library:count_call"},
    "provenance": {"owner": "@maintainer"},
}
OLD: dict[str, Any] = {
    **ASKS,
    "id": "immune.test.old",
    "maturity": "deprecated",
    "related": ["immune.test.asks"],
    "detect": {"keywords": ["hello"]},
}


def count_call(context: Any) -> None:
    CALLS.append(context.vaccine)


class AskedSensor(MockSensor):
    """A scripted sensor that remembers every question it was asked."""

    def __init__(self) -> None:
        super().__init__()
        self.asked: list[str] = []

    async def read(self, state: Mapping[str, Any], questions: Sequence[QuestionSpec]) -> SensorReading:
        self.asked.extend(question.key for question in questions)
        return await super().read(state, questions)


def library(directory: Path, *documents: dict[str, Any]) -> Path:
    folder = directory / "library"
    for document in documents:
        domain, name = document["id"].split(".")[1:]
        path = folder / domain / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return folder


def chat(harness: ImmuneHarness, site: str = "shop") -> None:
    with immune.site(site):
        harness.openai().chat.completions.create(
            model="gpt-5.5", messages=[{"role": "system", "content": OPERATOR}, {"role": "user", "content": "Hi"}]
        )


def library_questions(sensor: AskedSensor) -> list[str]:
    return [key for key in sensor.asked if key.startswith("immune_test_")]


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()


class TestOffIsFree:
    def test_a_switched_off_vaccine_asks_nothing_runs_nothing_and_builds_no_record(
        self, immune_harness: Factory, tmp_path: Path
    ) -> None:
        sensor = AskedSensor()
        config = {"vaccines": {"library": str(library(tmp_path, ASKS, COUNTS)), "entry_points": False}}
        harness = immune_harness(sensor=sensor, script=FakeReply(text="Hello there!"), config=config)
        chat(harness)
        assert sensor.asked, "the built-in output panel still runs"
        assert library_questions(sensor) == []
        assert CALLS == []
        stats = harness.runtime.ledger.stats(harness.verdict().site)
        assert not {"immune.test.asks", "immune.test.counts"} & set(stats)

    def test_switching_on_at_one_site_costs_only_there(self, immune_harness: Factory, tmp_path: Path) -> None:
        sensor = AskedSensor()
        config = {
            "vaccines": {"library": str(library(tmp_path, ASKS, COUNTS)), "entry_points": False},
            "sites": {"pharmacy": {"vaccines": {"enabled": ["immune.test.*"]}}},
        }
        harness = immune_harness(sensor=sensor, script=FakeReply(text="Hello there!"), config=config)
        chat(harness, site="shop")
        assert library_questions(sensor) == []
        assert CALLS == []
        chat(harness, site="pharmacy")
        assert library_questions(sensor) == ["immune_test_asks__says_hello"]
        assert CALLS == ["immune.test.counts"]

    def test_switching_on_is_live(self, tmp_path: Path) -> None:
        sensor = AskedSensor()
        immune.init(
            sensor=sensor,
            state_dir=tmp_path / "state",
            config={"vaccines": {"library": str(library(tmp_path, ASKS)), "entry_points": False}},
        )
        try:
            provider = FakeProvider(FakeReply(text="Hello there!"))
            client = openai.OpenAI(
                api_key="sk-test", http_client=httpx2.Client(transport=provider.transport(httpx2)), max_retries=0
            )

            def ask() -> None:
                messages = [{"role": "system", "content": OPERATOR}, {"role": "user", "content": "Hi"}]
                client.chat.completions.create(model="m", messages=messages)

            ask()
            assert library_questions(sensor) == []
            immune.configure(vaccines={"enabled": ["immune.test.asks"]})
            ask()
            assert library_questions(sensor) == ["immune_test_asks__says_hello"]
        finally:
            immune.shutdown()


class TestLibraryRules:
    def test_library_sources_must_follow_the_library_rules(self, tmp_path: Path) -> None:
        broken = {**ASKS, "default": "on", "enforcement": "enforce", "provenance": {}}
        broken.pop("maturity")
        with pytest.raises(VaccineError) as raised:
            LibraryCatalog.from_sources(library(tmp_path, broken))
        message = str(raised.value)
        assert "set maturity" in message
        assert "ship switched off" in message
        assert "ship observed" in message
        assert "provenance.owner" in message

    def test_custom_vaccines_cannot_use_the_reserved_namespace(self, tmp_path: Path) -> None:
        path = tmp_path / "mine.yaml"
        path.write_text(yaml.safe_dump({**ASKS, "maturity": None, "default": "on"}), encoding="utf-8")
        settings = VaccineSettings(paths=(path,), entry_points=False, library=False)
        with pytest.raises(VaccineError, match=r"immune\. namespace is reserved"):
            VaccineLoader(settings).load()

    def test_custom_vaccines_have_no_maturity(self, tmp_path: Path) -> None:
        path = tmp_path / "mine.yaml"
        path.write_text(yaml.safe_dump({**ASKS, "id": "acme.says_hello"}), encoding="utf-8")
        settings = VaccineSettings(paths=(path,), entry_points=False, library=False)
        with pytest.raises(VaccineError, match="maturity and related only apply to library vaccines"):
            VaccineLoader(settings).load()

    def test_a_bundle_round_trips_its_sources(self, tmp_path: Path) -> None:
        folder = library(tmp_path, ASKS, COUNTS, OLD)
        sources = LibraryCatalog.from_sources(folder)
        bundle = folder / "bundle.json"
        bundle.write_text(json.dumps(sources.bundle(folder, {"immune.test.asks": {"recall": 0.9}})), encoding="utf-8")
        loaded = LibraryCatalog.from_bundle(bundle)
        assert loaded.ids == sources.ids
        for entry in sources.entries:
            other = loaded.get(entry.vaccine.id)
            assert other is not None
            assert other.vaccine == entry.vaccine
            assert other.digest == entry.digest
            assert other.source == entry.source
        assert dict(loaded.get("immune.test.asks").card) == {"recall": 0.9}  # type: ignore[union-attr]

    def test_the_library_can_be_switched_off_entirely(self, tmp_path: Path) -> None:
        folder = library(tmp_path, ASKS)
        assert VaccineLoader(VaccineSettings(entry_points=False, library=folder)).load().ids == ["immune.test.asks"]
        assert VaccineLoader(VaccineSettings(entry_points=False, library=False)).load().ids == []


class TestLifecycle:
    def test_experimental_vaccines_are_never_promoted(self) -> None:
        class Promoted(PromotionLedger):
            def promoted(self, site: str, threat: str) -> bool:
                return True

        policy = EnforcementPolicy(Mode.AUTO, Promoted(PromotionPolicy()), unpromoted=frozenset({"immune.test.asks"}))
        experimental = ThreatSpec(
            id="immune.test.asks", invariant="U0", stage=Stage.OUTPUT, detector="jev", severity="low"
        )
        stable = experimental.model_copy(update={"id": "immune.test.stable"})
        assert not policy.could_enforce(experimental, "shop", None)
        assert policy.could_enforce(stable, "shop", None)

    def test_switching_on_a_deprecated_vaccine_names_its_replacement(
        self, immune_harness: Factory, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = {
            "vaccines": {"library": str(library(tmp_path, ASKS, OLD)), "entry_points": False, "enabled": ["*.old"]}
        }
        with caplog.at_level(logging.WARNING, logger="immune"):
            immune_harness(config=config)
        assert "vaccine immune.test.old is deprecated; switch on immune.test.asks instead" in caplog.text

    def test_a_library_function_that_cannot_load_is_skipped_with_a_warning(
        self, immune_harness: Factory, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        missing = {**COUNTS, "detect": {"python": "tests.no_such_module:detect"}}
        config = {
            "vaccines": {"library": str(library(tmp_path, missing)), "entry_points": False, "enabled": ["immune.*"]}
        }
        harness = immune_harness(script=FakeReply(text="Hello there!"), config=config)
        with caplog.at_level(logging.WARNING, logger="immune"):
            chat(harness)
            chat(harness)
        assert caplog.text.count("immune.test.counts is switched off because it cannot load") == 1


def test_off_vaccines_are_left_out_of_panels() -> None:
    owned = QuestionSpec(key="x__q", kind="noul", text="Q", owner="immune.test.asks")
    shared = QuestionSpec(key="built_in", kind="noul", text="B")
    panel = Spec.default().panel("output").model_copy(update={"questions": (owned, shared)})
    assert panel.applicable(PanelFacts()) == (owned, shared)
    assert panel.applicable(PanelFacts(off=frozenset({"immune.test.asks"}))) == (shared,)


PACKAGED = [
    "immune.brand.profanity",
    "immune.civic.political_persuasion",
    "immune.finance.personal_investment_advice",
    "immune.health.diagnosis",
    "immune.health.dosage_instructions",
    "immune.legal.case_specific_advice",
]


def run(*argv: str) -> tuple[int, str]:
    buffer = io.StringIO()
    code = CommandLine(console=Console(buffer)).run(list(argv))
    return code, buffer.getvalue()


class TestPackagedLibrary:
    def test_every_packaged_vaccine_ships_off_observed_and_measured(self) -> None:
        catalog = LibraryCatalog.packaged()
        assert catalog.ids == PACKAGED
        for entry in catalog.entries:
            assert library_problems(entry.vaccine) == []
            assert entry.source.is_file(), "the wheel ships each vaccine's source next to the bundle"
            assert entry.card["metrics"]["recall"] >= GATES_RECALL

    def test_an_application_loads_the_library_with_everything_off(self, immune_harness: Factory) -> None:
        harness = immune_harness(script=FakeReply(text="Take two 200 mg ibuprofen tablets every six hours."))
        assert set(PACKAGED) <= set(harness.runtime.vaccines.ids)
        chat(harness)
        verdict = harness.verdict()
        assert verdict is not None
        assert not {hit.threat for hit in verdict.hits} & set(PACKAGED)


class TestCommands:
    def test_list_shows_the_library(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        code, output = run("vaccines", "list", "--library")
        assert code == 0
        for vaccine_id in PACKAGED:
            assert vaccine_id in output
        assert "built-in" not in output
        assert "0 vaccines loaded, 6 library vaccines available (0 switched on)" in output

    def test_show_prints_the_card_and_how_to_switch_it_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        code, output = run("vaccines", "show", "immune.health.dosage_instructions")
        assert code == 0
        assert "library vaccine 1.0.0, experimental" in output
        assert "recall" in output
        assert "never promoted automatically" in output
        assert "enabled: [immune.health.dosage_instructions]" in output

    def test_fork_copies_a_library_vaccine_into_your_own(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        code, output = run("vaccines", "fork", "immune.health.dosage_instructions", "--as", "acme.dosage_instructions")
        assert code == 0
        path = tmp_path / "vaccines" / "acme.dosage_instructions.yaml"
        assert f"wrote {Path('vaccines') / 'acme.dosage_instructions.yaml'}" in output
        loaded = VaccineLoader(VaccineSettings(paths=(path,), entry_points=False, library=False)).load()
        fork = loaded.get("acme.dosage_instructions")
        assert fork is not None
        assert fork.vaccine.maturity is None
        assert fork.vaccine.default == "on"
        assert fork.vaccine.provenance == {"forked_from": "immune.health.dosage_instructions@1.0.0"}
        code, output = run("vaccines", "fork", "immune.health.diagnosis", "--as", "immune.mine.diagnosis")
        assert code == 2
        assert "reserved for the library" in output

    def test_vaccines_test_leaves_the_library_to_the_laboratory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        code, output = run("vaccines", "test")
        assert code == 1
        assert "no vaccines found" in output

    def test_doctor_counts_only_switched_on_library_vaccines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "immune.yaml").write_text(
            yaml.safe_dump({"vaccines": {"enabled": ["immune.health.*"]}}), encoding="utf-8"
        )
        _, output = run("doctor", "--state-dir", str(tmp_path / "state"))
        assert "library: 2 of 6 switched on (immune.health.diagnosis, immune.health.dosage_instructions)" in output
