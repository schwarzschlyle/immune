from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from immune.cli.console import Console
from immune.cli.main import CommandLine
from immune.config.settings import VaccineSettings
from immune.config.spec import QuestionSpec
from immune.sensing.offline import MockSensor
from immune.types import Stage
from immune.vaccines import VaccineError, VaccineLoader
from immune.vaccines.lab import Probe, SelfSamples, VaccineLab
from immune.vaccines.model import Vaccine

DOSAGE: dict[str, Any] = {
    "id": "acme.no_dosage_advice",
    "title": "Dosage advice",
    "stage": "output",
    "detect": {"questions": [{"key": "gives_dosage", "text": "The assistant output says how much medicine to take."}]},
    "applies_to": {"sites": ["pharmacy*"]},
    "tests": {"positives": ["Take two tablets every four hours."], "negatives": ["We open at 9am."]},
}
REFUNDS: dict[str, Any] = {
    "id": "acme.refund_over_limit",
    "title": "Large refunds",
    "stage": "tool",
    "detect": {"tool": "refund_order", "argument": {"path": "amount", "greater_than": 100}},
    "tests": {
        "positives": [{"tool": "refund_order", "arguments": {"amount": 250}}],
        "negatives": [{"tool": "refund_order", "arguments": {"amount": 20}}],
    },
}


def run(*argv: str) -> tuple[int, str]:
    buffer = io.StringIO()
    code = CommandLine(console=Console(buffer)).run(list(argv))
    return code, buffer.getvalue()


def write(directory: Path, document: dict[str, Any]) -> Path:
    path = directory / f"{document['id']}.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def load(path: Path) -> Any:
    return VaccineLoader(VaccineSettings(entry_points=False)).load_file(path)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("IMMUNE_CONFIG", raising=False)
    return tmp_path


class TestScaffolding:
    @pytest.mark.parametrize(
        ("kind", "stage"), [("keywords", "output"), ("regex", "input"), ("questions", "data"), ("tool", "tool")]
    )
    def test_new_files_validate_and_pass_their_own_tests(self, workspace: Path, kind: str, stage: str) -> None:
        code, output = run("vaccines", "new", "acme.example", "--kind", kind, "--stage", stage)
        assert code == 0, output
        path = workspace / "vaccines" / "acme.example.yaml"
        assert f"stage: {stage}" in path.read_text(encoding="utf-8")
        code, output = run("vaccines", "test", str(path))
        assert code == 0, output
        assert "1 vaccines, 2 examples: all passed" in output

    def test_python_scaffolds_document_the_function(self, workspace: Path) -> None:
        code, _ = run("vaccines", "new", "acme.vip_accounts", "--kind", "python")
        assert code == 0
        text = (workspace / "vaccines" / "acme.vip_accounts.yaml").read_text(encoding="utf-8")
        assert "python: your_package.vaccines:vip_accounts" in text
        assert "def vip_accounts(context: immune.vaccines.VaccineContext)" in text

    def test_new_does_not_overwrite(self, workspace: Path) -> None:
        assert run("vaccines", "new", "acme.example")[0] == 0
        code, output = run("vaccines", "new", "acme.example")
        assert code == 2
        assert "already exists; pass --force" in output
        assert run("vaccines", "new", "acme.example", "--force")[0] == 0

    def test_tool_rules_need_the_tool_stage(self, workspace: Path) -> None:
        code, output = run("vaccines", "new", "acme.example", "--kind", "tool", "--stage", "output")
        assert code == 2
        assert "a tool rule runs at stage tool" in output

    def test_bare_on_and_off_are_read_as_switches(self) -> None:
        document = yaml.safe_load("id: acme.x\ntitle: X\nstage: output\ndetect: {keywords: [x]}\ndefault: off\n")
        assert document["default"] is False
        assert Vaccine.model_validate(document).default == "off"

    def test_examples_must_match_the_stage(self, tmp_path: Path) -> None:
        path = write(tmp_path, {**REFUNDS, "tests": {"positives": ["refund 250"]}})
        with pytest.raises(VaccineError, match="tests for a tool vaccine are tool calls"):
            load(path)


class TestLab:
    def test_failing_examples_are_reported(self, workspace: Path) -> None:
        document = {**REFUNDS, "tests": {"positives": [{"tool": "refund_order", "arguments": {"amount": 90}}]}}
        code, output = run("vaccines", "test", str(write(workspace, document)))
        assert code == 1
        assert "FAIL" in output
        assert "did not fire" in output
        assert "1 of 1 vaccines failed: acme.refund_over_limit" in output

    def test_vaccines_without_tests_fail(self, workspace: Path) -> None:
        code, output = run("vaccines", "test", str(write(workspace, {**REFUNDS, "tests": {}})))
        assert code == 1
        assert "no tests" in output

    def test_site_scoped_vaccines_run_at_a_matching_site(self, tmp_path: Path) -> None:
        loaded = load(write(tmp_path, DOSAGE))
        assert VaccineLab().site_for(loaded.vaccine) == "pharmacy"
        report = VaccineLab().test(loaded)
        assert report.passed
        assert report.sensor == "scripted"
        assert "run with --live" in report.notes[0]
        elsewhere = VaccineLab(site="kiosk").test(loaded)
        assert [result.fired for result in elsewhere.results] == [False, False]
        assert not elsewhere.passed

    def test_scripted_answers_follow_the_head(self) -> None:
        detect = {
            "questions": [
                {"key": "gives_dosage", "text": "The assistant output gives a dose."},
                {"key": "cites_label", "text": "The assistant output quotes the product label."},
                {
                    "key": "topic",
                    "kind": "choice",
                    "text": "Topic?",
                    "options": {"dose": "Dose", "hours": "Hours"},
                    "flag": ["dose"],
                },
            ],
            "head": {"weights": {"gives_dosage": 2.0, "cites_label": -1.0, "topic": 1.0}},
        }
        vaccine = Vaccine.model_validate({**DOSAGE, "detect": detect})
        present = VaccineLab.answers(vaccine, present=True)
        absent = VaccineLab.answers(vaccine, present=False)
        prefix = "acme_no_dosage_advice__"
        assert (present[f"{prefix}gives_dosage"], present[f"{prefix}cites_label"], present[f"{prefix}topic"]) == (
            0.97,
            0.02,
            "dose",
        )
        assert (absent[f"{prefix}gives_dosage"], absent[f"{prefix}cites_label"], absent[f"{prefix}topic"]) == (
            0.02,
            0.97,
            "hours",
        )

    def test_a_live_sensor_judges_the_examples(self, tmp_path: Path) -> None:
        def judge(state: Any, question: QuestionSpec) -> float | None:
            if question.key != "acme_no_dosage_advice__gives_dosage":
                return None
            return 0.95 if "tablets" in json.dumps(state) else 0.05

        loaded = load(write(tmp_path, DOSAGE))
        report = VaccineLab(live=lambda: MockSensor(rules=[judge])).test(loaded)
        assert report.sensor == "live"
        assert report.notes == ()
        assert [(result.expected, result.fired) for result in report.results] == [(True, True), (False, False)]
        assert report.results[0].probability is not None

    def test_deterministic_vaccines_never_use_the_live_sensor(self, tmp_path: Path) -> None:
        def unavailable() -> MockSensor:
            raise AssertionError("a deterministic vaccine asked for Jev")

        assert VaccineLab(live=unavailable).test(load(write(tmp_path, REFUNDS))).passed


class TestTrials:
    def test_everyday_samples_exist_for_every_stage(self) -> None:
        samples = SelfSamples()
        for stage, minimum in ((Stage.INPUT, 30), (Stage.DATA, 20), (Stage.OUTPUT, 30), (Stage.TOOL, 20)):
            probes = samples.packaged(stage)
            assert len(probes) >= minimum
            assert all((probe.call is not None) == (stage is Stage.TOOL) for probe in probes)

    def test_trials_report_the_firing_rate(self, workspace: Path) -> None:
        document = {"id": "acme.classic", "title": "Classic", "stage": "output", "detect": {"keywords": ["Classic"]}}
        path = write(workspace, document)
        code, output = run("vaccines", "trial", str(path))
        assert code == 0
        assert "fired on 2 of 40 samples (5.0%)" in output
        assert "keyword 'Classic'" in output
        code, output = run("vaccines", "trial", str(path), "--max-rate", "0.01")
        assert code == 1
        assert "above --max-rate" in output

    def test_corpus_files_add_the_apps_own_traffic(self, workspace: Path) -> None:
        document = {"id": "acme.classic", "title": "Classic", "stage": "output", "detect": {"keywords": ["Classic"]}}
        path = write(workspace, document)
        (workspace / "replies.jsonl").write_text('"The Classic is back"\n{"text": "Hello"}\n\n', encoding="utf-8")
        (workspace / "replies.txt").write_text("One\nTwo Classic\n", encoding="utf-8")
        code, output = run(
            "vaccines", "trial", str(path), "--no-self", "--corpus", "replies.jsonl", "--corpus", "replies.txt"
        )
        assert code == 0
        assert "2 from replies.jsonl, 2 from replies.txt" in output
        assert "fired on 2 of 4 samples (50.0%)" in output

    def test_bad_corpus_lines_are_named(self, workspace: Path) -> None:
        (workspace / "bad.jsonl").write_text('{"text": "ok"}\n{"number": 3}\n', encoding="utf-8")
        with pytest.raises(VaccineError, match=r"bad.jsonl:2: use a string"):
            SelfSamples().corpus(workspace / "bad.jsonl")

    def test_tool_vaccines_need_tool_samples(self, tmp_path: Path) -> None:
        with pytest.raises(VaccineError, match="needs tool examples"):
            VaccineLab().trial(load(write(tmp_path, REFUNDS)), [Probe(text="refund please")], ["test"])

    def test_question_trials_need_jev(self, workspace: Path) -> None:
        code, output = run("vaccines", "trial", str(write(workspace, DOSAGE)))
        assert code == 2
        assert "its trial needs Jev: run with --live" in output

    def test_trials_find_vaccines_by_id(self, workspace: Path) -> None:
        (workspace / "vaccines").mkdir()
        write(workspace / "vaccines", REFUNDS)
        (workspace / "immune.yaml").write_text("vaccines: {paths: [vaccines/]}\n", encoding="utf-8")
        code, output = run("vaccines", "trial", "acme.refund_over_limit")
        assert code == 0, output
        assert "fired on 0 of 25 samples" in output
        code, output = run("vaccines", "trial", "acme.unknown")
        assert code == 2
        assert "known: acme.refund_over_limit" in output


class TestVaccinate:
    def test_builds_tests_and_writes_an_observed_vaccine(self, workspace: Path) -> None:
        code, output = run(
            "vaccinate",
            "acme.no_competitor_mentions",
            "--stage",
            "output",
            "--keyword",
            "Burger Palace",
            "--positive",
            "You might prefer the Burger Palace deal.",
            "--negative",
            "Our Classic is $9.",
            "--message",
            "I can only talk about Acme Burgers products.",
            "--site",
            "ordering*",
        )
        assert code == 0, output
        assert "fired on 0 of 40 samples" in output
        path = workspace / "vaccines" / "acme.no_competitor_mentions.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["enforcement"] == "observe"
        assert document["applies_to"] == {"sites": ["ordering*"]}
        assert document["tests"]["positives"] == ["You might prefer the Burger Palace deal."]
        assert document["provenance"] == {"created_by": "immune vaccinate"}
        assert load(path).vaccine.respond.message == "I can only talk about Acme Burgers products."

    def test_tool_examples_and_argument_rules(self, workspace: Path) -> None:
        code, output = run(
            "vaccinate",
            "acme.refund_over_limit",
            "--stage",
            "tool",
            "--tool",
            "refund_order",
            "--argument",
            "amount>100",
            "--positive",
            'refund_order {"amount": 250}',
            "--negative",
            'refund_order {"amount": 20}',
        )
        assert code == 0, output
        document = yaml.safe_load((workspace / "vaccines" / "acme.refund_over_limit.yaml").read_text("utf-8"))
        assert document["detect"] == {"tool": "refund_order", "argument": {"path": "amount", "greater_than": 100.0}}
        assert document["tests"]["positives"] == [{"tool": "refund_order", "arguments": {"amount": 250}}]

    def test_nothing_is_written_when_the_examples_fail(self, workspace: Path) -> None:
        code, output = run("vaccinate", "acme.bad", "--stage", "output", "--keyword", "Classic", "--positive", "Hi")
        assert code == 1
        assert "nothing was written" in output
        assert not (workspace / "vaccines").exists()

    def test_question_vaccines_skip_the_trial_offline(self, workspace: Path) -> None:
        code, output = run(
            "vaccinate",
            "acme.no_dosage_advice",
            "--stage",
            "output",
            "--question",
            "The assistant output says how much medicine to take.",
            "--positive",
            "Take two tablets.",
        )
        assert code == 0, output
        assert "trial skipped" in output
        document = yaml.safe_load((workspace / "vaccines" / "acme.no_dosage_advice.yaml").read_text("utf-8"))
        assert document["detect"]["questions"][0]["key"] == "q1"

    @pytest.mark.parametrize(
        ("extra", "message"),
        [
            (["--keyword", "x", "--python", "m:f", "--positive", "x"], "choose one detector"),
            (["--tool", "refund_order", "--positive", "x"], "--tool needs --stage tool"),
            (["--keyword", "x"], "at least one --positive"),
            (["--keyword", "x", "--argument", "amount>1", "--positive", "x"], "--argument needs --tool"),
        ],
    )
    def test_argument_errors(self, workspace: Path, extra: list[str], message: str) -> None:
        code, output = run("vaccinate", "acme.x", "--stage", "output", *extra)
        assert code == 2
        assert message in output


class TestListing:
    def test_list_explains_each_switch(self, workspace: Path) -> None:
        (workspace / "vaccines").mkdir()
        write(workspace / "vaccines", {**REFUNDS, "default": "off"})
        config = {
            "vaccines": {"paths": ["vaccines"], "disabled": ["output.claims_human", "tool.loop"]},
            "sites": {"kiosk": {"vaccines": {"enabled": ["output.claims_human", "acme.*"]}}},
        }
        (workspace / "immune.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        code, output = run("vaccines", "list")
        assert code == 0
        assert "vaccines.disabled 'output.claims_human'" in output
        assert "the vaccine says default: off" in output
        assert "REFUSED at startup without" in output
        assert "tool.loop (F6)" in output
        code, output = run("vaccines", "list", "--site", "kiosk", "--custom")
        assert "sites.kiosk.vaccines.enabled 'acme.*'" in output
        assert "output.claims_human" not in output
        code, output = run("vaccines", "list", "--off")
        assert "input.override" not in output
        assert "1 vaccines loaded" in output


class TestDoctor:
    def test_doctor_checks_vaccines_and_warns_about_cost_and_the_floor(
        self, workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        folder = workspace / "vaccines"
        folder.mkdir()
        for index in range(11):
            question = {"key": "q", "text": f"The assistant output breaks rule {index}."}
            write(folder, {**DOSAGE, "id": f"acme.rule_{index}", "detect": {"questions": [question]}})
        config = {"vaccines": {"paths": ["vaccines"]}, "sites": {"support": {"observe": ["output.exfil_link"]}}}
        (workspace / "immune.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        _, output = run("doctor", "--state-dir", str(workspace / "state"))
        assert "11 loaded" in output
        assert "warning: 11 custom questions add Jev tokens to every output request" in output
        assert "warning: only observed at sites.support.observe: output.exfil_link (F2)" in output

    def test_doctor_fails_on_a_broken_vaccine(self, workspace: Path) -> None:
        (workspace / "immune.yaml").write_text("vaccines: {paths: [missing/]}\n", encoding="utf-8")
        code, output = run("doctor", "--state-dir", str(workspace / "state"))
        assert code == 1
        assert "vaccine path" in output
        assert "does not exist" in output
