from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from immune.config.settings import SiteSettings, VaccineSettings
from immune.config.spec import Spec
from immune.errors import ConfigError
from immune.types import Stage
from immune.vaccines import Switchboard, VaccineError, VaccineLoader
from immune.vaccines.patterns import PatternSafety, UnsafePattern

KEYWORDS: dict[str, Any] = {
    "id": "acme.no_competitor_mentions",
    "title": "The reply names a competitor",
    "stage": "output",
    "detect": {"keywords": ["Burger Palace"]},
}
QUESTIONS: dict[str, Any] = {
    "id": "acme.no_dosage_advice",
    "title": "Dosage advice",
    "stage": "output",
    "detect": {
        "questions": [
            {"key": "gives_dosage", "text": "The assistant output tells the user how much medicine to take."}
        ],
        "head": {"bias": -0.5, "weights": {"gives_dosage": 2.0}},
        "threshold": 0.8,
    },
    "applies_to": {"sites": ["pharmacy*"]},
}


def write(directory: Path, document: dict[str, Any], name: str | None = None) -> Path:
    path = directory / f"{name or document['id']}.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def load(directory: Path, **settings: Any) -> Any:
    return VaccineLoader(VaccineSettings(paths=(directory,), entry_points=False, library=False, **settings)).load()


def failure(directory: Path, document: dict[str, Any]) -> str:
    write(directory, document)
    with pytest.raises(VaccineError) as caught:
        load(directory)
    return str(caught.value)


class TestValidation:
    def test_ids_must_be_namespaced(self, tmp_path: Path) -> None:
        message = failure(tmp_path, {**KEYWORDS, "id": "competitors"})
        assert "competitors.yaml: id: use a namespaced lowercase id" in message

    def test_exactly_one_detector(self, tmp_path: Path) -> None:
        detect = {"keywords": ["x"], "questions": QUESTIONS["detect"]["questions"]}
        message = failure(tmp_path, {**KEYWORDS, "detect": detect})
        assert "detect: choose exactly one detector" in message

    def test_actions_must_fit_the_stage(self, tmp_path: Path) -> None:
        message = failure(tmp_path, {**KEYWORDS, "respond": {"tool": "hold"}})
        assert "respond.tool does not apply to the output stage; use software, text" in message

    def test_actions_must_fit_the_sink(self, tmp_path: Path) -> None:
        message = failure(tmp_path, {**KEYWORDS, "respond": {"text": "confirm"}})
        assert "respond.text cannot be confirm" in message

    def test_tool_questions_need_the_call_placeholder(self, tmp_path: Path) -> None:
        detect = {"questions": [{"key": "risky", "text": "The action is risky."}]}
        message = failure(tmp_path, {**KEYWORDS, "stage": "tool", "detect": detect})
        assert "must refer to the call with {call}" in message

    def test_choice_questions_need_a_flag(self, tmp_path: Path) -> None:
        question = {"key": "topic", "kind": "choice", "text": "Which topic?", "options": {"a": "A", "b": "B"}}
        message = failure(tmp_path, {**KEYWORDS, "detect": {"questions": [question]}})
        assert "needs `flag`" in message

    def test_python_references_must_import(self, tmp_path: Path) -> None:
        message = failure(tmp_path, {**KEYWORDS, "stage": "tool", "detect": {"python": "no_such_module:check"}})
        assert "detect.python 'no_such_module:check' cannot be imported" in message

    def test_duplicate_ids_are_rejected(self, tmp_path: Path) -> None:
        write(tmp_path, KEYWORDS, "first")
        write(tmp_path, KEYWORDS, "second")
        with pytest.raises(VaccineError, match="defined twice"):
            load(tmp_path)

    def test_missing_paths_are_reported(self, tmp_path: Path) -> None:
        with pytest.raises(VaccineError, match="does not exist"):
            load(tmp_path / "nowhere")


class TestPatternSafety:
    @pytest.mark.parametrize("pattern", [r"(?i)\bpromo[- ]?code\b", r"\d{3}-\d{2}-\d{4}", r"(?:free|gratis){1,3}"])
    def test_bounded_patterns_are_accepted(self, pattern: str) -> None:
        PatternSafety().check(pattern)

    @pytest.mark.parametrize(
        ("pattern", "reason"),
        [(r"a+b", "without a bound"), (r"(?:ab{1,5}){1,5}", "nests one repetition"), (r"x{2,}", "without a bound")],
    )
    def test_risky_patterns_are_rejected(self, pattern: str, reason: str) -> None:
        with pytest.raises(UnsafePattern, match=reason):
            PatternSafety().check(pattern)

    def test_the_loader_names_the_file_and_field(self, tmp_path: Path) -> None:
        message = failure(tmp_path, {**KEYWORDS, "detect": {"regex": ["(a+)+$"]}})
        assert "detect.regex:" in message
        assert "acme.no_competitor_mentions.yaml" in message


class TestCompilation:
    def test_vaccines_become_threats_heads_and_questions(self, tmp_path: Path) -> None:
        write(tmp_path, KEYWORDS)
        write(tmp_path, QUESTIONS)
        bundle = load(tmp_path)
        spec = bundle.compile(Spec.default())
        assert spec.threats["acme.no_competitor_mentions"].detector == "deterministic"
        threat = spec.threats["acme.no_dosage_advice"]
        assert (threat.detector, threat.threshold, threat.invariant) == ("jev", 0.8, "U0")
        head = spec.heads["acme.no_dosage_advice"]
        assert head.bias == -0.5
        assert head.features[0].signal == "acme_no_dosage_advice__gives_dosage"
        assert head.features[0].weight == 2.0
        question = next(q for q in spec.panel("output").questions if q.key == "acme_no_dosage_advice__gives_dosage")
        assert question.when.sites == ("pharmacy*",)
        assert spec.digest.endswith(bundle.label)
        assert "U0" in spec.invariants

    def test_built_in_ids_cannot_be_reused(self, tmp_path: Path) -> None:
        write(tmp_path, {**KEYWORDS, "id": "output.claims_human"})
        with pytest.raises(VaccineError, match="cannot reuse built-in threat ids"):
            load(tmp_path).compile(Spec.default())

    def test_an_empty_bundle_leaves_the_spec_alone(self) -> None:
        spec = Spec.default()
        assert VaccineLoader(VaccineSettings(entry_points=False, library=False)).load().compile(spec) is spec

    def test_entry_points_provide_vaccine_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        write(tmp_path, KEYWORDS)

        class EntryPoint:
            name = "acme"

            @staticmethod
            def load() -> Any:
                return lambda: [str(tmp_path)]

        monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [EntryPoint()])
        bundle = VaccineLoader(VaccineSettings(library=False)).load()
        assert bundle.ids == ["acme.no_competitor_mentions"]


class TestSwitchboard:
    def test_site_settings_win_over_global_ones(self) -> None:
        switchboard = Switchboard(
            VaccineSettings(disabled=("output.claims_human", "acme.*"), enabled=("acme.keep",)),
            default_off=frozenset({"acme.opt_in"}),
        )
        kiosk = SiteSettings.model_validate(
            {"vaccines": {"enabled": ["output.claims_human"], "disabled": ["acme.keep"]}}
        )
        assert switchboard.disabled("output.claims_human")
        assert not switchboard.disabled("output.claims_human", kiosk)
        assert switchboard.disabled("acme.anything")
        assert not switchboard.disabled("acme.keep")
        assert switchboard.disabled("acme.keep", kiosk)
        assert switchboard.disabled("acme.opt_in")
        assert not switchboard.disabled("input.off_task")

    def test_disabling_the_floor_needs_permission(self) -> None:
        from immune.config.loader import SettingsLoader

        settings = SettingsLoader(environ={}).load({"vaccines": {"disabled": ["output.*"]}})
        with pytest.raises(ConfigError, match=r"output.exfil_link \(F2\)"):
            Switchboard.validate(settings, Spec.default())
        allowed = SettingsLoader(environ={}).load({"vaccines": {"disabled": ["output.*"], "allow_floor_changes": True}})
        Switchboard.validate(allowed, Spec.default())

    def test_observing_the_floor_at_a_site_is_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        from immune.config.loader import SettingsLoader

        settings = SettingsLoader(environ={}).load({"sites": {"support": {"observe": ["output.exfil_*"]}}})
        assert Switchboard.floor_observed(settings, Spec.default()) == ["sites.support.observe: output.exfil_link (F2)"]
        Switchboard.validate(settings, Spec.default())
        assert "floor protection only observed at this site: sites.support.observe: output.exfil_link" in caplog.text

    def test_site_floor_changes_are_checked_too(self) -> None:
        from immune.config.loader import SettingsLoader

        settings = SettingsLoader(environ={}).load({"sites": {"kiosk": {"vaccines": {"disabled": ["tool.loop"]}}}})
        with pytest.raises(ConfigError, match=r"sites.kiosk.vaccines.disabled: tool.loop \(F6\)"):
            Switchboard.validate(settings, Spec.default())


def test_stages_are_limited_to_the_four_screened_channels(tmp_path: Path) -> None:
    message = failure(tmp_path, {**KEYWORDS, "stage": "operator"})
    assert "vaccines apply to the input, data, tool or output stage" in message
    assert Stage.OPERATOR.value == "operator"
