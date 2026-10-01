from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from tools.vaccine_lab import (
    GATES,
    LabError,
    Layout,
    best_threshold,
    check,
    drift,
    fit,
    index,
    main,
    measure,
    new,
    record,
    write_card,
    write_fit,
)

from immune.config.spec import QuestionSpec
from immune.sensing.offline import MockSensor
from immune.types import Stage
from immune.vaccines import LibraryCatalog

VACCINE = "immune.test.marker"
EVERYDAY = ["We open at 9am.", "Your order has shipped.", "The museum is free on Sundays.", "Happy birthday, Sam!"]


def jev(state: Mapping[str, Any], question: QuestionSpec) -> float | None:
    """A stand-in for Jev: the reply is a case of the vaccine exactly when it says MARKER."""
    if not question.key.endswith("immune_test_marker__present"):
        return None
    return 0.95 if "MARKER" in json.dumps(state) else 0.03


def laboratory(tmp_path: Path) -> Layout:
    layout = Layout(tmp_path)
    corpus = layout.self_corpus(Stage.OUTPUT)
    corpus.parent.mkdir(parents=True)
    corpus.write_text("".join(json.dumps({"text": text}) + "\n" for text in EVERYDAY), encoding="utf-8")
    new(layout, VACCINE, "questions", Stage.OUTPUT, "Marker", "@maintainer")
    workspace = layout.workspace(VACCINE)
    document = yaml.safe_load(workspace.source.read_text(encoding="utf-8"))
    document["detect"]["questions"][0]["text"] = "The assistant output says MARKER."
    document["tests"] = {"positives": ["It says MARKER."], "negatives": ["It says nothing special."]}
    workspace.source.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    rows = []
    for index_, label in enumerate([True, False] * 8):
        reply = f"Row {index_} says MARKER." if label else f"Row {index_} is an ordinary reply."
        rows.append(
            {
                "id": f"row-{index_}",
                "source": "authored",
                "split": "train" if index_ < 8 else "test",
                "user": "Tell me something.",
                "reply": reply,
                "labels": {VACCINE: label},
            }
        )
    workspace.evidence.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return layout


@pytest.fixture
def small_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(GATES, "experimental", {**GATES["experimental"], "positives": 4, "hard_negatives": 4})


def test_new_scaffolds_a_valid_library_vaccine(tmp_path: Path) -> None:
    layout = Layout(tmp_path)
    paths = new(layout, "immune.health.example", "keywords", Stage.OUTPUT, "Example", "@maintainer")
    assert [path.name for path in paths] == ["example.yaml", "manifest.yaml", "evidence.jsonl", "NOTES.md"]
    entry = LibraryCatalog.read_source(paths[0])
    assert entry.vaccine.maturity == "experimental"
    assert entry.vaccine.default == "off"
    with pytest.raises(LabError, match="already exists"):
        new(layout, "immune.health.example", "keywords", Stage.OUTPUT, "Example", "@maintainer")
    with pytest.raises(LabError, match=r"immune\.<domain>\.<name>"):
        new(layout, "acme.example", "keywords", Stage.OUTPUT, "Example", "@maintainer")


def test_record_fit_measure_index_and_check(tmp_path: Path, small_gates: None) -> None:
    layout = laboratory(tmp_path)
    workspace = layout.workspace(VACCINE)
    recorded = record(workspace, MockSensor(rules=[jev]))
    assert recorded["requests"] == 16 + 2 + len(EVERYDAY), "evidence, inline tests and the self corpus"
    assert recorded["unanswered"] == 0
    assert recorded["model"] == "scripted (MockSensor)"

    result = fit(workspace)
    assert result.weights["present"] > 1.0
    assert result.f1 == 1.0
    write_fit(workspace, result)

    card = measure(workspace)
    assert card == measure(workspace), "a replayed measurement is reproducible"
    assert card["metrics"]["recall"] == 1.0
    assert card["metrics"]["hard_negative_fpr"] == 0.0
    assert card["metrics"]["self_rate"] == 0.0
    assert card["unanswered"] == 0
    write_card(workspace, card)

    assert index(layout) == ["src/immune/vaccines/library/bundle.json", "docs/reference/vaccine-catalog.md"]
    assert index(layout, check_only=True) == []
    bundled = LibraryCatalog.from_bundle(layout.library / "bundle.json")
    assert bundled.ids == [VACCINE]
    assert bundled.get(VACCINE).card["metrics"]["recall"] == 1.0  # type: ignore[union-attr]
    assert VACCINE in layout.catalog_page.read_text(encoding="utf-8")

    assert check(layout, [VACCINE]) == []
    assert main(["check", "--all"], layout) == 0


def test_check_catches_stale_cards_and_unmet_gates(tmp_path: Path, small_gates: None) -> None:
    layout = laboratory(tmp_path)
    workspace = layout.workspace(VACCINE)
    record(workspace, MockSensor(rules=[jev]))
    write_card(workspace, measure(workspace))
    with workspace.evidence.open("a", encoding="utf-8") as handle:
        row = {
            "id": "late",
            "source": "authored",
            "split": "test",
            "user": "Hi",
            "reply": "Hello.",
            "labels": {VACCINE: False},
        }
        handle.write(json.dumps(row) + "\n")
    problems = check(layout, [VACCINE])
    assert any("card.json is stale" in problem for problem in problems)
    assert any("no recorded Jev answer" in problem for problem in problems), "the new row was never recorded"


def test_drift_compares_a_fresh_recording_with_the_card(tmp_path: Path, small_gates: None) -> None:
    layout = laboratory(tmp_path)
    workspace = layout.workspace(VACCINE)
    record(workspace, MockSensor(rules=[jev]))
    write_card(workspace, measure(workspace))
    committed = workspace.cassette.read_text(encoding="utf-8")
    assert drift(workspace, MockSensor(rules=[jev])) == []
    assert workspace.cassette.read_text(encoding="utf-8") == committed, "drift never touches the committed recording"
    blind = MockSensor(signals={"present": 0.03})
    assert drift(workspace, blind)[0] == "recall fell from 1.0 to 0.0"


def test_check_applies_the_maturity_gates(tmp_path: Path) -> None:
    layout = laboratory(tmp_path)
    workspace = layout.workspace(VACCINE)
    record(workspace, MockSensor(rules=[jev]))
    write_card(workspace, measure(workspace))
    problems = check(layout, [VACCINE])
    assert f"{VACCINE}: positives is 8; experimental needs at least 25" in problems


def test_evidence_must_be_licensed_and_synthetic(tmp_path: Path) -> None:
    layout = laboratory(tmp_path)
    workspace = layout.workspace(VACCINE)
    row = {
        "id": "leaky",
        "source": "scraped",
        "user": "My card is 4111 1111 1111 1111",
        "reply": "Use key sk-proj-abcdefghijklmnopqrstuvwxyz0123456789 to log in.",
        "labels": {VACCINE: False},
    }
    with workspace.evidence.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    problems = check(layout, [VACCINE])
    assert any("source 'scraped' is not in manifest.yaml" in problem for problem in problems)
    assert any("looks like it contains a secret" in problem for problem in problems)
    assert any("looks like it contains personal data" in problem for problem in problems)


def test_the_best_threshold_keeps_the_widest_margin() -> None:
    assert best_threshold([0.9, 0.7, 0.2], [True, True, False]) == (0.6, 1.0)
    assert best_threshold([0.99, 0.97, 0.03], [True, True, False]) == (0.73, 1.0)
    assert best_threshold([0.9, 0.6, 0.55], [True, False, True])[1] < 1.0
