from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from immune.config.spec import Spec

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "docs" / "reference" / "threats.md"
README = ROOT / "README.md"
_ROW = re.compile(
    r"^\| `(?P<threat>[a-z_]+\.[a-z_]+)` \| [^|]+ \| (?P<check>[^|]+) \| (?P<default>[^|]+) \| [^|]+ \|$", re.M
)
_FLOOR = re.compile(r"\((F\d+)\)")
_OWASP = re.compile(
    r"^\| \*\*(?P<id>(?:LLM|ASI)\d\d)\*\* [^|]+ \| [^|]+ \| (?P<count>\d+) \| (?P<floors>[^|]+) \|$", re.M
)


def catalog() -> list[re.Match[str]]:
    return list(_ROW.finditer(CATALOG.read_text(encoding="utf-8")))


def test_the_catalog_lists_every_threat_exactly_once() -> None:
    assert sorted(row["threat"] for row in catalog()) == sorted(Spec.default().threats)


def test_the_catalog_matches_each_threat_detector_and_floor() -> None:
    threats = Spec.default().threats
    for row in catalog():
        threat = threats[row["threat"]]
        expected = {"jev": "Jev", "candidate": "Reflex → Jev"}.get(threat.detector)
        assert (row["check"].strip() == expected) if expected else row["check"].startswith("Reflex on"), row["threat"]
        floor = _FLOOR.search(row["default"])
        assert (floor.group(1) if floor else None) == (threat.floor.id if threat.floor else None), row["threat"]


def test_the_readme_owasp_table_matches_the_spec() -> None:
    mapped: defaultdict[str, list[str]] = defaultdict(list)
    floors: defaultdict[str, set[str]] = defaultdict(set)
    for threat in Spec.default().threats.values():
        for framework in threat.frameworks:
            mapped[framework].append(threat.id)
            if threat.floor is not None:
                floors[framework].add(threat.floor.id)
    rows = {row["id"]: row for row in _OWASP.finditer(README.read_text(encoding="utf-8"))}
    assert sorted(rows) == sorted(mapped)
    for framework, row in rows.items():
        assert int(row["count"]) == len(mapped[framework]), framework
        listed = {item.strip() for item in row["floors"].split(",")} - {"—"}
        assert listed == floors[framework], framework
