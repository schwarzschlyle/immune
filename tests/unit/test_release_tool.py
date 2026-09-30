from __future__ import annotations

import datetime

import pytest
from tools.release import CHANGELOG, Version, check, dev_version, next_version, prepare, sections

LOG = """# Changelog

## [Unreleased]

### Fixed

- A bug.

## [0.1.0] - 2026-09-30

### Added

- Everything.

[Unreleased]: https://github.com/schwarzschlyle/immune/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/schwarzschlyle/immune/releases/tag/v0.1.0
"""


@pytest.mark.parametrize(
    ("bump", "candidate", "tags", "expected"),
    [
        ("minor", False, [], "0.1.0"),
        ("patch", False, ["v0.1.0"], "0.1.1"),
        ("minor", False, ["v0.1.0", "v0.1.1"], "0.2.0"),
        ("major", False, ["v0.9.3"], "1.0.0"),
        ("minor", True, ["v0.1.0"], "0.2.0rc1"),
        ("minor", True, ["v0.1.0", "v0.2.0rc1", "v0.2.0rc2"], "0.2.0rc3"),
        ("patch", False, ["v0.1.0", "v0.2.0rc1", "not-a-version"], "0.1.1"),
    ],
)
def test_next_version_follows_semantic_versioning(bump: str, candidate: bool, tags: list[str], expected: str) -> None:
    assert str(next_version(bump, candidate, tags)) == expected


def test_release_candidates_sort_before_their_final_version() -> None:
    assert Version.parse("v0.2.0rc1").final == Version.parse("0.2.0")
    with pytest.raises(ValueError, match="not a release version"):
        Version.parse("0.2.0.dev3")


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        ([], "0.1.0.dev7"),
        (["v0.1.0"], "0.2.0.dev7"),
        (["v0.1.0", "v0.1.1"], "0.2.0.dev7"),
        (["v0.1.0", "v0.2.0rc1"], "0.2.0rc2.dev7"),
        (["v0.1.0", "v0.2.0rc1", "v0.2.0"], "0.3.0.dev7"),
        (["v1.4.2", "not-a-version"], "1.5.0.dev7"),
    ],
)
def test_dev_builds_are_development_releases_of_the_next_version(tags: list[str], expected: str) -> None:
    assert dev_version(7, tags) == expected


def test_prepare_rolls_the_unreleased_section_over() -> None:
    rolled = prepare(Version.parse("0.2.0"), LOG, datetime.date(2026, 11, 13))
    assert "## [Unreleased]\n\n## [0.2.0] - 2026-11-13\n\n### Fixed" in rolled
    assert "[Unreleased]: https://github.com/schwarzschlyle/immune/compare/v0.2.0...HEAD" in rolled
    assert "[0.2.0]: https://github.com/schwarzschlyle/immune/compare/v0.1.0...v0.2.0" in rolled
    assert sections(rolled)["0.2.0"] == "### Fixed\n\n- A bug."
    with pytest.raises(ValueError, match="already has a section"):
        prepare(Version.parse("0.1.0"), LOG, datetime.date(2026, 11, 13))
    with pytest.raises(ValueError, match="empty"):
        prepare(Version.parse("0.3.0"), rolled, datetime.date(2026, 11, 13))


def test_check_requires_a_v_tag_with_a_changelog_section() -> None:
    assert check("v0.1.0", LOG) == []
    assert check("v0.1.0rc1", LOG) == []
    assert check("0.1.0", LOG) == ["tags start with v: use v0.1.0"]
    assert check("v0.2.0", LOG) == ["CHANGELOG.md has no '## [0.2.0] - YYYY-MM-DD' section"]


def test_the_real_changelog_has_a_section_for_every_release_heading() -> None:
    found = sections(CHANGELOG.read_text(encoding="utf-8"))
    assert "Unreleased" in found
    assert found["0.1.0"]
