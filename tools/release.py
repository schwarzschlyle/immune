"""Release helper: the next version, the changelog roll-over, tag checks and release notes.

Versions come from git tags (hatch-vcs), so a release is a tag. This tool keeps the tag, the changelog and semantic
versioning consistent:

    python tools/release.py next --bump minor            # 0.1.0 -> 0.2.0
    python tools/release.py next --bump patch --rc       # 0.1.0 -> 0.1.1rc1 (or rc2 if rc1 exists)
    python tools/release.py prepare 0.2.0                 # [Unreleased] -> [0.2.0] - today in CHANGELOG.md
    python tools/release.py check v0.2.0                  # the tag is valid and the changelog has its section
    python tools/release.py notes 0.2.0                   # the changelog section, for the GitHub release
    python tools/release.py dev-version --number 57       # 0.2.0.dev57: the version of a dev or staging build
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
REPOSITORY = "https://github.com/schwarzschlyle/immune"
_VERSION = re.compile(r"^v?(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)(?:rc(?P<rc>[1-9]\d*))?$")
_SECTION = re.compile(r"^## \[(?P<version>[^\]]+)\](?: - (?P<date>\d{4}-\d{2}-\d{2}))?[ \t]*$", re.M)


@dataclass(frozen=True, order=True)
class Version:
    major: int
    minor: int
    patch: int
    rc: int | None = None

    @classmethod
    def parse(cls, text: str) -> Version:
        match = _VERSION.match(text.strip())
        if match is None:
            raise ValueError(f"{text!r} is not a release version (X.Y.Z or X.Y.ZrcN, optionally prefixed with v)")
        rc = match["rc"]
        return cls(int(match["major"]), int(match["minor"]), int(match["patch"]), int(rc) if rc else None)

    @property
    def final(self) -> Version:
        return Version(self.major, self.minor, self.patch)

    @property
    def is_prerelease(self) -> bool:
        return self.rc is not None

    def bumped(self, part: str) -> Version:
        if part == "major":
            return Version(self.major + 1, 0, 0)
        if part == "minor":
            return Version(self.major, self.minor + 1, 0)
        if part == "patch":
            return Version(self.major, self.minor, self.patch + 1)
        raise ValueError(f"unknown bump {part!r}; use major, minor or patch")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}" + (f"rc{self.rc}" if self.rc else "")


def tagged_versions(tags: list[str] | None = None) -> list[Version]:
    if tags is None:
        output = subprocess.run(["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True, cwd=ROOT)
        tags = output.stdout.split()
    versions = []
    for tag in tags:
        try:
            versions.append(Version.parse(tag))
        except ValueError:
            continue
    return sorted(versions, key=_sort_key)


def _sort_key(version: Version) -> tuple[int, int, int, int]:
    # A release candidate sorts before its final version: 0.2.0rc1 < 0.2.0rc2 < 0.2.0.
    return version.major, version.minor, version.patch, version.rc if version.rc is not None else 1_000_000


def next_version(bump: str, candidate: bool, tags: list[str] | None = None) -> Version:
    versions = tagged_versions(tags)
    finals = [version for version in versions if not version.is_prerelease]
    target = (finals[-1] if finals else Version(0, 0, 0)).bumped(bump)
    if not candidate:
        return target
    candidates = [version.rc or 0 for version in versions if version.is_prerelease and version.final == target]
    return Version(target.major, target.minor, target.patch, max(candidates, default=0) + 1)


def dev_version(number: int, tags: list[str] | None = None) -> str:
    """The version of a dev or staging build, which TestPyPI receives.

    Builds between releases are development releases of the next version: the next minor after the latest release
    (0.1.0 -> 0.2.0.devN), or the next candidate after a release candidate (0.2.0rc1 -> 0.2.0rc2.devN), so they sort
    above every earlier version. N is the workflow's run number, which is unique across branches; git distance,
    the default, is not, and a dev and a staging build could otherwise claim the same version.
    """
    versions = tagged_versions(tags)
    latest = versions[-1] if versions else None
    if latest is None:
        base = Version(0, 1, 0)
    elif latest.is_prerelease:
        base = Version(latest.major, latest.minor, latest.patch, (latest.rc or 0) + 1)
    else:
        base = latest.bumped("minor")
    return f"{base}.dev{number}"


def sections(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    matches = list(_SECTION.finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        body = re.split(r"^\[[^\]]+\]: ", body, maxsplit=1, flags=re.M)[0]
        found[match["version"]] = body.strip()
    return found


def prepare(version: Version, text: str, today: datetime.date) -> str:
    """Move the [Unreleased] notes under a new version heading and update the link references."""
    if version.is_prerelease:
        raise ValueError("release candidates share their final version's changelog section; prepare the final version")
    current = sections(text)
    if str(version) in current:
        raise ValueError(f"CHANGELOG.md already has a section for {version}")
    if not current.get("Unreleased"):
        raise ValueError("the [Unreleased] section is empty; add the changes first")
    heading = f"## [Unreleased]\n\n## [{version}] - {today.isoformat()}"
    text = re.sub(r"^## \[Unreleased\][ \t]*$", heading, text, count=1, flags=re.M)
    previous = [key for key in current if key != "Unreleased"]
    if previous:
        compare = f"{REPOSITORY}/compare/v{previous[0]}...v{version}"
    else:
        compare = f"{REPOSITORY}/releases/tag/v{version}"
    links = f"[Unreleased]: {REPOSITORY}/compare/v{version}...HEAD\n[{version}]: {compare}\n"
    text = re.sub(r"^\[Unreleased\]: .*\n", links, text, count=1, flags=re.M)
    if links not in text:
        text = text.rstrip("\n") + "\n\n" + links
    return text


def check(tag: str, text: str) -> list[str]:
    problems: list[str] = []
    try:
        version = Version.parse(tag)
    except ValueError as error:
        return [str(error)]
    if not tag.startswith("v"):
        problems.append(f"tags start with v: use v{version}")
    if str(version.final) not in sections(text):
        problems.append(f"CHANGELOG.md has no '## [{version.final}] - YYYY-MM-DD' section")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    next_parser = commands.add_parser("next", help="print the next version")
    next_parser.add_argument("--bump", choices=["major", "minor", "patch"], required=True)
    next_parser.add_argument("--rc", action="store_true", help="a release candidate of that version")
    prepare_parser = commands.add_parser("prepare", help="roll the changelog's [Unreleased] section over")
    prepare_parser.add_argument("version")
    check_parser = commands.add_parser("check", help="validate a release tag against the changelog")
    check_parser.add_argument("tag")
    notes_parser = commands.add_parser("notes", help="print a version's changelog section")
    notes_parser.add_argument("version")
    dev_parser = commands.add_parser("dev-version", help="print the version of a dev or staging build")
    dev_parser.add_argument("--number", type=int, required=True, help="the build number (the workflow run number)")
    args = parser.parse_args(argv)

    if args.command == "dev-version":
        print(dev_version(args.number))
        return 0
    text = CHANGELOG.read_text(encoding="utf-8")
    if args.command == "next":
        print(next_version(args.bump, args.rc))
        return 0
    if args.command == "prepare":
        CHANGELOG.write_text(prepare(Version.parse(args.version), text, datetime.date.today()), encoding="utf-8")
        print(f"CHANGELOG.md: [Unreleased] is now [{Version.parse(args.version)}]")
        return 0
    if args.command == "check":
        problems = check(args.tag, text)
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 1 if problems else 0
    notes = sections(text).get(str(Version.parse(args.version).final))
    if notes is None:
        print(f"error: CHANGELOG.md has no section for {args.version}", file=sys.stderr)
        return 1
    print(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
