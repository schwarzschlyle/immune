"""The vaccine library: vaccines that ship with Immune, written and measured in the project's laboratory.

Every library vaccine is switched off until `vaccines.enabled` (or a site's) names it, and observed until enforced, so
upgrading Immune never changes what an application does. Library ids live in the reserved ``immune.`` namespace.

The packaged library loads from ``library/bundle.json``: documents the laboratory has already validated and measured,
read in one step. Contributors point ``vaccines.library`` at the YAML sources instead while they work on a vaccine.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from immune.config.yaml_io import load_yaml
from immune.vaccines.model import LIBRARY_PREFIX, Vaccine, VaccineError, describe

BUNDLE = "bundle.json"
BUNDLE_FORMAT = 1
_SUFFIXES = (".yaml", ".yml")
_ID_PARTS = 3


@dataclass(frozen=True, slots=True)
class LibraryEntry:
    vaccine: Vaccine
    source: Path
    digest: str
    card: Mapping[str, Any] = field(default_factory=dict)


def library_problems(vaccine: Vaccine) -> list[str]:
    """What keeps a vaccine out of the library; empty when it follows the library's rules."""
    problems: list[str] = []
    if not vaccine.library or len(vaccine.id.split(".")) != _ID_PARTS:
        problems.append(f"library ids are {LIBRARY_PREFIX}<domain>.<name>, such as immune.health.dosage_instructions")
    if vaccine.maturity is None:
        problems.append("set maturity: experimental, stable or deprecated")
    if vaccine.default != "off":
        problems.append('library vaccines ship switched off: set default: "off"')
    if vaccine.enforcement != "observe":
        problems.append("library vaccines ship observed: set enforcement: observe")
    if not vaccine.provenance.get("owner"):
        problems.append("set provenance.owner to the maintainer's GitHub handle")
    if vaccine.maturity == "stable" and not vaccine.provenance.get("references"):
        problems.append("a stable vaccine needs provenance.references: the incidents or guidance that motivate it")
    return problems


class LibraryCatalog:
    def __init__(self, entries: Sequence[LibraryEntry] = ()) -> None:
        self._entries: dict[str, LibraryEntry] = {}
        for entry in entries:
            existing = self._entries.get(entry.vaccine.id)
            if existing is not None:
                raise VaccineError(
                    f"library vaccine {entry.vaccine.id} is defined twice: {existing.source} and {entry.source}"
                )
            self._entries[entry.vaccine.id] = entry

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[LibraryEntry]:
        return [self._entries[key] for key in sorted(self._entries)]

    @property
    def ids(self) -> list[str]:
        return sorted(self._entries)

    def get(self, vaccine_id: str) -> LibraryEntry | None:
        return self._entries.get(vaccine_id)

    @classmethod
    def open(cls, setting: bool | Path, root: Path | None = None) -> LibraryCatalog:
        """The library that a `vaccines.library` setting names."""
        if setting is False:
            return cls()
        if setting is True:
            return cls.packaged()
        path = setting if setting.is_absolute() or root is None else root / setting
        return cls.from_bundle(path) if path.suffix == ".json" else cls.from_sources(path)

    @classmethod
    def packaged(cls) -> LibraryCatalog:
        folder = resources.files("immune.vaccines").joinpath("library")
        bundle = folder.joinpath(BUNDLE)
        if not bundle.is_file():
            return cls()
        return cls._from_document(json.loads(bundle.read_text(encoding="utf-8")), Path(str(folder)), BUNDLE)

    @classmethod
    def from_bundle(cls, path: Path) -> LibraryCatalog:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise VaccineError(f"cannot read the vaccine library bundle {path}: {error}") from error
        return cls._from_document(document, path.parent, str(path))

    @classmethod
    def from_sources(cls, path: Path) -> LibraryCatalog:
        if path.is_dir():
            files = sorted(item for item in path.rglob("*") if item.suffix in _SUFFIXES and item.is_file())
        elif path.is_file():
            files = [path]
        else:
            raise VaccineError(f"vaccine library {path} does not exist")
        return cls([cls.read_source(file) for file in files])

    @staticmethod
    def read_source(path: Path) -> LibraryEntry:
        try:
            raw = path.read_text(encoding="utf-8")
            document = load_yaml(raw)
        except (OSError, yaml.YAMLError) as error:
            raise VaccineError(f"cannot read library vaccine {path}: {error}") from error
        if not isinstance(document, dict):
            raise VaccineError(f"{path}: a vaccine file must contain one mapping with id, title, stage and detect")
        try:
            vaccine = Vaccine.model_validate(document)
        except ValidationError as error:
            raise VaccineError(f"{path}: {describe(error)}") from error
        problems = library_problems(vaccine)
        if problems:
            raise VaccineError(f"{path}: {'; '.join(problems)}")
        return LibraryEntry(vaccine, path, hashlib.sha256(raw.encode("utf-8")).hexdigest())

    def bundle(self, root: Path, cards: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """The bundle document for this library: sources relative to root, with each vaccine's card summary."""
        return {
            "format": BUNDLE_FORMAT,
            "vaccines": [
                {
                    "id": entry.vaccine.id,
                    "path": entry.source.relative_to(root).as_posix(),
                    "digest": entry.digest,
                    "document": entry.vaccine.model_dump(mode="json", exclude_defaults=True),
                    "card": dict(cards.get(entry.vaccine.id, {})),
                }
                for entry in self.entries
            ],
        }

    @classmethod
    def _from_document(cls, document: Any, folder: Path, where: str) -> LibraryCatalog:
        if not isinstance(document, dict) or document.get("format") != BUNDLE_FORMAT:
            raise VaccineError(f"{where}: not a vaccine library bundle (format {BUNDLE_FORMAT})")
        entries: list[LibraryEntry] = []
        for item in document.get("vaccines", []):
            try:
                vaccine = Vaccine.model_validate(item["document"])
                entries.append(LibraryEntry(vaccine, folder / item["path"], item["digest"], item.get("card") or {}))
            except (KeyError, TypeError) as error:
                raise VaccineError(f"{where}: a library entry is missing {error}") from error
            except ValidationError as error:
                raise VaccineError(f"{where}: library vaccine {item.get('id')}: {describe(error)}") from error
        return cls(entries)
