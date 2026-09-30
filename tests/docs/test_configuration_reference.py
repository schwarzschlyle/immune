from __future__ import annotations

import re
import typing
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from immune.config.loader import SettingsLoader
from immune.config.settings import Settings, SiteSettings, ToolSettings

REFERENCE = Path(__file__).resolve().parents[2] / "docs" / "guides" / "configuration.md"
_YAML = re.compile(r"^```yaml\n(?P<body>.*?)^```", re.M | re.S)


def reference() -> dict[str, Any]:
    match = _YAML.search(REFERENCE.read_text(encoding="utf-8"))
    assert match is not None
    document: dict[str, Any] = yaml.safe_load(match["body"])
    return document


def missing(model: type[BaseModel], data: Mapping[str, Any], prefix: str = "") -> Iterator[str]:
    for name, field in model.model_fields.items():
        if name not in data:
            yield f"{prefix}{name}"
            continue
        nested = _nested(field.annotation)
        if nested is not None and isinstance(data[name], Mapping):
            entries = data[name].values() if nested in (SiteSettings, ToolSettings) else [data[name]]
            for entry in entries:
                yield from missing(nested, entry, f"{prefix}{name}.")


def _nested(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    arguments = typing.get_args(annotation)
    if arguments and isinstance(arguments[-1], type) and issubclass(arguments[-1], BaseModel):
        return arguments[-1]
    return None


def test_the_reference_sets_every_option() -> None:
    assert list(missing(Settings, reference())) == []


def test_the_reference_shows_the_defaults() -> None:
    loaded = SettingsLoader(environ={}).load(reference())
    defaults = Settings()
    for name in ("mode", "sensor", "privacy", "promotion", "limits", "telemetry", "vaccines"):
        assert getattr(loaded, name) == getattr(defaults, name), name
