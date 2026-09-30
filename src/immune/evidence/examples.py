from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from immune.config.yaml_io import load_yaml
from immune.errors import ConfigError

Split = Literal["train", "dev", "test"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExampleCall(_Model):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Example(_Model):
    id: str
    source: str
    split: Split = "train"
    language: str = "en"
    operator: str = "You are a helpful assistant."
    user: str
    data: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    reply: str = ""
    calls: tuple[ExampleCall, ...] = ()
    labels: dict[str, bool] = Field(default_factory=dict)


class DatasetSource(_Model):
    name: str
    path: str
    origin: str
    license: str
    permitted_use: str
    notes: str = ""


class DatasetManifest(_Model):
    sources: tuple[DatasetSource, ...]

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        try:
            return cls.model_validate(load_yaml(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            raise ConfigError(f"cannot read dataset manifest {path}: {error}") from error

    def examples(self, root: Path) -> Iterator[Example]:
        for source in self.sources:
            yield from ExampleSet.read(root / source.path)


class ExampleSet:
    @staticmethod
    def read(path: Path) -> Iterator[Example]:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield Example.model_validate(json.loads(line))
                    except ValueError as error:
                        raise ConfigError(f"{path}:{number}: {error}") from error
