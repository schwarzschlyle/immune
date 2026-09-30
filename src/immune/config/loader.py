from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from immune.config.settings import Settings
from immune.config.yaml_io import load_yaml
from immune.errors import ConfigError
from immune.types import Mode

_CONFIG_ENV = "IMMUNE_CONFIG"
_MODE_ENV = "IMMUNE_MODE"
_STATE_URL_ENV = "IMMUNE_STATE_URL"
_REDIS_SCHEMES = ("redis://", "rediss://", "unix://")
_SQLITE_SCHEME = "sqlite:///"
_DEFAULT_FILE = "immune.yaml"
_FILE_PATHS = (("state_dir",), ("state", "path"), ("heads", "artifact"), ("privacy", "verdict_log"))


class SettingsLoader:
    def __init__(self, environ: Mapping[str, str] | None = None, cwd: Path | None = None) -> None:
        self._environ = os.environ if environ is None else environ
        self._cwd = cwd or Path.cwd()

    def load(self, config: str | Path | Mapping[str, Any] | Settings | None = None, **overrides: Any) -> Settings:
        if isinstance(config, Settings):
            base: dict[str, Any] = config.model_dump()
        elif isinstance(config, Mapping):
            base = dict(config)
        else:
            base = self._read(self._locate(config))
        mode = self._environ.get(_MODE_ENV)
        if mode:
            base["mode"] = mode
        state_url = self._environ.get(_STATE_URL_ENV)
        if state_url:
            base["state"] = {**dict(base.get("state") or {}), **self._state_from_url(state_url)}
        base.update({key: value for key, value in overrides.items() if value is not None})
        try:
            return Settings.model_validate(base)
        except ValidationError as error:
            raise ConfigError(f"invalid immune configuration: {error}") from error

    def _locate(self, config: str | Path | None) -> Path | None:
        if config is not None:
            path = Path(config)
            if not path.exists():
                raise ConfigError(f"configuration file {path} does not exist")
            return path
        configured = self._environ.get(_CONFIG_ENV)
        if configured:
            return self._locate(configured)
        default = self._cwd / _DEFAULT_FILE
        return default if default.exists() else None

    @staticmethod
    def _state_from_url(url: str) -> dict[str, Any]:
        if url.startswith(_REDIS_SCHEMES):
            return {"backend": "redis", "url": url}
        if url.startswith(_SQLITE_SCHEME):
            return {"backend": "sqlite", "path": url.removeprefix(_SQLITE_SCHEME)}
        raise ConfigError(
            f"{_STATE_URL_ENV} must start with redis://, rediss://, unix:// or sqlite:///, got {url.split(':', 1)[0]!r}"
        )

    @staticmethod
    def _read(path: Path | None) -> dict[str, Any]:
        if path is None:
            return {}
        try:
            data = load_yaml(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            raise ConfigError(f"cannot read {path}: {error}") from error
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a mapping")
        return SettingsLoader._anchored(data, path.resolve().parent)

    @staticmethod
    def _anchored(data: dict[str, Any], base: Path) -> dict[str, Any]:
        """Resolve relative paths in a configuration file against the file's own directory."""

        def anchor(value: Any) -> Any:
            # A rooted path such as /opt/shared counts as anchored on Windows too, where it has no drive letter.
            if isinstance(value, str) and value and not value.startswith("~") and not Path(value).root:
                return str(base / value)
            return value

        for keys in _FILE_PATHS:
            section: Any = data
            for key in keys[:-1]:
                section = section.get(key) if isinstance(section, dict) else None
            if isinstance(section, dict) and keys[-1] in section:
                section[keys[-1]] = anchor(section[keys[-1]])
        vaccines = data.get("vaccines")
        if isinstance(vaccines, dict) and isinstance(vaccines.get("paths"), list):
            vaccines["paths"] = [anchor(item) for item in vaccines["paths"]]
        return data


def parse_mode(value: str | Mode | None) -> Mode | None:
    if value is None or isinstance(value, Mode):
        return value
    try:
        return Mode(value)
    except ValueError as error:
        raise ConfigError(f"unknown mode {value!r}; expected auto, observe, strict or off") from error
