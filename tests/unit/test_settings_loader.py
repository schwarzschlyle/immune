from __future__ import annotations

from pathlib import Path

import pytest

from immune.config.loader import SettingsLoader
from immune.errors import ConfigError
from immune.types import Mode


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestLocating:
    def test_immune_yaml_in_the_working_directory_is_read(self, tmp_path: Path) -> None:
        write(tmp_path / "immune.yaml", "mode: observe\n")
        assert SettingsLoader(environ={}, cwd=tmp_path).load().mode is Mode.OBSERVE

    def test_immune_config_names_another_file(self, tmp_path: Path) -> None:
        write(tmp_path / "immune.yaml", "mode: observe\n")
        other = write(tmp_path / "config" / "production.yaml", "mode: strict\n")
        loader = SettingsLoader(environ={"IMMUNE_CONFIG": str(other)}, cwd=tmp_path)
        assert loader.load().mode is Mode.STRICT

    def test_an_explicit_file_wins_over_immune_config(self, tmp_path: Path) -> None:
        named = write(tmp_path / "named.yaml", "mode: strict\n")
        explicit = write(tmp_path / "explicit.yaml", "mode: observe\n")
        loader = SettingsLoader(environ={"IMMUNE_CONFIG": str(named)}, cwd=tmp_path)
        assert loader.load(explicit).mode is Mode.OBSERVE

    def test_no_file_means_defaults(self, tmp_path: Path) -> None:
        assert SettingsLoader(environ={}, cwd=tmp_path).load().mode is Mode.AUTO

    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="does not exist"):
            SettingsLoader(environ={"IMMUNE_CONFIG": str(tmp_path / "nope.yaml")}, cwd=tmp_path).load()

    def test_environment_variables_override_the_file(self, tmp_path: Path) -> None:
        write(tmp_path / "immune.yaml", "mode: observe\nstate: {key_prefix: 'app:'}\n")
        environ = {"IMMUNE_MODE": "strict", "IMMUNE_STATE_URL": "redis://cache:6379/0"}
        settings = SettingsLoader(environ=environ, cwd=tmp_path).load()
        assert settings.mode is Mode.STRICT
        assert (settings.state.backend, settings.state.url, settings.state.key_prefix) == (
            "redis",
            "redis://cache:6379/0",
            "app:",
        )


class TestRelativePaths:
    def test_paths_in_a_file_are_relative_to_the_file(self, tmp_path: Path) -> None:
        config = write(
            tmp_path / "deploy" / "immune.yaml",
            "state_dir: state\n"
            "state: {backend: sqlite, path: data/immune.db}\n"
            "heads: {artifact: models/heads.json}\n"
            "privacy: {verdict_log: logs/verdicts.jsonl}\n"
            "vaccines: {paths: [vaccines/, /opt/shared/vaccines]}\n",
        )
        settings = SettingsLoader(environ={}, cwd=tmp_path / "elsewhere").load(config)
        base = config.parent.resolve()
        assert settings.state_dir == base / "state"
        assert settings.state.path == base / "data" / "immune.db"
        assert settings.heads.artifact == base / "models" / "heads.json"
        assert settings.privacy.verdict_log == base / "logs" / "verdicts.jsonl"
        assert settings.vaccines.paths == (base / "vaccines", Path("/opt/shared/vaccines"))

    def test_home_relative_paths_are_left_alone(self, tmp_path: Path) -> None:
        config = write(tmp_path / "immune.yaml", "state_dir: ~/.immune-app\n")
        assert SettingsLoader(environ={}).load(config).state_dir == Path("~/.immune-app")

    def test_mappings_passed_in_code_stay_relative_to_the_working_directory(self) -> None:
        settings = SettingsLoader(environ={}).load({"vaccines": {"paths": ["vaccines/"]}})
        assert settings.vaccines.paths == (Path("vaccines/"),)
