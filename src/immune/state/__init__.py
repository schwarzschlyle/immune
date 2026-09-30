from __future__ import annotations

from immune.config.settings import Settings
from immune.errors import ConfigError
from immune.state.backend import StateBackend
from immune.state.local import LocalBackend
from immune.state.resilient import ResilientBackend
from immune.state.sqlite import SqliteBackend

__all__ = ["LocalBackend", "ResilientBackend", "SqliteBackend", "StateBackend", "StateBackends"]


class StateBackends:
    @staticmethod
    def open(settings: Settings) -> StateBackend:
        state = settings.state
        directory = settings.resolved_state_dir()
        if state.backend == "local":
            return LocalBackend(directory)
        if state.backend == "sqlite":
            primary: StateBackend = SqliteBackend(state.path or directory / "immune.db")
        else:
            if not state.url:
                raise ConfigError("state.backend is redis but state.url is not set")
            from immune.state.redis import RedisBackend

            primary = RedisBackend.from_url(state.url, state.key_prefix)
        return ResilientBackend(primary, LocalBackend(), state.failover_cooldown_s)
