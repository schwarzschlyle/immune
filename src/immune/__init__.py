from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from immune.errors import Blocked, ConfigError, ImmuneError
from immune.types import Action, Hit, Mode, SensorInfo, SiteStatus, Stage, Taint, Verdict

if TYPE_CHECKING:
    from immune.api import (
        configure,
        coverage,
        feedback,
        init,
        on_alert,
        on_verdict,
        protect,
        protected,
        runtime,
        session,
        shutdown,
        site,
        status,
        untrusted,
        verdict,
    )

try:
    from immune._version import __version__
except ImportError:
    __version__ = "0.0.0"

_API = frozenset(
    {
        "configure",
        "coverage",
        "feedback",
        "init",
        "on_alert",
        "on_verdict",
        "protect",
        "protected",
        "runtime",
        "session",
        "shutdown",
        "site",
        "status",
        "untrusted",
        "verdict",
    }
)

__all__ = [
    "Action",
    "Blocked",
    "ConfigError",
    "Hit",
    "ImmuneError",
    "Mode",
    "SensorInfo",
    "SiteStatus",
    "Stage",
    "Taint",
    "Verdict",
    "__version__",
    "configure",
    "coverage",
    "feedback",
    "init",
    "on_alert",
    "on_verdict",
    "protect",
    "protected",
    "runtime",
    "session",
    "shutdown",
    "site",
    "status",
    "untrusted",
    "verdict",
]


if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name in _API:
            value = getattr(importlib.import_module("immune.api"), name)
            globals()[name] = value
            return value
        raise AttributeError(f"module 'immune' has no attribute {name!r}")

    def __dir__() -> list[str]:
        return sorted({*globals(), *_API})
