from __future__ import annotations

import functools
import importlib

from immune.errors import Blocked
from immune.types import Verdict

_SDK_ERRORS = {
    "openai": ("openai", "OpenAIError"),
    "anthropic": ("anthropic", "AnthropicError"),
}


class BlockedErrors:
    @staticmethod
    def raise_for(provider: str, verdict: Verdict) -> None:
        raise BlockedErrors.for_provider(provider)(verdict)

    @staticmethod
    @functools.cache
    def for_provider(provider: str) -> type[Blocked]:
        family = provider.split("_", maxsplit=1)[0]
        location = _SDK_ERRORS.get(family)
        if location is None:
            return Blocked
        module_name, class_name = location
        try:
            base = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            return Blocked
        name = f"{family.capitalize()}Blocked"
        return type(name, (Blocked, base), {"__module__": Blocked.__module__})
