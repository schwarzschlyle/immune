"""YAML parsing that uses libyaml when PyYAML was built with it.

The pure-Python loader was most of `immune.init()`'s startup time; the C loader builds the same safe objects several
times faster.
"""

from __future__ import annotations

from typing import Any

import yaml

_SAFE_LOADER: type[yaml.SafeLoader] = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load_yaml(text: str) -> Any:
    """Parse one YAML document with a safe loader; raises `yaml.YAMLError` like `yaml.safe_load`."""
    return yaml.load(text, Loader=_SAFE_LOADER)  # noqa: S506 - always a SafeLoader (or its C version)
