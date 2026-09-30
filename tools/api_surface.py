from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
from collections.abc import Iterable
from typing import Any

PUBLIC_MODULES = ("immune", "immune.testing", "immune.integrations.claude_agent", "immune.integrations.local")


class ApiSurface:
    def __init__(self, modules: Iterable[str] = PUBLIC_MODULES) -> None:
        self._modules = tuple(modules)

    def snapshot(self) -> dict[str, Any]:
        return {module: self._module(module) for module in self._modules}

    def _module(self, name: str) -> dict[str, Any]:
        module = importlib.import_module(name)
        exported = getattr(module, "__all__", None) or [
            attribute for attribute in vars(module) if not attribute.startswith("_")
        ]
        return {attribute: self._describe(getattr(module, attribute)) for attribute in sorted(exported)}

    def _describe(self, value: Any) -> Any:
        if isinstance(value, str):
            return "str"
        if inspect.isclass(value) and issubclass(value, enum.Enum):
            return {"enum": [member.value for member in value]}
        if inspect.isclass(value) and dataclasses.is_dataclass(value):
            return {
                "dataclass": {field.name: str(field.type).replace("'", "") for field in dataclasses.fields(value)},
                "members": self._members(value),
            }
        if inspect.isclass(value):
            return {"class": self._signature(value), "members": self._members(value)}
        if callable(value):
            return {"function": self._signature(value)}
        return type(value).__name__

    def _members(self, cls: type[Any]) -> dict[str, Any]:
        members: dict[str, Any] = {}
        for name, member in vars(cls).items():
            if name.startswith("_"):
                continue
            if isinstance(member, property):
                members[name] = "property"
            elif isinstance(member, (staticmethod, classmethod)):
                members[name] = self._signature(member.__func__)
            elif inspect.isfunction(member):
                members[name] = self._signature(member)
        return dict(sorted(members.items()))

    @staticmethod
    def _signature(value: Any) -> Any:
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            return "(...)"
        variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        return {
            "parameters": {
                name: {
                    "position": position if parameter.kind in _POSITIONAL else None,
                    "kind": parameter.kind.name,
                    "required": parameter.default is inspect.Parameter.empty and parameter.kind not in variadic,
                    "default": None if parameter.default is inspect.Parameter.empty else repr(parameter.default),
                    "annotation": _clean(parameter.annotation),
                }
                for position, (name, parameter) in enumerate(signature.parameters.items())
            },
            "returns": _clean(signature.return_annotation),
        }


_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


def _clean(annotation: Any) -> str:
    if annotation is inspect.Signature.empty:
        return ""
    return str(annotation).replace("'", "")


class ApiDifference:
    def __init__(self, expected: Any, actual: Any) -> None:
        self.removed: list[str] = []
        self.changed: list[str] = []
        self.added: list[str] = []
        self._compare("", expected, actual)

    @property
    def breaking(self) -> bool:
        return bool(self.removed or self.changed)

    @property
    def empty(self) -> bool:
        return not (self.removed or self.changed or self.added)

    def lines(self) -> list[str]:
        return [
            *(f"removed  {path}" for path in self.removed),
            *(f"changed  {path}" for path in self.changed),
            *(f"added    {path}" for path in self.added),
        ]

    def _compare(self, path: str, expected: Any, actual: Any) -> None:
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in expected.keys() - actual.keys():
                self.removed.append(f"{path}.{key}".lstrip("."))
            for key in actual.keys() - expected.keys():
                target = self.changed if _required(actual[key]) else self.added
                target.append(f"{path}.{key}".lstrip("."))
            for key in expected.keys() & actual.keys():
                self._compare(f"{path}.{key}", expected[key], actual[key])
        elif isinstance(expected, list) and isinstance(actual, list):
            self.removed.extend(f"{path}[{item}]".lstrip(".") for item in expected if item not in actual)
            self.added.extend(f"{path}[{item}]".lstrip(".") for item in actual if item not in expected)
        elif expected != actual:
            self.changed.append(f"{path}: {expected} -> {actual}".lstrip("."))


def _required(value: Any) -> bool:
    return isinstance(value, dict) and value.get("required") is True
