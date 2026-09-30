from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import sys
import threading
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

_LOGGER = logging.getLogger("immune")
Callback = Callable[[ModuleType], None]


class _NotifyingLoader(importlib.abc.Loader):
    def __init__(self, inner: importlib.abc.Loader, notify: Callable[[ModuleType], None]) -> None:
        self._inner = inner
        self._notify = notify

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._inner.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._inner.exec_module(module)
        self._notify(module)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class PostImportHooks(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._callbacks: dict[str, list[Callback]] = {}
        self._lock = threading.RLock()
        self._installed = False

    def when_imported(self, module_name: str, callback: Callback) -> None:
        with self._lock:
            loaded = sys.modules.get(module_name)
            if loaded is not None:
                self._run(callback, loaded)
                return
            self._callbacks.setdefault(module_name, []).append(callback)
            if not self._installed:
                sys.meta_path.insert(0, self)
                self._installed = True

    def clear(self) -> None:
        with self._lock:
            self._callbacks.clear()
            if self._installed:
                sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not self]
                self._installed = False

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in self._callbacks:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
                spec.loader = _NotifyingLoader(spec.loader, self._notify)
                return spec
        return None

    def _notify(self, module: ModuleType) -> None:
        with self._lock:
            callbacks = self._callbacks.pop(module.__name__, [])
        for callback in callbacks:
            self._run(callback, module)

    @staticmethod
    def _run(callback: Callback, module: ModuleType) -> None:
        try:
            callback(module)
        except Exception:
            _LOGGER.exception("immune: failed to instrument %s", module.__name__)
