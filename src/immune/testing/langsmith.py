from __future__ import annotations

import threading
import time
from typing import Any

from immune.telemetry.langsmith import use_client


class LangSmithRecorder:
    """Records what Immune would send to LangSmith, for tests. It stands in for ``langsmith.Client``.

    Install it before ``immune.init()`` or before building an ``ImmuneHarness``. While it is installed, new runtimes
    trace to it even when ``LANGSMITH_TRACING`` and ``LANGSMITH_API_KEY`` are unset::

        with LangSmithRecorder() as recorder:
            immune.init()
            ...  # make calls
            recorder.wait()
        assert recorder.immune_runs()[0]["name"] == "immune · rewrite · output.secret_leak"

    Runs are the keyword arguments ``RunTree.post`` passes to ``Client.create_run``; feedback is the arguments of
    ``Client.create_feedback``.
    """

    otel_exporter: Any = None

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.feedback: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._restore: Any = None

    def install(self) -> LangSmithRecorder:
        if self._restore is None:
            self._restore = use_client(self)
        return self

    def uninstall(self) -> None:
        if self._restore is not None:
            self._restore()
            self._restore = None

    def __enter__(self) -> LangSmithRecorder:
        return self.install()

    def __exit__(self, *exc_info: object) -> None:
        self.uninstall()

    def create_run(
        self, name: str, inputs: dict[str, Any], run_type: str, *, project_name: str | None = None, **kwargs: Any
    ) -> None:
        run = {"name": name, "inputs": inputs, "run_type": run_type, **kwargs}
        run["project"] = kwargs.get("session_name") or project_name
        with self._lock:
            self.runs.append(run)

    def update_run(self, run_id: Any, **kwargs: Any) -> None:
        with self._lock:
            for run in self.runs:
                if str(run.get("id")) == str(run_id):
                    run.update(kwargs)

    def create_feedback(self, run_id: Any = None, key: str = "unnamed", **kwargs: Any) -> None:
        with self._lock:
            self.feedback.append({"run_id": run_id, "key": key, **kwargs})

    def flush(self, timeout: float | None = None) -> None:
        return None

    def wait(self, runs: int = 1, timeout_s: float = 5.0) -> None:
        """Wait until at least ``runs`` Immune runs (not counting Jev child runs) have been recorded."""
        deadline = time.monotonic() + timeout_s
        while len(self.immune_runs()) < runs:
            if time.monotonic() > deadline:
                raise TimeoutError(f"expected {runs} Immune runs in LangSmith, got {len(self.immune_runs())}")
            time.sleep(0.01)

    def immune_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [run for run in self.runs if str(run["name"]).startswith("immune ·")]

    def children(self, run: dict[str, Any]) -> list[dict[str, Any]]:
        with self._lock:
            return [child for child in self.runs if str(child.get("parent_run_id")) == str(run["id"])]

    def feedback_on(self, run_id: Any, key: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            return [
                item
                for item in self.feedback
                if str(item["run_id"]) == str(run_id) and (key is None or item["key"] == key)
            ]
