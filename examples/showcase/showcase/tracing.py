"""LangSmith tracing for the showcase.

- "on": LANGSMITH_TRACING and LANGSMITH_API_KEY are set. App functions, OpenAI calls and Immune's threat runs go to
  your LangSmith project (LANGSMITH_PROJECT, default immune-showcase).
- "recorder": no LangSmith account yet. The same runs go to immune.testing.LangSmithRecorder, and
  `write_report()` renders them as a local HTML page.
- "off": no tracing.
"""

from __future__ import annotations

import functools
import html
import importlib.util
import os
import warnings
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from immune.testing import LangSmithRecorder
from showcase.config import Tracing

F = TypeVar("F", bound=Callable[..., Any])


class Tracer:
    def __init__(self, mode: Tracing) -> None:
        self.mode: Tracing = mode if importlib.util.find_spec("langsmith") is not None else "off"
        self.project = os.environ.get("LANGSMITH_PROJECT", "immune-showcase")
        self.recorder = LangSmithRecorder().install() if self.mode == "recorder" else None
        self._langsmith: Any = None
        if self.mode != "off":
            import langsmith

            self._langsmith = langsmith

    @property
    def active(self) -> bool:
        return self.mode != "off"

    def traceable(self, name: str) -> Callable[[F], F]:
        def decorate(function: F) -> F:
            if not self.active:
                return function
            traced = self._langsmith.traceable(name=name, run_type="chain", client=self.recorder)(function)

            @functools.wraps(function)
            def call(*args: Any, **kwargs: Any) -> Any:
                with self._langsmith.tracing_context(enabled=True, client=self.recorder, project_name=self.project):
                    return traced(*args, **kwargs)

            return call  # type: ignore[return-value]

        return decorate

    def wrap(self, client: Any) -> Any:
        if not self.active:
            return client
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from langsmith.wrappers import wrap_openai

        # LangSmith serializes parsed structured outputs; pydantic warns about the parsed field's declared type.
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)

        extra = {"client": self.recorder} if self.recorder is not None else None
        return wrap_openai(client, tracing_extra=extra)  # type: ignore[arg-type]

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.uninstall()

    def write_report(self, path: Path) -> Path | None:
        """Render the recorded runs as a nested HTML page (recorder mode only)."""
        if self.recorder is None:
            return None
        runs = self.recorder.runs
        children: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in runs:
            children[str(run.get("parent_run_id"))].append(run)
        feedback: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in self.recorder.feedback:
            feedback[str(item["run_id"])].append(item)
        roots = [run for run in runs if run.get("parent_run_id") is None]
        body = "\n".join(
            _tree(run, children, feedback) for run in sorted(roots, key=lambda run: str(run["start_time"]))
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_PAGE.format(count=len(runs), body=body), encoding="utf-8")
        return path


def _tree(
    run: dict[str, Any], children: dict[str, list[dict[str, Any]]], feedback: dict[str, list[dict[str, Any]]]
) -> str:
    name = html.escape(str(run["name"]))
    immune_run = name.startswith("immune ·")
    tags = "".join(f"<span class=tag>{html.escape(str(tag))}</span>" for tag in run.get("tags") or [])
    notes = "".join(
        f"<div class=feedback>feedback {html.escape(item['key'])} = {html.escape(str(item.get('score')))}"
        f" {html.escape(str(item.get('value') or ''))}</div>"
        for item in feedback.get(str(run["id"]), [])
    )
    outputs = ""
    if immune_run:
        threats = ", ".join(threat["id"] for threat in (run.get("outputs") or {}).get("threats", []))
        outputs = f"<div class=threats>threats: {html.escape(threats)}</div>"
    nested = "".join(_tree(child, children, feedback) for child in children.get(str(run["id"]), []))
    kind = html.escape(str(run.get("run_type")))
    css = "run immune" if immune_run else "run"
    summary = f"<summary><b>{name}</b> <i>{kind}</i> {tags}</summary>"
    return f"<details open class='{css}'>{summary}{outputs}{notes}{nested}</details>"


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Showcase traces</title>
<style>
:root {{ --bg:#fbfbf9; --fg:#1d1d1b; --muted:#6b6b66; --accent:#b4441f; --card:#fff; --line:#e4e2dc; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#161614; --fg:#ecebe6; --muted:#a19f98; --accent:#f08a5d; --card:#1f1f1c; --line:#33322e; }}
}}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:980px;
  padding:24px 16px; }}
h1 {{ font-size:22px; }} p {{ color:var(--muted); }}
details {{ border-left:2px solid var(--line); margin:6px 0 6px 12px; padding-left:10px; }}
details.immune {{ border-left-color:var(--accent); }}
summary {{ cursor:pointer; }} i {{ color:var(--muted); }}
.tag {{ border:1px solid var(--line); border-radius:10px; font-size:12px; margin-left:4px; padding:0 6px; }}
.threats, .feedback {{ color:var(--accent); font-size:13px; margin:2px 0 2px 12px; }}
</style></head><body>
<h1>Showcase traces ({count} runs)</h1>
<p>Recorded by immune.testing.LangSmithRecorder. With LANGSMITH_TRACING and LANGSMITH_API_KEY set, the same runs go to
your LangSmith project instead. Immune's runs are marked in orange.</p>
{body}
</body></html>
"""
