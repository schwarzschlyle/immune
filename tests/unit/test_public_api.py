from __future__ import annotations

import importlib
import json
import pkgutil
from pathlib import Path

import pytest
from tools.api_surface import ApiDifference, ApiSurface

import immune

SNAPSHOT = Path(__file__).resolve().parents[2] / "api" / "public-api.json"


def test_public_functions_survive_importing_every_submodule() -> None:
    for module in pkgutil.walk_packages(immune.__path__, "immune."):
        if not module.name.startswith(("immune.cli.main", "immune.testing.pytest_plugin")):
            importlib.import_module(module.name)
    for name in ("session", "runtime", "site", "status", "verdict", "init", "coverage"):
        assert callable(getattr(immune, name)), name


def test_unknown_attributes_raise() -> None:
    missing = "definitely_not_public"
    with pytest.raises(AttributeError):
        getattr(immune, missing)


def test_dir_lists_the_lazy_api() -> None:
    assert {"init", "protect", "verdict"} <= set(dir(immune))


def test_the_public_api_matches_the_snapshot() -> None:
    difference = ApiDifference(json.loads(SNAPSHOT.read_text(encoding="utf-8")), ApiSurface().snapshot())
    assert difference.empty, "\n".join(difference.lines())


def test_verdicts_are_found_by_the_objects_frameworks_return() -> None:
    from types import SimpleNamespace

    from immune.telemetry.verdicts import VerdictIndex
    from immune.types import Action, Verdict

    index = VerdictIndex()
    verdict = Verdict(trace_id="t1", site="s", session_id=None, action=Action.ALLOW, would_action=Action.ALLOW)
    index.add(verdict, "chatcmpl-123")
    assert index.lookup(SimpleNamespace(id="chatcmpl-123")) is verdict
    langchain_message = SimpleNamespace(id="lc_run--1", response_metadata={"id": "chatcmpl-123"})
    assert index.lookup(langchain_message) is verdict
    assert index.lookup("t1") is verdict


def _verdict(trace_id: str) -> immune.Verdict:
    return immune.Verdict(
        trace_id=trace_id, site="s", session_id=None, action=immune.Action.ALLOW, would_action=immune.Action.ALLOW
    )


def test_site_blocks_hand_back_verdicts_recorded_in_copied_contexts() -> None:
    import contextvars

    from immune.telemetry.verdicts import VerdictIndex

    index = VerdictIndex()
    VerdictIndex.forget_last()
    with immune.site("support-chat"):
        contextvars.copy_context().run(index.add, _verdict("copied"))  # what LangChain does for each step
        assert VerdictIndex.last() is None  # unchanged inside the block
    last = VerdictIndex.last()
    assert last is not None
    assert last.trace_id == "copied"
    with immune.site("outer"), immune.session("inner"):
        contextvars.copy_context().run(index.add, _verdict("nested"))
    last = VerdictIndex.last()
    assert last is not None
    assert last.trace_id == "nested"


async def test_tasks_inside_a_site_block_keep_their_own_verdicts() -> None:
    import asyncio

    from immune.telemetry.verdicts import VerdictIndex

    index = VerdictIndex()

    async def call(trace_id: str, delay: float) -> str | None:
        await asyncio.sleep(delay)
        index.add(_verdict(trace_id))
        await asyncio.sleep(0.02)
        seen = VerdictIndex.last()
        return seen.trace_id if seen else None

    with immune.site("support-chat"):
        results = await asyncio.gather(call("a", 0.0), call("b", 0.01))
    assert results == ["a", "b"]
