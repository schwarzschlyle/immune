"""FastAPI server for the inspector page. Run with `python -m showcase serve`."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

import immune
from immune.types import Mode, Verdict
from showcase.app import Platform
from showcase.budget import BudgetExceeded
from showcase.config import RUNS
from showcase.features import Turn
from showcase.replay import Replays

STATIC = Path(__file__).parent / "static"


class Message(BaseModel):
    site: str = "ordering"
    session: str = "web"
    message: str


class ModeChange(BaseModel):
    mode: Mode


class Toggle(BaseModel):
    on: bool


class ReplayRequest(BaseModel):
    scenario: str


class SessionRequest(BaseModel):
    session: str


def create_app(platform: Platform) -> FastAPI:
    app = FastAPI(title="Immune inspector", docs_url=None, redoc_url=None)
    replays = Replays()

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        runtime = immune.runtime()
        return {
            "options": {
                "live": platform.options.live,
                "jev": platform.options.jev,
                "model": platform.options.model,
                "tracing": platform.tracer.mode,
                "describe": platform.options.describe(),
            },
            "mode": runtime.settings.mode.value if runtime else "off",
            "sites": list(Platform.SITES),
            "replays": [{"id": scenario, "sensitive": scenario in _sensitive()} for scenario in replays.ids()],
            "ledger": platform.ledger.as_dict(),
            "flaky_stock": platform.ordering.tools.flaky_stock,
        }

    @app.post("/api/chat")
    def chat(request: Message) -> dict[str, Any]:
        _check_site(request.site)
        try:
            turn = platform.ask(request.site, request.session, request.message)
        except BudgetExceeded as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        return _turn(turn, platform)

    @app.post("/api/stream")
    async def stream(request: Message) -> StreamingResponse:
        if request.site == "help-center":
            raise HTTPException(status_code=400, detail="streaming is available for ordering and ops-assistant")
        _check_site(request.site)
        events: queue.Queue[tuple[str, Any] | None] = queue.Queue()

        def work() -> None:
            agent = platform.agent(request.site)
            try:
                for delta in agent.stream(platform.session(request.session), request.message):
                    events.put(("delta", delta))
                verdict = agent.llm.last_stream_verdict
                events.put(("verdict", _verdict(verdict) if verdict else None))
                events.put(("ledger", platform.ledger.as_dict()))
            except BudgetExceeded as error:
                events.put(("error", str(error)))
            finally:
                events.put(None)

        threading.Thread(target=work, name="inspector-stream", daemon=True).start()

        async def body() -> AsyncIterator[str]:
            while True:
                item = await asyncio.to_thread(events.get)
                if item is None:
                    return
                name, data = item
                yield f"event: {name}\ndata: {json.dumps(data)}\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    @app.post("/api/mode")
    def mode(request: ModeChange) -> dict[str, str]:
        immune.configure(mode=request.mode.value)
        return {"mode": request.mode.value}

    @app.post("/api/flaky-stock")
    def flaky(request: Toggle) -> dict[str, bool]:
        platform.ordering.tools.flaky_stock = request.on
        return {"flaky_stock": request.on}

    @app.post("/api/reset")
    def reset(request: SessionRequest) -> dict[str, str]:
        platform.reset(request.session)
        return {"session": request.session}

    @app.post("/api/replay")
    def replay(request: ReplayRequest) -> dict[str, Any]:
        if request.scenario not in replays.ids():
            raise HTTPException(status_code=404, detail=f"no scenario {request.scenario}")
        result = replays.run(request.scenario)
        return {
            "scenario": result.scenario,
            "title": result.title,
            "passed": result.passed,
            "summary": result.summary,
            "sensitive": result.sensitive,
        }

    @app.get("/api/verdicts")
    def verdicts() -> list[dict[str, Any]]:
        return [_verdict(verdict) for verdict in list(platform.verdicts)[-50:]][::-1]

    @app.get("/traces", response_class=HTMLResponse)
    def traces() -> HTMLResponse:
        path = platform.tracer.write_report(RUNS / "langsmith.html")
        if path is None:
            return HTMLResponse("<p>Traces are going to your LangSmith project.</p>")
        return HTMLResponse(path.read_text(encoding="utf-8"))

    return app


def serve(platform: Platform, host: str, port: int) -> None:
    import uvicorn

    print(f"Immune inspector on http://{host}:{port}  ·  {platform.options.describe()}")
    uvicorn.run(create_app(platform), host=host, port=port, log_level="warning")


def _check_site(site: str) -> None:
    if site not in Platform.SITES:
        raise HTTPException(status_code=400, detail=f"unknown site {site}")


def _sensitive() -> frozenset[str]:
    from showcase.replay import SENSITIVE

    return SENSITIVE


def _turn(turn: Turn, platform: Platform) -> dict[str, Any]:
    return {
        "site": turn.site,
        "reply": turn.reply,
        "verdicts": [_verdict(verdict) for verdict in turn.verdicts],
        "tools": [
            {"tool": event.tool, "arguments": dict(event.arguments), "result": event.result} for event in turn.tools
        ],
        "ledger": platform.ledger.as_dict(),
    }


def _verdict(verdict: Verdict) -> dict[str, Any]:
    data: dict[str, Any] = verdict.to_dict()
    data["blocked"] = verdict.blocked
    return data
