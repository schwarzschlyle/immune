"""Command line: python -m showcase {demo,chat,serve,triage,replay,report}."""

from __future__ import annotations

import argparse
import json
import sys

from showcase.app import Platform
from showcase.config import DATA, RUNS, Options
from showcase.console import Console


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m showcase", description="The Immune showcase platform")
    parser.add_argument("--offline", action="store_true", help="scripted model and Jev answers, no keys needed")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="the guided demo")
    demo.add_argument("--only", help="comma-separated chapter names (see --list)")
    demo.add_argument("--list", action="store_true", help="list the chapters")
    chat = commands.add_parser("chat", help="chat with one feature in the terminal")
    chat.add_argument("--site", choices=Platform.SITES, default="ordering")
    serve = commands.add_parser("serve", help="the web inspector")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--host", default="127.0.0.1")
    commands.add_parser("triage", help="classify data/reviews.jsonl")
    commands.add_parser("replay", help="run every incident in Immune's scenario library")
    commands.add_parser("report", help="summarize runs/verdicts.jsonl")
    args = parser.parse_args(argv)
    console = Console()

    if args.command == "report":
        from showcase.report import load, render, summarize

        for row in render(summarize(load())):
            console.line(row)
        return 0
    if args.command == "demo" and args.list:
        from showcase.demo import CHAPTERS

        for slug, title, _, _ in CHAPTERS:
            console.line(f"  {slug:<20} {title}")
        return 0

    platform = Platform(Options.detect(offline=args.offline))
    try:
        if args.command == "demo":
            from showcase.demo import run

            only = {name.strip() for name in args.only.split(",")} if args.only else None
            return run(platform, console, only)
        if args.command == "serve":
            from showcase.web.server import serve as serve_web

            serve_web(platform, args.host, args.port)
            return 0
        if args.command == "triage":
            for line in (DATA / "reviews.jsonl").read_text(encoding="utf-8").splitlines():
                review = json.loads(line)
                parsed, verdict = platform.triage.classify(review["text"])
                result = parsed.model_dump() if parsed else "refused"
                console.line(f"{review['id']}: {result}  [{verdict.action.value if verdict else '-'}]")
            return 0
        if args.command == "replay":
            from showcase.replay import Replays

            for replayed in Replays().run_all():
                console.line(f"{'pass' if replayed.passed else 'FAIL'}  {replayed.scenario:<30} {replayed.summary}")
            return 0
        return _chat(platform, console, args.site)
    finally:
        report = platform.tracer.write_report(RUNS / "langsmith.html")
        platform.close()
        if report is not None and args.command in ("demo", "chat"):
            console.line(console.paint(f"traces: {report}", "dim"))


def _chat(platform: Platform, console: Console, site: str) -> int:
    console.line(f"Chatting with {site} · {platform.options.describe()} · empty line to quit")
    session = f"cli-{site}"
    while True:
        try:
            message = input("you> ").strip()
        except EOFError:
            break
        if not message:
            break
        turn = platform.ask(site, session, message)
        console.field(site, turn.reply, "cyan")
        for event in turn.tools:
            console.field("tool ran", f"{event.tool}({event.arguments}) → {event.result}", "dim")
        for verdict in turn.verdicts:
            if verdict.hits:
                console.verdict(verdict)
        console.line(console.paint(f"    {platform.ledger.summary()}", "dim"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
