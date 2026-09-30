# Immune showcase: the Bob's Burgers assistant platform

A realistic app with four LLM features, each protected by Immune with its own posture. It calls OpenAI for real
(`gpt-5.4-mini` by default), traces to LangSmith, and comes with a guided demo and a web inspector that show every
decision Immune makes. Everything also runs offline, with no keys.

| Feature | Site | What it does | Immune posture (`immune.yaml`) |
| --- | --- | --- | --- |
| Ordering assistant | `ordering` | Customer chat with tools: menu, orders, refunds, receipts, stock, loyalty points | User-facing; off-task and this app's vaccines enforced |
| Help center | `help-center` | Answers from `knowledge/*.md` with the Responses API; articles are marked untrusted | User-facing; grounding observed |
| Review triage | `review-triage` | Classifies reviews into a Pydantic schema | Schema echo enforced |
| Ops assistant | `ops-assistant` | Reads supply tickets and emails suppliers | Agent organ; only `supplier.example` addresses allowed |

## Start in five minutes

From this folder (`examples/showcase` in the Immune repository):

```bash
pip install -r requirements.txt
python -m showcase --offline demo        # the whole guided demo, no keys needed
```

To run it for real, put your keys in `.env` here or in the repository root (see `.env.example`):

```bash
python -m showcase demo                  # live: OpenAI + Jev
python -m showcase serve                 # the web inspector on http://127.0.0.1:8000
python -m showcase chat --site ordering  # chat in the terminal
python -m showcase report                # summary of runs/verdicts.jsonl
```

`python -m showcase demo --list` lists the chapters and `--only confirmations,streaming` runs a few.

## Budget guards

Live runs stop before they cost more than you allow:

- `DEMO_MAX_CALLS` (default 80) caps OpenAI calls per run.
- `DEMO_BUDGET_USD` (default 3.00) caps the total across every run of the showcase and the getting-started notebook.
  Spend is estimated from token usage at gpt-5.4-mini's list prices ($0.75 per million input tokens, $4.50 per
  million output tokens) and kept in `~/.immune/demo-spend.json`.
- Every call sets `reasoning_effort: none` and caps output tokens.

A full live demo made 32 OpenAI calls for about $0.015 at those prices, plus about 50,000 Jev tokens.

## The guided demo

| # | Chapter | What you see | Source |
| --- | --- | --- | --- |
| 1 | Before and after | Four conversations with Immune off, then on, and a policy quote Jev lets through | Live |
| 2 | Observe, then enforce | Observed hits, per-site enforcement, and promotion | Live and offline |
| 3 | Agents that ask first | A $250 refund waits for a sealed confirmation; "yes" releases it | Live |
| 4 | Poisoned knowledge | A poisoned article is neutralized before the model reads it | Live, with a library document |
| 5 | Where did that address come from? | An allowed supplier email, then a recorded exfiltration attempt held by destination provenance | Live and replay |
| 6 | Runaway tools | A flaky tool and a retrying model, stopped by the loop breaker | Live |
| 7 | Structured outputs you can trust | Schema-echo confidence per field | Live and replay |
| 8 | Streaming | A key redacted mid-stream | Live |
| 9 | Incident replays | Every incident in Immune's library, run as recorded | Replay |
| 10 | Vaccinate a gap | `immune vaccinate` builds a new protection from examples; it catches the next reply | Live |
| 11 | When Jev is down | Floor candidates act without Jev (`sensor.on_outage`) | Live OpenAI |
| 12 | Several workers | Shared state through SQLite | Offline |
| 13 | LangSmith | Threat runs nested in the app's traces | Live or recorder |
| 14 | Measure and train | The evidence engine trains detector heads | Offline |

Live chapters use ordinary questions. The real model and Jev decide, so a check that depends on their choice is
reported as "answered differently this time" instead of failing.

### How attacks are shown

This project contains no attack text. Attacks are replayed from Immune's scenario library (`../../scenarios`), which
reconstructs publicly reported incidents; the model's side of each attack is the recorded reply, served by a fake
provider through the same app code, so no real model is ever asked to misbehave. Incidents about harmful content or
personal crisis are shown as a verdict line only. To measure Immune against real attacks, run
`immune evidence collect` on a labeled dataset you supply.

## The web inspector

`python -m showcase serve` opens a two-pane page: chat with any feature on the left, and see the verdict for every
call on the right, with each threat's probability, whether it was enforced or observed, its floor rule, evidence, and
Jev's latency and tokens. It also switches Immune's mode live, makes the stock tool flaky, streams replies, replays
library incidents, shows recent verdicts, links to the trace report, and shows the budget meter.

## LangSmith

1. Create an API key at [smith.langchain.com](https://smith.langchain.com).
2. Add to `.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY=...`, `LANGSMITH_PROJECT=immune-showcase`.
3. Run `python -m showcase demo` or `serve`.

Every feature call is a traced function, OpenAI calls are wrapped with `langsmith.wrappers.wrap_openai`, and Immune's
own runs (`immune · confirm · bobs.refund_over_limit`) nest under the model call that caused them, with one child run
per Jev request and `immune.threat` / `immune.blocked` feedback. Clean calls are never sent. Useful views:

- feedback key `immune.threat`: every call with a possible threat
- feedback `immune.blocked = 1`: calls Immune changed
- tag `threat:bobs.*`: this app's vaccines

Without LangSmith keys, the same runs go to `immune.testing.LangSmithRecorder` and are written to
`runs/langsmith.html` (also at `/traces` in the inspector).

## Layout

```text
immune.yaml            four sites, vaccines, verdict log, LangSmith project
vaccines/              one vaccine per detector kind: keywords, regex, Jev question, tool rule, Python
knowledge/             help-center articles
data/                  reviews, tickets, and a corpus of replies for vaccine trials
scenarios/             this app's own scenarios for `immune replay`
showcase/
  app.py               composition root: Immune, OpenAI client, tracing, budget, features
  features.py          the four features
  llm.py               the only place that calls OpenAI (live, offline or replay), with the budget
  tools.py             fake tools that record instead of acting
  offline.py           the scripted model and Jev answers for --offline
  replay.py            library incident replays
  tracing.py           LangSmith and the local trace report
  demo.py              the guided demo
  web/                 the inspector
tests/                 offline tests: pytest
runs/                  created at run time: verdict log, replies, traces, state (gitignored)
```

## Reuse it

- Copy `immune.yaml` and `vaccines/` into your app and change the site names to your own `immune.site(...)` names.
- `showcase/llm.py` shows one way to put every model call behind a budget and a replay switch.
- `tests/test_showcase.py` shows how to test an app's real configuration offline.
