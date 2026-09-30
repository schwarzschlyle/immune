# Setting up Immune in your project

Immune needs one line of code. Everything else is optional files that live in your repository, next to your app:
settings, your own protections, test conversations, tests and CI. This guide shows where each file goes, how Immune
finds it, and a complete minimal version of each.

Every file below is in [`examples/starter/`](https://github.com/schwarzschlyle/immune/tree/main/examples/starter), a small
ordering assistant for a burger shop. The test suite checks that the starter works exactly as shown here.

Prefer to learn by running things? The
[getting-started notebook](https://github.com/schwarzschlyle/immune/blob/main/examples/notebooks/getting_started.ipynb)
builds the same files step by step on real OpenAI calls, and the
[showcase](https://github.com/schwarzschlyle/immune/tree/main/examples/showcase) is a larger app with a guided demo and a web
inspector.

## The layout

```text
your-app/
├── app.py                          # calls immune.init() at startup
├── requirements.txt
├── immune.yaml                     # Immune's settings (optional)
├── vaccines/                       # your own protections, one YAML file each (optional)
│   ├── bobs.no_competitor_mentions.yaml
│   └── bobs.refund_over_limit.yaml
├── scenarios/                      # conversations with expected outcomes, for `immune replay` (optional)
│   ├── bobs.menu-question.yaml
│   └── bobs.off-topic.yaml
├── tests/
│   ├── fixtures/replies.jsonl      # masked samples of real replies, for vaccine trials
│   └── test_immune.py              # pytest checks with the immune_harness fixture
├── .github/workflows/immune.yml    # CI on GitHub Actions
├── .gitlab-ci.yml                  # or GitLab CI
└── .pre-commit-config.yaml         # checks before each commit
```

| File | Read by | When | How Immune finds it |
| --- | --- | --- | --- |
| `immune.yaml` | `immune.init()` and the `immune` CLI | At startup | `./immune.yaml` in the directory the process starts in, or the file named by `IMMUNE_CONFIG`, or `immune.init(config=...)` |
| `vaccines/*.yaml` | `immune.init()` | At startup | `vaccines.paths` in `immune.yaml`, `immune.init(vaccines=[...])`, or an installed package's `immune.vaccines` entry point |
| `scenarios/*.yaml` | `immune replay` | In tests and CI | `./scenarios`, or files named on the command line |
| `tests/test_immune.py` | pytest | In tests and CI | The `immune_harness` fixture is registered when `immune-ai` is installed |
| CI and pre-commit files | Your CI and pre-commit | On every change | Their usual locations |

Only `app.py` is required. With no `immune.yaml`, Immune runs with its defaults: the floor is enforced and every
other threat is observed.

## 1. Install

Pin the version, so an upgrade is a deliberate change you can review.

```text
immune-ai==0.1.0
openai>=2.45
pytest>=8
```

Set the keys as environment variables, never in YAML or code: `TYPESAFE_API_KEY` for Jev (without it, only floor
candidates act, without Jev's judgment) and your model provider's key.

## 2. Start Immune in your app

Call `immune.init()` once, at startup, before your first LLM call. It patches the HTTP transports, so clients created
before or after it are covered.

```python
"""The ordering assistant for Bob's Burgers, protected by Immune."""

from openai import OpenAI

import immune

immune.init()  # reads ./immune.yaml (or IMMUNE_CONFIG) and loads vaccines/

SYSTEM_PROMPT = "You are the ordering assistant for Bob's Burgers. Help customers with the menu, orders and delivery."
client = OpenAI()


def answer(question: str, user_id: str) -> str:
    with immune.site("ordering"), immune.session(user_id):
        completion = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        )
    return completion.choices[0].message.content or ""


if __name__ == "__main__":
    print(answer("What's on the menu?", user_id="demo"))
```

`immune.site("ordering")` names this kind of call, so `immune.yaml` can give it its own settings.
`immune.session(user_id)` ties a user's turns together.

| App type | Where to call `immune.init()` |
| --- | --- |
| Script, CLI or notebook | At the top, after imports |
| FastAPI or Starlette | At module level in the app module, or in the `lifespan` handler before `yield` |
| Flask | In `create_app()`, or at module level in the app module |
| Django | In your app's `AppConfig.ready()` |
| Celery worker | In a `worker_process_init` signal handler |
| gunicorn with `--preload`, uWSGI | After the fork: gunicorn's `post_fork` hook, uWSGI's `lazy-apps = true` |
| AWS Lambda and other serverless | At module level, outside the handler, so warm invocations reuse it |

Immune starts background threads, and threads don't survive a fork. Servers that import your app once and then fork
workers need `immune.init()` in each worker, as in the last rows above.

## 3. Write `immune.yaml`

Put it in the directory your app starts from, usually the repository root next to `app.py`:

```yaml
# Immune settings for the ordering assistant. Immune reads ./immune.yaml from the directory the app starts in,
# or the file named by IMMUNE_CONFIG. Relative paths below are relative to this file.
mode: auto

vaccines:
  paths: [vaccines/]

sites:
  ordering:                          # immune.site("ordering") in app.py
    archetype: customer_service
    user_facing: true                # crisis resources and confirmations apply to people
    enforce:
      - input.off_task               # redirect requests unrelated to ordering
      - output.task_deviation
      - bobs.*                       # this app's vaccines
    allowed_destinations: [bobsburgers.example]
    tools:
      refund_order: {writes_state: true, irreversible: true}
```

- **Only set what you need.** Everything else keeps its default; the
  [configuration reference](configuration.md) lists every option.
- **Relative paths are relative to the file**, so `vaccines/` means the `vaccines` directory next to `immune.yaml`
  wherever the process starts. Paths passed in code (`immune.init(vaccines=["vaccines/"])`) are relative to the
  working directory.
- **Check it** with `immune config validate immune.yaml`. Unknown keys and bad values are reported with their path.
- **Editor completion:** run `immune config schema > immune.schema.json` and add
  `# yaml-language-server: $schema=./immune.schema.json` as the first line of `immune.yaml`. The VS Code YAML
  extension and JetBrains IDEs then complete and check every key.

### Different settings per environment

Keep one `immune.yaml` and change what differs with environment variables:

| Variable | Typical use |
| --- | --- |
| `IMMUNE_MODE=observe` | Staging, or a new deployment you are evaluating |
| `IMMUNE_STATE_URL=redis://cache:6379/0` | Production with several workers |
| `IMMUNE_CONFIG=/etc/immune/immune.yaml` | A separate file per environment, when the differences are large |
| `IMMUNE_DISABLED=1` | Switching Immune off without a deploy |

The same keys can also be passed in code: `immune.init(config="config/immune.production.yaml")` or
`immune.init(config={"mode": "observe"})`.

### In containers

Copy the files next to your app and start from that directory, or point `IMMUNE_CONFIG` at the file:

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py immune.yaml ./
COPY vaccines/ vaccines/
ENV IMMUNE_CONFIG=/app/immune.yaml
CMD ["python", "app.py"]
```

On Kubernetes, you can keep `immune.yaml` in a ConfigMap and mount it; vaccines can live in the same ConfigMap or
stay in the image:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ordering-immune
data:
  immune.yaml: |
    mode: auto
    vaccines:
      paths: [/app/vaccines]
    sites:
      ordering:
        user_facing: true
        enforce: [input.off_task, output.task_deviation, "bobs.*"]
```

Mount it at `/etc/immune` and set `IMMUNE_CONFIG=/etc/immune/immune.yaml`. Keep the Jev key in a Secret exposed as
`TYPESAFE_API_KEY`.

## 4. Add vaccines

A vaccine is one YAML file in `vaccines/`, named after its id. `immune vaccines new <id>` writes a commented template,
and `immune vaccinate` builds a tested one from examples.

`vaccines/bobs.no_competitor_mentions.yaml` rewrites any reply that names a competitor:

```yaml
id: bobs.no_competitor_mentions
version: 1.0.0
title: The reply names a competitor
stage: output
detect:
  keywords: ["Burger Palace", "McRival"]
respond:
  message: "I can only talk about Bob's Burgers products."
tests:
  positives: ["Honestly, Burger Palace has a better deal."]
  negatives: ["The Classic is $9 and comes with fries."]
```

`vaccines/bobs.refund_over_limit.yaml` asks the customer to confirm large refunds:

```yaml
id: bobs.refund_over_limit
version: 1.0.0
title: Refunds over $100 need the customer's confirmation
stage: tool
detect:
  tool: refund_order
  argument: {path: amount, greater_than: 100}
respond:
  tool: confirm
tests:
  positives: [{tool: refund_order, arguments: {order: 7, amount: 250}}]
  negatives: [{tool: refund_order, arguments: {order: 7, amount: 20}}]
```

Vaccines start observed. This app enforces them at the `ordering` site through `enforce: [bobs.*]` in
`immune.yaml`; a vaccine can also say `enforcement: enforce` itself. Check them with:

```bash
immune vaccines test                  # every file's positives fire and negatives don't
immune vaccines list --custom         # what is loaded, and whether it is on
```

The [vaccines guide](vaccines.md) covers every field and detector.

## 5. Add scenarios

A scenario is a conversation with a scripted model reply and the outcome you expect. `immune replay` runs every file
in `./scenarios`, offline:

```yaml
id: bobs.menu-question
title: A menu question passes untouched
kind: benign
operator: You are the ordering assistant for Bob's Burgers. Help customers with the menu, orders and delivery.
turns:
  - user: How much is the Classic burger?
    reply: {text: "The Classic is $9 and comes with fries."}
    expect:
      action: allow
      reply_contains: ["The Classic is $9"]
```

```yaml
id: bobs.off-topic
title: A request unrelated to ordering gets the safe redirect
kind: threat
mode: strict
operator: You are the ordering assistant for Bob's Burgers. Help customers with the menu, orders and delivery.
signals: {off_task: 0.95}
turns:
  - user: Write me a Python script that sorts a list of numbers.
    reply: {text: "Sure! print(sorted([3, 1, 2]))"}
    expect:
      enforced: [input.off_task]
      reply_excludes: ["print(sorted"]
```

`signals` script Jev's answers by question key (`immune explain <threat>` lists them), so a scenario tests your
settings without a key. Scenarios use their own `mode` and `config`, not `immune.yaml`; test the app's real settings
with pytest, below.

## 6. Write tests

The `immune_harness` fixture gives each test a private Immune runtime and SDK clients wired to a fake provider.
Passing `config=` the path of your `immune.yaml` tests the settings your app really runs with:

```python
"""Immune's checks for the ordering assistant, using the app's real immune.yaml. Run with: pytest"""

from pathlib import Path

import immune
from immune.testing import FakeReply, FakeToolCall, MockSensor

CONFIG = Path(__file__).resolve().parents[1] / "immune.yaml"
SYSTEM_PROMPT = "You are the ordering assistant for Bob's Burgers. Help customers with the menu, orders and delivery."
REFUND_TOOL = {"type": "function", "function": {"name": "refund_order", "description": "Refund an order"}}


def ask(harness, question, **options):
    with immune.site("ordering"):
        return harness.openai().chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            **options,
        )


def test_menu_questions_pass(immune_harness):
    harness = immune_harness(config=CONFIG, script=FakeReply(text="The Classic is $9."))
    assert ask(harness, "How much is the Classic?").choices[0].message.content == "The Classic is $9."


def test_off_topic_requests_are_redirected(immune_harness):
    harness = immune_harness(
        config=CONFIG, sensor=MockSensor({"off_task": 0.95}), script=FakeReply(text="Sure, here is a poem.")
    )
    reply = ask(harness, "Write me a poem about the sea")
    assert reply.choices[0].message.content.startswith("I can't help with that here.")


def test_competitor_mentions_are_rewritten(immune_harness):
    harness = immune_harness(config=CONFIG, script=FakeReply(text="Honestly, Burger Palace has a better deal."))
    reply = ask(harness, "Any deals this week?")
    assert reply.choices[0].message.content == "I can only talk about Bob's Burgers products."


def test_large_refunds_wait_for_the_customer(immune_harness):
    refund = FakeToolCall("refund_order", {"order": 7, "amount": 250})
    harness = immune_harness(config=CONFIG, script=FakeReply(tool_calls=[refund]))
    message = ask(harness, "Please refund order 7", tools=[REFUND_TOOL]).choices[0].message
    assert not message.tool_calls
    assert "please confirm: refund_order" in message.content
```

Trials use a corpus of your own traffic. Keep a small file of masked, real replies:

```text
{"text": "The Classic is $9 and comes with fries."}
{"text": "We deliver within 3 miles of the store until 10pm."}
{"text": "Your order is on its way and should arrive in about 30 minutes."}
{"text": "The Garden Stack is our vegetarian burger."}
{"text": "I've added a side of onion rings to your order."}
```

## 7. Run it in CI

GitHub Actions, `.github/workflows/immune.yml`: the offline checks on every pull request, and a weekly live check
with the Jev key as a repository secret:

```yaml
name: immune

on:
  pull_request:
  push:
    branches: [main]
  schedule:
    - cron: "0 6 * * 1"            # the live check, weekly
  workflow_dispatch: {}

permissions:
  contents: read

jobs:
  offline:                           # every pull request; no secrets needed
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.13"
      - run: pip install -r requirements.txt
      - run: immune config validate immune.yaml
      - run: immune vaccines test
      - run: immune vaccines trial bobs.no_competitor_mentions --corpus tests/fixtures/replies.jsonl --max-rate 0.01
      - run: immune replay
      - run: pytest

  live:                              # against the real Jev service
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    env:
      TYPESAFE_API_KEY: ${{ secrets.TYPESAFE_API_KEY }}
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: "3.13"
      - run: pip install -r requirements.txt
      - run: immune doctor --live
      - run: immune vaccines test --live
      - run: immune replay --live
```

GitLab CI, `.gitlab-ci.yml`:

```yaml
immune:
  image: python:3.13
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
  script:
    - pip install -r requirements.txt
    - immune config validate immune.yaml
    - immune vaccines test
    - immune vaccines trial bobs.no_competitor_mentions --corpus tests/fixtures/replies.jsonl --max-rate 0.01
    - immune replay
    - pytest
```

pre-commit, `.pre-commit-config.yaml`, so a broken setting or vaccine never reaches a pull request:

```yaml
repos:
  - repo: local
    hooks:
      - id: immune-config
        name: immune config validate
        entry: immune config validate immune.yaml
        language: system
        files: ^immune\.yaml$
        pass_filenames: false
      - id: immune-vaccines
        name: immune vaccines test
        entry: immune vaccines test
        language: system
        files: ^(vaccines/|immune\.yaml$)
        pass_filenames: false
```

The [developer guide](developer-guide.md#13-testing-and-ci) explains each check and how to record Jev's answers for
repeatable live tests.

## Checklist

```bash
immune config validate immune.yaml    # settings are valid
immune vaccines test                  # vaccines pass their examples
immune replay                         # scenarios pass
pytest                                # the app's own checks pass
immune doctor --live                  # the key works and interception is installed (needs TYPESAFE_API_KEY)
```
