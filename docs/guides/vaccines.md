# Vaccines

A vaccine teaches Immune one threat that is specific to your application: a competitor you never name, advice you
never give, an action that needs a person. It is a small YAML file (optionally backed by a Python function) that
compiles into the same spec as the 49 built-in threats, so it gets modes, per-site settings, promotion, verdicts,
OpenTelemetry and LangSmith for free.

This page walks through building one for your app, step by step, then documents every field and command.
[Project setup](setup.md#4-add-vaccines) shows where the files go in a project.

## Build a vaccine for your app

Everything on this page ships in `pip install immune-ai`. Your vaccines live in your own repository, next to your
code, and you don't need to contribute anything to Immune. (To add a vaccine to the library that ships *with* Immune,
see [the vaccine laboratory](../contributing/vaccine-laboratory.md) instead.)

The user guide's [Designing and testing your own vaccines](https://immune-user-guide.vercel.app/#/12b_designing_vaccines)
walks through one complete example with real outputs.

A vaccine answers four questions:

| Question | Field | Example |
| --- | --- | --- |
| Where does Immune look? | `stage` | `output`: the reply. Or `input`, `data` (documents, tool results) or `tool` (tool calls) |
| How does it recognize the problem? | `detect` | Keywords, a regular expression, questions for Jev, a tool rule or a Python function |
| What happens when it fires? | `respond` | Rewrite the reply with your `message`, hold a tool call, and so on |
| How do you know it works? | `tests` | Examples that must fire, and near misses that must not |

### 1. Create it

Start from a commented template:

```bash
immune vaccines new acme.no_competitor_mentions --kind keywords --stage output
# writes vaccines/acme.no_competitor_mentions.yaml for you to edit
```

Or build one from examples in a single step. `vaccinate` writes the file only if the vaccine passes its own examples,
and it runs a trial too:

```bash
immune vaccinate acme.no_competitor_mentions --stage output \
  --keyword "Burger Palace" --keyword McRival \
  --positive "You might prefer the Burger Palace deal." \
  --negative "Our Classic is \$9." \
  --message "I can only talk about Acme products."
```

Either way you end up with a file like this:

```yaml
id: acme.no_competitor_mentions
title: The reply recommends a competitor
description: The reply suggests the customer eat at a rival restaurant instead of Acme Burgers.
stage: output
detect:
  keywords: ["Burger Palace", "McRival"]
  confirm: jev                        # Jev confirms each match, so "we're next to Burger Palace" passes
respond:
  message: "I can only talk about Acme products."
tests:
  positives: ["You might prefer the Burger Palace deal."]
  negatives: ["Our Classic is $9."]
```

Use your own namespace for the id (your company or app name, such as `acme.`); `immune.` is reserved for the
library. The [file reference](#file-reference) below lists every field.

### 2. Turn it on

Point Immune at the folder, in `immune.yaml`:

```yaml
vaccines:
  paths: [vaccines/]
```

Or in code: `immune.init(vaccines=["vaccines/"])`. Every vaccine starts **observed**: Immune records its hits in
verdicts but changes nothing yet.

### 3. Test its examples

```bash
immune vaccines test                                          # every vaccine in vaccines.paths
immune vaccines test vaccines/acme.no_competitor_mentions.yaml
```

Each positive must fire and each negative must pass.
- **Keywords, regex, tool rules and Python functions** are checked for real, offline and without a key.
- **Jev questions and `confirm: jev`** need Jev to judge meaning. Offline, a scripted stand-in answers, which checks
  the wiring but not the meaning. In particular, a negative that contains a keyword "fails" offline because the
  stand-in confirms every match. Add `--live` (with `TYPESAFE_API_KEY` set) for the real check.

### 4. Trial it on everyday traffic

Tests show it catches what you meant; a trial shows it leaves everything else alone:

```bash
immune vaccines trial acme.no_competitor_mentions --corpus my_replies.jsonl --max-rate 0.01 --live
```

`--corpus` takes your own masked traffic, one message per line or as JSON Lines, and predicts false alarms best.
`--max-rate 0.01` exits with an error if more than 1% of the samples fire, which is handy in CI.

### 5. Test it in your own test suite

The `immune_harness` pytest fixture is registered when you install the package. It runs your real code against a
scripted model, with your vaccines loaded; [Testing and trials](#testing-and-trials) below has a complete example. To
keep CI offline for vaccines Jev judges, record Jev's answers once with `immune.testing.RecordingSensor` and replay
them with `ReplaySensor`, as the [developer guide](developer-guide.md#132-what-the-testing-kit-gives-you) shows.

### 6. Deploy it observed, then enforce it

Ship it observed and watch its hits in verdicts, logs, OpenTelemetry or LangSmith. When they look right, enforce it
at the sites where it matters (or set `enforcement: enforce` in the file):

```yaml
sites:
  ordering:
    enforce: [acme.no_competitor_mentions]
```

`immune vaccines list --custom` shows whether each of your vaccines is on, and which setting decided it.

### Your vaccines and the library

| | Your vaccines | The vaccine library |
| --- | --- | --- |
| Who writes them | You, for your app | Immune's contributors, for problems many apps share |
| Where they live | Your repository's `vaccines/` folder | Inside the `immune-ai` package |
| Ids | Your namespace, such as `acme.` | `immune.<domain>.<name>` |
| When they're on | As soon as `vaccines.paths` loads them (unless `default: off`) | Only when `vaccines.enabled` names them |
| How they're checked | `test`, `trial` and your own test suite | The same, plus the laboratory's measured cards and maturity gates |

You can also start from a library vaccine:
`immune vaccines fork immune.health.dosage_instructions --as acme.dosage_instructions` copies it into your
`vaccines/` folder to tailor. The laboratory's tool for recording, fitting and measured cards lives in Immune's
repository rather than the package. For your own vaccines, `test --live` and `trial --corpus` on your real traffic
answer the two questions that matter: does it catch what you meant, and does it leave everything else alone?

### When you need a key

| Detector | Without `TYPESAFE_API_KEY` | With it |
| --- | --- | --- |
| Keywords, regex, tool rules, Python | Fully tested and trialed offline | Same |
| Jev questions, `confirm: jev` | Wiring checked by a scripted stand-in | `--live` tests and trials judge meaning; recorded answers keep CI offline |

## File reference

```yaml
id: acme.no_competitor_mentions
version: 1.0.0
title: The reply names a competitor
description: Our assistant only discusses Acme products.
stage: output
invariant: U0
severity: medium
detect:
  keywords: ["Burger Palace", "McRival"]
respond:
  text: rewrite
  message: "I can only talk about Acme products."
applies_to:
  sites: ["ordering*"]
  organs: []
enforcement: observe
default: "on"
frameworks: []
tests:
  positives: ["You might prefer the Burger Palace deal."]
  negatives: ["Our Classic is $9."]
provenance:
  owner: growth-team
```

| Field | Required | Default | Meaning |
| --- | --- | --- | --- |
| `id` | Yes | | Namespaced, lowercase (`acme.no_competitor_mentions`). Built-in ids can't be reused |
| `version` | | `1.0.0` | Semantic version; recorded in LangSmith runs and the spec digest |
| `title` | Yes | | One line, shown in verdicts, listings and LangSmith |
| `description` | | | Longer explanation for reviewers |
| `stage` | Yes | | `input` (the user's message), `data` (tool results and documents), `tool` (tool calls) or `output` (the reply) |
| `invariant` | | `U0` | `U0` (custom) or one of the built-in invariants `U1`–`U10` |
| `severity` | | `medium` | `low`, `medium`, `high` or `critical` |
| `detect` | Yes | | Exactly one detector ([below](#detectors)); `detect.confirm: jev` has Jev confirm each match ([below](#asking-jev-to-confirm-a-match)) |
| `respond` | | Stage default | The action per sink and an optional `message` ([below](#responses)) |
| `applies_to.sites` | | Every site | Site name globs |
| `applies_to.organs` | | Every site | Only where one of these organs is active |
| `enforcement` | | `observe` | `observe` until promoted or enforced by configuration, or `enforce` in `auto` and `strict` modes |
| `default` | | `"on"` | `"off"` ships it switched off until `vaccines.enabled` lists it. Quote it: YAML reads a bare `on` or `off` as a boolean, which Immune also accepts |
| `frameworks` | | | Framework ids (such as OWASP `LLM01`) copied into each hit |
| `tests.positives` | | | Examples that must fire: text, or `{tool, arguments}` for tool vaccines |
| `tests.negatives` | | | Examples that must not fire |
| `provenance` | | | Free-form strings, such as an owner or a ticket |

Invalid files fail at startup, naming the file, the field and the fix.

## Detectors

### Keywords

```yaml
detect:
  keywords: ["Burger Palace", "McRival"]
```

Whole words, case-insensitive, compiled into one matcher. Evidence: `keyword 'Burger Palace'`.

### Regular expressions

```yaml
detect:
  regex: ['\bACC-\d{6}\b', '\bORD-\d{4,8}\b']
```

Patterns are checked at load time. They must use bounded repetition (`{0,200}` instead of `*`, `{1,200}` instead of
`+`, at most 1,000) and must not nest one repetition inside another, so no vaccine can hang on long text. Keywords and
regex can be combined in one vaccine.

### Jev questions

```yaml
detect:
  questions:
    - key: gives_dosage
      text: "The assistant output tells the user how much of a medicine to take."
    - key: topic
      kind: choice
      text: "What is the reply mainly about?"
      options: {dosage: A dose or schedule, hours: Opening hours, other: Something else}
      flag: [dosage]
  head:
    bias: 0.0
    weights: {gives_dosage: 2.0, topic: 1.0}
  threshold: 0.8
```

- **Questions** are yes/no statements (`kind: noul`) or choices (`kind: choice` with `options` and a `flag` listing
  the options that mean the threat is present). Keys are lowercase and unique within the vaccine.
- **Placeholders:** data questions must refer to the item as `{item}` and tool questions to the call as `{call}`,
  because they are asked once per item or call.
- **The head** is a logistic combination: `bias` plus a weight per question (default 1.0). A negative weight makes a
  "yes" count against the threat.
- **`threshold`** (default 0.8) is the probability at which the vaccine fires.
- Questions join the Jev request for their stage and site, so they add tokens but no round trip. Tool questions are
  asked for tool calls Immune screens with Jev: calls to tools with a declared or inferred capability, or any call
  after the session read untrusted content.

### Tool rules

```yaml
stage: tool
detect:
  tool: refund_*
  argument: {path: amount, greater_than: 100}
```

`tool` is a name or glob. `argument` is optional: `path` is a dotted path into the arguments (`customer.tier`,
`items.0.sku`), with one or more of `equals`, `one_of`, `contains` (case-insensitive), `greater_than` and `less_than`.
All given conditions must hold.

### Python functions

```yaml
detect:
  python: acme_immunity.vaccines:touches_vip_account
  budget_ms: 50
```

```python
from immune.vaccines import VaccineContext


def touches_vip_account(context: VaccineContext) -> str | None:
    account = str(context.arguments.get("account", ""))
    return f"account {account} is a VIP account" if account.startswith("vip-") else None
```

| Context field | Meaning |
| --- | --- |
| `vaccine` | The vaccine's id |
| `stage` | `Stage.INPUT`, `DATA`, `TOOL` or `OUTPUT` |
| `site` | The site id |
| `text` | The text for input, data and output vaccines |
| `tool`, `arguments` | The tool call for tool vaccines |
| `tainted` | Whether the session has read untrusted content |

Return `True`, evidence text or a list of evidence when the threat is present, and `None` or `False` otherwise. A
function that raises is skipped for that call with a warning; one slower than `budget_ms` is logged. It runs in the
request path, so keep it fast and free of side effects. The module must be importable where the app runs.

### Asking Jev to confirm a match

Keyword, regex, tool and Python vaccines are your own rules, so they decide on their own by default. Add
`confirm: jev` to let the rule only find candidates and have Jev decide whether each match is a real case:

```yaml
title: Don't recommend competitors
description: The reply suggests the customer buy from a rival restaurant.
detect:
  keywords: ["Burger Palace", "McRival"]
  confirm: jev
  threshold: 0.8
```

Jev is asked, for each match, whether it is a real case of the rule described by `title` and `description` rather
than an innocent use. "We're across the road from Burger Palace" then passes, while "try Burger Palace instead"
fires. `threshold` (default 0.8) applies to Jev's answer. The question joins the Jev request Immune already makes for
candidates, and while Jev is unreachable a confirmed vaccine does not fire (`sensor.on_outage` only covers the
built-in floor rules). Question detectors already ask Jev, so `confirm` doesn't apply to them.

## Responses

| Stage | Sink | Allowed actions | Default |
| --- | --- | --- | --- |
| `input` | `text` | `annotate`, `redirect`, `rewrite`, `refuse`, `handoff`, `end_session` | `redirect` |
| `input` | `software` | `annotate`, `refuse` | `refuse` |
| `data` | `data` | `annotate`, `neutralize` | `neutralize` |
| `tool` | `tool` | `annotate`, `confirm`, `hold`, `deny` | `hold` |
| `output` | `text` | `annotate`, `redirect`, `rewrite`, `refuse`, `handoff`, `end_session` | `rewrite` |
| `output` | `software` | `annotate`, `refuse` | `refuse` |

The *text* sink is a reply a person reads; *software* is a reply your code parses, such as a JSON extraction.
`message` replaces the default redirect or rewrite text, or becomes the reason in a tool note. On sites that talk to
users, `hold` becomes a confirmation request; use `deny` for actions the user must not be able to approve.

## Switching protections on and off

```yaml
vaccines:
  paths: [vaccines/]
  entry_points: true                            # also load the immune.vaccines entry point group
  disabled: ["output.claims_human", "acme.*"]   # built-in threats or vaccines, globs allowed
  enabled: ["acme.refund_over_limit"]           # re-enables, and switches on default: off vaccines
  allow_floor_changes: false                    # true is required to disable any F1–F10 rule
sites:
  kiosk:
    vaccines: {enabled: ["output.claims_human"], disabled: ["acme.upsell_*"]}
```

Site switches win over global ones, and `enabled` wins over `disabled` at the same level. The switches work live
with `immune.configure(vaccines={...})`; `vaccines.paths` needs `immune.init()` again. `immune vaccines list` shows
the outcome for a site and the setting that decided each one.

## Testing and trials

| Command | Checks |
| --- | --- |
| `immune vaccines test [paths] [--live] [--site]` | Positives fire and negatives don't, through a private runtime. Question vaccines use scripted answers offline (checking the head, stage and site wiring) and Jev with `--live`. Without paths it tests your own vaccines; library vaccines are measured in the laboratory |
| `immune vaccines trial <id or file> [--corpus FILE] [--max-rate R] [--live]` | How often the vaccine fires on everyday traffic: packaged samples for its stage plus your corpus. Works for library ids too. Question vaccines need `--live` |

Corpus files are JSON Lines (a string, `{"text": ...}` or `{"tool": ..., "arguments": {...}}` per line) or plain text
with one message per line. Masked samples of your own traffic predict false positives best.

To test a vaccine inside your own pytest suite, load it into the harness:

```python
import tempfile
from pathlib import Path

import yaml

from immune.testing import FakeReply

COMPETITOR = {
    "id": "acme.no_competitor_mentions",
    "title": "The reply names a competitor",
    "stage": "output",
    "detect": {"keywords": ["Burger Palace"]},
    "respond": {"message": "I can only talk about Acme products."},
    "enforcement": "enforce",
}


def test_the_competitor_vaccine_rewrites_the_reply(immune_harness):
    folder = Path(tempfile.mkdtemp())
    (folder / "competitors.yaml").write_text(yaml.safe_dump(COMPETITOR))
    harness = immune_harness(script=FakeReply(text="Try Burger Palace."), config={"vaccines": {"paths": [str(folder)]}})
    reply = harness.openai().chat.completions.create(model="m", messages=[{"role": "user", "content": "Ideas?"}])
    assert reply.choices[0].message.content == "I can only talk about Acme products."
    assert "acme.no_competitor_mentions" in harness.verdict().threats()
```

In your own suite, point `paths` at your `vaccines/` directory instead of a temporary one.

## The vaccine library

Immune ships vaccines for problems many applications share, written and measured in the project's
[laboratory](../contributing/vaccine-laboratory.md). The [vaccine catalog](../reference/vaccine-catalog.md) lists them
with their measured numbers. Every library vaccine:

- **Is off until you switch it on**, with the switches above, globally, per site or live with `immune.configure()`.
  Upgrading Immune never switches one on.
- **Costs nothing while off:** no Jev questions, no detector time and no promotion record.
- **Starts observed.** List it in a site's `enforce` to act on it. `stable` vaccines are also promoted automatically
  once their firing rate on your own traffic proves low; `experimental` ones never are.
- **Has an id in the reserved `immune.` namespace,** such as `immune.health.dosage_instructions`. Your own vaccines
  can't use it.

```yaml
vaccines:
  enabled: [immune.finance.personal_investment_advice]   # everywhere
sites:
  pharmacy-chat:
    vaccines: {enabled: [immune.health.*]}                # a whole domain, at one site
    enforce: [immune.health.dosage_instructions]
```

| Command | Does |
| --- | --- |
| `immune vaccines list --library [--site]` | Every library vaccine, its maturity, and whether it's on |
| `immune vaccines show <id>` | Its card: what it catches and leaves alone, recall, false-positive rates, cost, and how to switch it on |
| `immune vaccines fork <id> --as acme.<name>` | Copies it into your `vaccines/` as your own vaccine, to tailor |
| `immune vaccines trial <id> --corpus FILE --live` | How often it would fire on your traffic, before you switch it on |

`vaccines.library: false` leaves the library out entirely.

## Sharing vaccines

Package vaccine files and expose them through the `immune.vaccines` entry point group; every service with the
package installed loads them:

```toml
[project.entry-points."immune.vaccines"]
acme = "acme_immunity:vaccine_paths"
```

`vaccine_paths` returns a path or a list of paths (it may also be a plain path value). Set
`vaccines.entry_points: false` to ignore installed packages.

## Safety and performance

- Deterministic vaccines run in your process in microseconds; regex safety rules bound their worst case.
- Question vaccines add Jev tokens to their stage's request. `immune doctor` warns above 10 custom questions for
  one stage.
- Python vaccines run your code on every matching call. Review them like any other code; a `CODEOWNERS` entry for
  `vaccines/` helps.
- Nothing switches off the floor silently: disabling F1–F10 needs `allow_floor_changes: true`, and a site `observe`
  list that covers a floor rule is logged at startup and shown by `immune vaccines list`.
