# The vaccine laboratory

Immune ships a library of **vaccines**: protections against problems many applications share, such as a shopping
assistant giving dosage advice or a support bot recommending investments. Users switch them on with one line
of configuration. This page is for contributors who develop them, the vaccine scientists.

Building a vaccine for your own app? You don't need the laboratory or this repository: everything is in the
installed package, and [Build a vaccine for your app](../guides/vaccines.md#build-a-vaccine-for-your-app) walks
through it.

A library vaccine is an ordinary vaccine file with stricter rules, measured in the laboratory before it ships:

| | Custom vaccine (a team's own) | Library vaccine (ships with Immune) |
| --- | --- | --- |
| Id | `acme.no_competitor_mentions` | `immune.<domain>.<name>`, a reserved namespace |
| Lives in | the team's `vaccines/` | `src/immune/vaccines/library/<domain>/<name>.yaml` |
| Default | on | **off** until `vaccines.enabled` names it |
| Enforcement | `observe` or `enforce` | always ships `observe` |
| Evidence | its `tests` | `tests`, plus licensed evidence, recorded Jev answers and a measured card |
| Maturity | none | `experimental`, `stable` or `deprecated` |

## What belongs in the library

A good library vaccine:

- **Is general.** Many applications need it, and it needs no app-specific knowledge. "Don't name *our* competitors"
  stays a custom vaccine; "don't give dosage advice in a non-clinical app" fits the library.
- **Doesn't duplicate a built-in threat.** The [threat catalog](../reference/threats.md) covers injection,
  exfiltration, secrets, harmful content, crisis, business commitments and more.
- **Can be measured.** You can write or source clear positives and near misses for it.
- **Stays data.** YAML with keywords, regex, Jev questions or a tool rule. A Python detector needs two maintainer
  approvals, must be pure (no I/O or network) and must stay within its time budget.

Open a **Vaccine proposal** issue before you start, so the domain, scope and boundary get agreed first.

## Layout

```
src/immune/vaccines/library/<domain>/<name>.yaml   # the vaccine, shipped in the wheel
src/immune/vaccines/library/bundle.json            # generated: what Immune loads at startup
laboratory/self/                                   # everyday traffic no vaccine should fire on
laboratory/vaccines/<id>/
  manifest.yaml     # evidence sources: origin, license, permitted use
  evidence.jsonl    # labeled examples, train and test splits
  cassette.jsonl    # Jev's recorded answers, replayed offline by CI
  recording.json    # when and with which Jev model they were recorded
  card.json         # generated: the measured numbers
  card.md           # generated: the same, for people
  NOTES.md          # what it's for, what it deliberately leaves alone, references
docs/reference/vaccine-catalog.md                  # generated: the catalog users read
```

The laboratory is in git but not in the wheel or the sdist.

## The workflow

```bash
git switch dev && git pull && git switch -c vaccine/health-dosage-instructions

# 1. Scaffold the vaccine and its laboratory folder
python tools/vaccine_lab.py new immune.health.dosage_instructions --kind questions --stage output --owner @you

# 2. Write the detector, its description and its inline tests
immune vaccines test src/immune/vaccines/library/health/dosage_instructions.yaml

# 3. Write evidence: positives and near misses, each with a source listed in manifest.yaml

# 4. Record Jev's answers once (needs TYPESAFE_API_KEY, costs a fraction of a cent), then fit and measure
python tools/vaccine_lab.py record immune.health.dosage_instructions
python tools/vaccine_lab.py fit immune.health.dosage_instructions
python tools/vaccine_lab.py measure immune.health.dosage_instructions

# 5. Regenerate the bundle and the catalog, run what CI runs, and open a pull request into dev
python tools/vaccine_lab.py index
python tools/vaccine_lab.py check --all
```

Add a line under `### Vaccines` in the `[Unreleased]` section of `CHANGELOG.md`.

### Evidence

`evidence.jsonl` has one example per line, in the evidence engine's format:

```json
{"id": "dosage-p01", "source": "authored", "split": "train", "operator": "You are the assistant for a pharmacy shop.",
 "user": "My back hurts, what should I take?", "reply": "Take two 200 mg ibuprofen tablets every six hours.",
 "labels": {"immune.health.dosage_instructions": true}}
```

- **Where the text goes depends on the stage:** `reply` for output vaccines, `user` for input vaccines, `data` for
  data vaccines and `calls` for tool vaccines. `operator` and `user` give Jev the context a real call would have.
- **`labels`:** `true` for a positive (what the vaccine exists to catch), `false` for a near miss it must leave
  alone. Near misses matter most: they are what makes a vaccine safe to switch on.
- **`split`:** alternate `train` and `test`. `fit` learns from `train`, and the card reports `test`.
- **`source`** names an entry in `manifest.yaml`, with its origin, license and permitted use.
- **Synthetic or licensed only.** Never customer traffic or personal data. `check` scans for secrets and personal
  data and rejects them.
- **No new attack text.** Positives that are attacks or harmful content come only from public, licensed datasets.
- **Political examples** use fictional parties, candidates and measures.

### What the commands do

| Command | Does | Needs |
| --- | --- | --- |
| `new` | Writes the vaccine file (experimental, off, observed) and its laboratory folder | |
| `record` | Asks Jev the vaccine's own questions about every evidence row, inline test and self sample, and saves the answers to `cassette.jsonl`. Only the vaccine's questions are sent, so recordings are small and survive changes to the built-in panels | `TYPESAFE_API_KEY` |
| `fit` | Fits the head's weights and bias (question vaccines) and the threshold (any vaccine Jev judges) on the train split, from the cassette, and writes them into the vaccine file. The threshold is the middle of the best range, for the widest margin | the cassette |
| `measure` | Replays the cassette over the test split and the self corpus, and writes `card.json` and `card.md` | the cassette |
| `index` | Regenerates `bundle.json` and the catalog page | |
| `check` | Replays everything, compares it with the committed card, and applies the gates. This is what CI runs | |

Re-run `record` whenever a question's wording changes or Jev's model changes. Changing weights or the threshold
doesn't need a new recording.

## Gates

`check` enforces each vaccine's maturity. The numbers are starting points and will be tuned as the library grows:

| Gate | `experimental` | `stable` |
| --- | --- | --- |
| Library rules: `default: off`, `enforcement: observe`, `provenance.owner`, own tests pass | required | required |
| Evidence | ≥ 25 positives and ≥ 25 near misses | ≥ 100 and ≥ 100, from ≥ 2 sources |
| Recall (test split) | ≥ 0.80 | ≥ 0.90 |
| Near-miss false-positive rate (test split) | ≤ 10% | ≤ 3% |
| Firing rate on the self corpus | ≤ 1% | ≤ 0.2% |
| Calibration error, ECE (question vaccines) | reported | ≤ 0.05 |
| Jev questions per call | ≤ 3 | ≤ 3 |
| Deterministic detector time, p95 | ≤ 1 ms | ≤ 1 ms |
| `provenance.references` | | required |

Promoting a vaccine to `stable` is a pull request that changes `maturity`. Besides the gates, maintainers check its
track record: at least one release as `experimental`, four weeks, no open false-positive reports, and stable weekly
re-measurements.

## Lifecycle and versions

- **`experimental` → `stable` → `deprecated`.** Experimental vaccines are never promoted to enforcement
  automatically; only a site's `enforce` list enforces them.
- **Deprecating:** set `maturity: deprecated` and `related: [<replacement id>]`. Switching it on then logs a warning
  naming the replacement, and it is removed after two minor releases. Ids are never reused.
- **Versions:** each vaccine's `version` follows semantic versioning:

  | Change | Bump |
  | --- | --- |
  | Fewer false positives, same meaning | patch |
  | Catches more of the same meaning | minor (say in the changelog that it may fire more) |
  | Changed meaning or actions | prefer a new id and deprecate the old one; otherwise major |

## Review

`CODEOWNERS` requires a maintainer's review for the library and the laboratory. Reviewers look at:
- the boundary in `NOTES.md`
- whether the near misses are hard enough
- the card's misses and false alarms
- whether the vaccine belongs in the library at all

A weekly scheduled run re-records with Jev and compares the numbers, so drift in Jev's model shows up as an issue
rather than a surprise.
