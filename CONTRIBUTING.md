# Contributing to Immune

Thank you for helping make LLM applications safer. This guide explains how the codebase is organized and what a
change needs before it can merge.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
just check          # lint, types, tests with coverage, scenario replay
```

No API keys are needed. Tests run against `immune.testing.FakeProvider` (a local stand-in for the OpenAI,
Anthropic and Gemini wire formats) and `MockSensor` (scripted Jev answers).

If you change a workflow in `.github/workflows/`, lint it with
[actionlint](https://github.com/rhysd/actionlint): `pipx run --spec actionlint-py actionlint`. It is not part of
the `dev` extra because installing it downloads a binary from GitHub, which made every CI test job depend on
GitHub's download servers.

## How the code is organized

| Package | Responsibility |
| --- | --- |
| `immune.spec` | Language-neutral data: threats, questions, heads, organs, endpoints, reflex patterns, templates |
| `immune.codecs` | Provider wire formats to channels and back |
| `immune.intercept` | Transport interception, routing, upstream adapters, `protect()` |
| `immune.core` | The call pipeline: segmentation, assessment, decisions, response composition |
| `immune.reflexes` | Tier 0 deterministic checks (pure functions, no I/O) |
| `immune.sensing` | Tier 1 Jev sensing, state building, panels, schema echo, sensor guards |
| `immune.heads` | Calibrated heads, calibration, promotion |
| `immune.profiling` | Call-site fingerprints, profiles, capabilities, posture |
| `immune.sessions` | Session state, taint, provenance memory, resolution |
| `immune.organs` | Use-case modules turned on by the profile |
| `immune.telemetry` | State store, verdict index and log, promotion ledger, labels, alerts, OpenTelemetry and LangSmith observers |
| `immune.vaccines` | Vaccine files: validation, detectors, compilation into the spec, switches, the test and trial lab |
| `immune.state` | State backends: local files, SQLite, Redis and failover |
| `immune.evidence` | Training heads from labeled examples: collection, fitting, model cards, drift |
| `immune.config` | Settings, loading and the spec model |
| `immune.cli` | The `immune` command line |
| `immune.testing` | The harness, fake providers, scripted sensors, scenarios, the pytest plugin and `LangSmithRecorder` |
| `immune.integrations` | Framework hooks that the HTTP patch can't reach (the Claude Agent SDK) |

Code should explain itself: descriptive names, small classes with one responsibility, and no comments that restate
the code. Public behavior belongs in the docs, not in comments.

## Branches and pull requests

Open pull requests against `dev`. Changes are promoted `dev` to `staging` to `main` by maintainers, and releases are
tagged on `main`; urgent fixes use a `hotfix/*` branch. Write commit messages in the
[Conventional Commits](https://www.conventionalcommits.org/) style (`feat:`, `fix:`, `docs:`, `chore:`), which makes
the next semantic version easy to see. [RELEASING.md](RELEASING.md) describes the whole flow.

## What every change needs

- **Tests.** Unit tests for new logic; an integration test through a real SDK client for new behavior on the wire.
- **Scenarios for new threats or questions.** Add at least three attack scenarios and two hard-benign twins to
  `scenarios/`. A new question never ships without evidence that it does not fire on benign traffic.
- **Spec changes stay backward compatible within a minor version.** Bump `src/immune/spec/version.txt` when the
  spec changes.
- **An ADR for architectural changes** (`docs/adr/`), especially anything that changes default enforcement or what
  data leaves the process.
- **A changelog entry** under `[Unreleased]`.

## Reporting false positives and missed attacks

Use the issue templates. A false positive report should include the call-site archetype, the verdict (`immune test`
output) and why the traffic is legitimate. A missed-attack report should include a minimal reproduction; if possible,
submit it as a scenario file.

## Security issues

Do not open public issues for vulnerabilities in Immune itself. See [SECURITY.md](SECURITY.md).
