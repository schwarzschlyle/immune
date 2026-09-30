# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). While the version is `0.x`, minor versions may include breaking changes;
each one is called out under **Changed** or **Removed**.

## [Unreleased]

### Changed

- The user guide is published at https://immune-user-guide.vercel.app/, and the package's Documentation link points there.

### Fixed

- The Redis state backend no longer fails an update with "could not update ... after 20 attempts" when several
  workers write the same key at once; it now waits a short, random, growing moment between retries.

## [0.1.0] - 2026-09-30

First public release. Immune is pre-release software: the API may change between minor versions, and the detector
weights are provisional until they are trained on labeled data.

### Added

- **Interception:** `immune.init()` protects every LLM call in the process at the `httpx` and `httpx2` transports,
  with adapters for google-genai's aiohttp mode, the OpenAI aiohttp client and DSPy, and hooks for boto3 Bedrock
  Converse and ConverseStream. `immune.protect(client)` protects one client and keeps its proxies, timeouts, limits
  and event hooks; `immune.protected()` protects a block. `immune.coverage()` reports what is intercepted.
- **Providers:** OpenAI Chat Completions and Responses (including `previous_response_id` chains), Anthropic
  Messages, Gemini, Azure OpenAI, Vertex, Bedrock Converse and OpenAI-compatible servers, streaming and not.
  Frameworks built on these SDKs (LangChain, LangGraph, LlamaIndex, LiteLLM, DSPy and others) are covered, and the
  Claude Agent SDK through its hooks.
- **Channels:** each call is read as operator, user, data, output and sinks. Tool results are data automatically;
  `immune.untrusted(text, source=...)` marks pasted content.
- **Code nominates, Jev decides** ([ADR 0006](docs/adr/0006-code-nominates-jev-decides.md)): reflexes find exact
  candidates (smuggled characters, hidden markup, exfiltration links, unsafe markup, secrets, the canary, copied
  prompts, off-list destinations, loops, dangerous commands, poisoned tool descriptions), and TypeSafe Jev decides
  each one with a targeted question and a calibrated head. The Jev detectors add split-view screening and schema echo
  for structured outputs.
- **The floor** (F1–F10), enforced from the first call in `auto` mode; every other detector starts observed and is
  promoted per site once its firing rate is provably low. Modes `auto`, `observe`, `strict` and `off`.
- **Sites and sessions:** self-profiling per call site with organs (agent, coding, pipeline, business, care) and
  posture checks; `immune.site()` and `immune.session()`; session taint, provenance memory across calls, session
  risk, and confirmations sealed to the exact call and conversation.
- **Verdicts:** `immune.verdict()` for responses, stream chunks, LangChain messages and trace ids;
  `immune.on_verdict()`, `immune.feedback()` and per-site calibration from labels.
- **Streaming:** progressive release that holds text from the first candidate until Jev decides, falling back to
  buffered where a whole-reply check could enforce.
- **Vaccines:** custom protections as YAML files with keyword, bounded-regex, Jev-question, tool-rule and Python
  detectors, optional `confirm: jev`, custom messages, site and organ scoping, `enforcement: enforce` and
  `default: off`, loaded from `vaccines.paths`, `immune.init(vaccines=...)` or the `immune.vaccines` entry point
  group. Switches turn any built-in threat or vaccine off globally, per site or live; disabling a floor rule needs
  `vaccines.allow_floor_changes: true`.
- **Operations:** shared state on SQLite or Redis with failover to memory (`IMMUNE_STATE_URL`); a Jev quota governor
  with key pools, priorities and load shedding; deadlines, a circuit breaker and `sensor.on_outage`; fail-open
  internal errors; `immune.configure()` for live changes and per-site kill switches.
- **Observability:** OpenTelemetry spans and metrics, spike alerts with `immune.on_alert()`, privacy log modes, an
  opt-in JSON Lines verdict log, and LangSmith runs for calls with possible threats only.
- **Command line:** `immune doctor`, `test`, `replay`, `threats`, `explain`, `status`, `posture`, `promote`, `label`,
  `calibrate`, `evidence`, `vaccines`, `vaccinate`, `export`, `init`, `config` and `version`.
- **Testing:** `immune.testing` with `ImmuneHarness`, a fake provider, `MockSensor`, record and replay cassettes, a
  Jev wire stub, a LangSmith recorder, a pytest plugin and a library of reconstructed incidents that ships in the
  wheel.
- **Evidence engine:** `immune evidence collect|fit|drift|study|questions`, heads artifacts pinned to the spec version
  and Jev model, and model cards.
- **Documentation:** an 18-notebook user guide, a developer guide, configuration and threat references, and runnable
  examples: a starter project, a showcase app, a getting-started notebook and a feature tour.

### Security

- Vaccine regular expressions are rejected at load when they repeat without a bound or nest repetitions.
- Personal data and secrets are masked before anything is sent to Jev; state files are owner-only.
- `SECURITY.md` sets response targets: acknowledgement in 2 business days, triage in 5, critical fixes in 14 days.

[Unreleased]: https://github.com/schwarzschlyle/immune/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/schwarzschlyle/immune/releases/tag/v0.1.0
