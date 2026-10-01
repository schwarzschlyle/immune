# Configuration reference

Nothing is required beyond `TYPESAFE_API_KEY`. Settings come from, in increasing priority:

1. the defaults below
2. `immune.yaml` in the working directory, or the file named by `IMMUNE_CONFIG`
3. environment variables
4. arguments to `immune.init()`: `mode=`, `config=` (a path, a dict or a `Settings`), `state_dir=` and `vaccines=`
5. live changes with `immune.configure(...)`, merged into the running settings

Relative paths in a configuration file are relative to that file; paths passed in code or environment variables are
relative to the working directory. [Project setup](setup.md) shows where the file goes and a minimal example.

`immune config validate` checks a file and names any unknown or invalid key. `immune config schema` prints the JSON
Schema (also in [`schema/immune.schema.json`](https://github.com/schwarzschlyle/immune/blob/main/schema/immune.schema.json))
for editor completion.

## Environment variables

| Variable | Effect |
| --- | --- |
| `TYPESAFE_API_KEY` | Jev credentials |
| `TYPESAFE_BASE_URL` | Alternative Jev endpoint; Immune excludes it from interception |
| `IMMUNE_MODE` | `auto`, `observe`, `strict` or `off` |
| `IMMUNE_CONFIG` | Path to the configuration file |
| `IMMUNE_HOME` | State directory (default `~/.immune`) |
| `IMMUNE_STATE_URL` | Shared state: `redis://…`, `rediss://…` or `unix://…` for Redis, `sqlite:///path` for SQLite |
| `IMMUNE_DISABLED` | `1` makes `immune.init()` do nothing |
| `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` | LangSmith tracing of possible threats (`telemetry.langsmith.enabled: auto`) |

## Every setting

Values are the defaults unless a comment says otherwise. The `endpoints` entry and the `ordering` site are examples.

```yaml
mode: auto                      # auto | observe | strict | off
on_block: respond               # respond (a safe reply) | raise (immune.Blocked)
on_internal_error: pass         # pass (fail open) | block (refuse)
streaming: progressive          # progressive | buffered
canary: true                    # add a reference token to system prompts
state_dir: null                 # default: IMMUNE_HOME or ~/.immune
endpoints:                      # extra LLM routes, <codec>=<path regex> (default: none)
  - openai_chat=^/llm/complete$
sensor:
  model: jev-1.13.0
  timeout_ms: 800
  deadline_margin_ms: 250
  api_key: null                 # prefer the TYPESAFE_API_KEY environment variable
  api_keys: []                  # several keys to spread the load
  requests_per_minute: 1200
  coalesce: false
  breaker_error_rate: 0.2
  breaker_window_s: 60
  breaker_cooldown_s: 30
  on_outage: act_on_candidates  # act_on_candidates | pass: floor candidates while Jev is unreachable
privacy:
  redact_before_sensor: true
  log: redacted                 # off | redacted | full
  profiling: jev                # jev | local
  verdict_log: null             # path of a JSONL audit log
promotion:
  enabled: true
  max_firing_rate: 0.001
  min_calls: 5000
  min_days: 14
limits:
  max_identical_tool_calls: 3
  max_tool_calls_per_turn: 25
  session_ttl_s: 86400
  max_sessions: 50000
state:
  backend: local                # local | sqlite | redis
  path: null                    # the SQLite file
  url: null                     # redis://host:6379/0
  key_prefix: "immune:"
  failover_cooldown_s: 30
telemetry:
  opentelemetry: true
  alerts:
    enabled: true
    window_s: 900
    ratio: 5.0
    min_fired: 5
    absolute_rate: 0.05
    cooldown_s: 3600
  langsmith:
    enabled: auto                # auto (on with LANGSMITH_TRACING + LANGSMITH_API_KEY) | true | false
    project: null                # default: LANGSMITH_PROJECT
    inputs: masked               # masked | none | raw (raw also needs privacy.log: full)
    jev_runs: true               # a child run per Jev request
    feedback: true               # immune.blocked and immune.threat feedback
    mark_blocked_as_error: false
heads:
  artifact: null                # written by `immune evidence fit`
vaccines:
  paths: []                     # vaccine files or directories, e.g. [vaccines/]
  entry_points: true            # also load the immune.vaccines entry point group
  disabled: []                  # threat ids or globs to switch off, e.g. [output.claims_human]
  enabled: []                   # vaccines that declare default: off, e.g. library vaccines [immune.health.*]
  allow_floor_changes: false    # must be true to disable any F1–F10 protection
  library: true                 # the vaccine library that ships with Immune (all off until enabled); false for none
sites:                          # per call site, named with immune.site("ordering") (default: none)
  ordering:
    enabled: true
    archetype: customer_service
    user_facing: true
    organs: [business]
    enforce: [input.off_task, output.task_deviation]
    observe: []
    streaming: progressive
    allowed_destinations: [bobsburgers.example]
    echo:
      enforce: false
      min_agreement: 0.2
    vaccines:                   # site switches win over the global ones
      disabled: []
      enabled: []
    tools:
      place_order:
        egress: false
        writes_state: true
        executes: false
        reads_private: false
        irreversible: false
        allowed_destinations: []
```

## What can change at runtime

`immune.configure()` applies from the next call. `state`, `sensor`, `state_dir`, `canary` and `vaccines.paths` decide
how Immune is built, so changing them needs `immune.init()` again; `immune.configure()` raises a `ConfigError` naming
them.

```python
import immune

immune.configure(mode="observe")
immune.configure(sites={"ordering": {"enforce": ["input.off_task"]}})
immune.configure(vaccines={"disabled": ["output.claims_human"]})
```

## Guides by topic

- Modes and per-site postures: [developer guide, section 6](developer-guide.md#6-modes-and-sites-a-posture-for-each-part-of-your-app)
- Every option explained by concern: [developer guide, section 7](developer-guide.md#7-customizing-immune)
- Switching protections and adding your own: [vaccines](vaccines.md)
