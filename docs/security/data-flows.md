# Data flows

| Data | Leaves the process? | Controls |
| --- | --- | --- |
| Masked operator-prompt template (profiling) | To TypeSafe, once per site version | Numbers, names and ids masked; `privacy.profiling: local` never sends it |
| User messages, data items, replies, tool calls (screening) | To TypeSafe | `privacy.redact_before_sensor` (on by default) masks personal data and secrets with typed placeholders |
| Vaccine questions | To TypeSafe, inside the same requests | Only the question text you wrote |
| Calls with possible threats (LangSmith tracing) | To your LangSmith project, only when `LANGSMITH_TRACING` and `LANGSMITH_API_KEY` are set (or `telemetry.langsmith.enabled: true`) | Clean calls are never sent; inputs masked by default (`inputs: none` omits them, `raw` also needs `privacy.log: full`) |
| OpenTelemetry spans and metrics | To your collector, when your app configures an SDK | Ids, threat names, actions and timings; no conversation text |
| Verdicts, labels, sites, statistics, calibration | Never, unless you export them | Stored in the state directory or your shared state backend |
| Anything to the Immune project | Never | No telemetry |
