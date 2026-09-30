# The reflex arc

| Tier | When | Latency | Contents |
| --- | --- | --- | --- |
| 0: spinal | Every call, inline | < 5 ms | Reflexes that find candidates |
| 1: innate | Every call; input and data run alongside the LLM call, output and tools after it | ~100–300 ms | Jev questions (one per candidate, plus each stage's panel), calibrated heads, split view, schema echo |

Three rules govern every call:

1. **Nothing leaves before its verdict.** Input and data screening race the LLM call because Immune holds the response
   until every verdict is in. A flagged data item cancels the upstream call and restarts it with the item neutralized.
2. **Prefix-stable changes.** Every change Immune makes to a request is deterministic and identical on later turns, so
   prompt caching keeps working.
3. **Fail open on internal errors.** Any exception inside Immune passes the original traffic through and logs a
   warning. `on_internal_error: block` switches to refusing instead.

Every Jev answer is a measurement. Each threat is a small logistic head over several atomic questions and confirmed
candidates, calibrated per deployment from labels (`immune.feedback`, `immune calibrate`).

The spinal tier decides nothing on its own. A reflex that finds something (a link carrying data, a key, a copied
passage, a repeated call) hands Jev a *candidate*, and the candidate's head decides from Jev's answer
([ADR 0006](../adr/0006-code-nominates-jev-decides.md)). While Jev is unreachable, `sensor.on_outage` decides what
floor candidates do.
