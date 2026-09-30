# ADR 0006: Code nominates, Jev decides

**Status:** accepted (spec `2026.10.1`). Amends [ADR 0004](0004-fail-open.md) and [ADR 0005](0005-destination-provenance.md).

## Context

Seventeen threats, including seven of the ten floor rules, were decided by code: a reflex that found something also
declared it a threat. Code is exact about what it finds and blind to what it means. A customer who asked for the menu
got it redacted, because the menu was in the system prompt and F3 redacted any long passage copied from it. The same
blindness redacts AWS's documented example key and holds a call to an address the user did ask for, by reference.

## Decision

Reflexes and organs only **nominate candidates**. Each candidate becomes one yes/no Jev question about it (what was
found, where, and an excerpt with the match marked), asked in a separate request in parallel with the stage's other
questions. The threat's calibrated head turns the answer into a probability, and the threat fires only when it clears
the threshold. Actions are unchanged and still apply to the candidate's exact span or call.

Code keeps the **mechanics** Jev can't do: locating exact spans, parsing wire formats, commands and tool calls,
decoding invisible characters, counting calls across a session, remembering where destinations came from, and
matching the canary.

- **Outages.** `sensor.on_outage: act_on_candidates` (the default) acts on floor candidates without Jev while it is
  unreachable, with evidence saying so; `pass` lets them through. Candidates outside the floor always wait for Jev.
- **F1 stripping** uses the neutralization path: the call goes out while screening runs, and if Jev confirms, Immune
  cancels it and re-sends the cleaned text.
- **Streaming** releases text until the first candidate, then holds the rest until Jev decides at the end of the
  reply.
- **Rule-based vaccines** (keywords, regex, tool rules, Python) stay the developer's explicit rules and decide on
  their own; `detect.confirm: jev` opts one into the candidate path.
- **Offline**, `MockSensor` confirms candidate questions unless a test scripts a lower answer, so floor behaviour holds
  in tests and a test can model Jev dismissing a candidate.

## Consequences

- The floor no longer holds "even if every detector is fooled": a Jev miss is a floor miss. The floor now means
  "enforced from the first call", not "decided without a model".
- Jev can't verify that a key is real. On a small live check it rated keys used as real at 0.47–0.69 and documentation
  samples at 0.07–0.13, so the provisional `output.secret_leak` head places its cut near 0.4 until trained on labeled
  data.
- With live Jev, every floor candidate in the scenario library was confirmed. None of the 130 packaged everyday samples
  nominates a candidate, so ordinary traffic makes no extra Jev request.
- A reply or request with candidates makes one more Jev request, in parallel. In live runs that added about 600–700
  Jev tokens to the call.
