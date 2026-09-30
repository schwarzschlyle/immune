# Immune's own threat model

Immune sits in the request path of every LLM call, so its own failure modes matter as much as the threats it
detects.

| Risk | Control |
| --- | --- |
| Immune breaks production traffic | Fail open by default (`on_internal_error: pass`), kill switch (`IMMUNE_DISABLED`), a circuit breaker and deadlines on Jev, fault-injection tests |
| Immune intercepts non-LLM traffic | Strict endpoint matching plus a body check per codec; everything else passes through unparsed |
| Log and verdict leakage | Redacted logs and evidence by default (`privacy.log`); the canary is never logged; the durable verdict log is opt-in |
| Configuration tampering | Schema validation, `yaml.safe_load`, and config and spec digests on every verdict |
| Silently weakened protection | Disabling a floor rule (F1–F10) fails at startup without `vaccines.allow_floor_changes: true`; a site `observe` list covering a floor rule is logged and shown by `immune doctor` and `immune vaccines list` |
| ReDoS in Tier 0 and vaccines | Bounded built-in patterns with an adversarial-input performance test; vaccine regular expressions are rejected at load if they repeat without a bound or nest repetitions |
| Malicious or buggy Python vaccines | Loaded only from configured `vaccines.paths` or installed `immune.vaccines` entry points; an exception skips the vaccine for that call; slow functions are logged. They run with the app's privileges, so review them like code (`CODEOWNERS` on `vaccines/`) |
| Attacks on the judge | Split-view screening and deterministic defanging |
| Talking Jev out of a floor candidate | Floor rules act on candidates Jev confirms ([ADR 0006](../adr/0006-code-nominates-jev-decides.md)), so text written to make a real find look harmless ("this key is only an example") targets that decision. Each candidate question sees the exact match marked in its excerpt, facts code computed (such as whether a key looks like a placeholder) and the operator's task, and hits keep both the reflex's finding and Jev's answer as evidence. Rule-based vaccines, or `confirm: jev` left off, keep a rule independent of Jev |
| Forcing a Jev outage to skip the floor | `sensor.on_outage: act_on_candidates` (the default) acts on floor candidates without Jev while it is unreachable, and the hit's evidence says so |
| Forged confirmations | Confirmation references are HMAC-sealed to the exact call and turn with a per-deployment secret |
| State tampering or disclosure | State files are owner-only (0600, directories 0700); shared state uses a key prefix per deployment |
| Data sent to third parties | Masking before Jev; LangSmith is off unless its environment variables are set, sends only calls with possible threats, and masks inputs by default ([data flows](data-flows.md)) |
| Package compromise | Trusted Publishing, build attestations, pinned actions, dependency review, minimal dependencies |
