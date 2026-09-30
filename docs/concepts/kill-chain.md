# Invariants and the kill chain

Ten invariants hold for every LLM application:

| # | Invariant |
| --- | --- |
| U1 | Rule provenance: rules come only from the operator |
| U2 | Task fidelity: output serves the operator's task and the user's request |
| U3 | Action intent: every action traces to the user's request or the operator's task |
| U4 | Confidentiality: secrets, credentials and private data stay inside |
| U5 | Sink safety: output is safe for where it lands |
| U6 | Harm floor and duty of care |
| U7 | Grounding: claims about provided context are supported by it |
| U8 | Honesty about self and actions |
| U9 | Proportional resources |
| U10 | Homeostasis |

Every attack needs every link of the chain: **delivery → hijack → action or exfiltration → persistence → impact**.
Immune places independent checks on each link. The reflexes on action and exfiltration find candidates precisely
(an off-list destination, a link carrying data, a key in outgoing arguments), and a targeted Jev question about each one
decides, independent of whether the judgments upstream caught the hijack.

| Floor | Reflex |
| --- | --- |
| F1 | Strip smuggled and invisible characters and hidden HTML from inbound text |
| F2 | Remove exfiltration links and unsafe markup from rendered output |
| F3 | Redact secrets, the canary and copied operator-prompt spans |
| F4 | Destination provenance: hold egress to destinations only untrusted data mentioned |
| F5 | Hold egress calls carrying secrets or personal data |
| F6 | Break tool-call loops |
| F7 | Neutralize data items carrying instructions to the assistant (high confidence, both views agree) |
| F8 | Block severe harm |
| F9 | Append crisis resources on user-facing call sites |
| F10 | Confirm irreversible tool calls after untrusted data has been read |
