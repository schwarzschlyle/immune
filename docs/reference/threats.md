# Threat catalog

Every built-in threat, how it is detected, whether it is enforced by default, and what your app receives when it is.
`immune threats` prints the same catalog, `immune explain <threat>` shows how one threat is detected, and
`immune vaccines list` adds the vaccines you have loaded and whether each protection is on.

Immune ships with 49 built-in threats, grouped by where they appear in a call. Vaccines add your own (see the [vaccines guide](../guides/vaccines.md)).

- **Check** says how each is detected. *Jev* means a calibrated detector built on Jev's answers. *Reflex → Jev*
  means a reflex in your process finds a candidate (a link, a key-shaped string, a copied passage, a repeated call) and
  Jev decides whether it is a threat, from a question about that candidate. While Jev is unreachable, floor candidates
  are acted on without it (`sensor.on_outage: act_on_candidates`, the default). The two *on Jev's answers* and *on
  accumulated risk* checks combine earlier Jev decisions.
- **By default** says whether it is enforced from the first call in `auto` mode, as part of the floor (F1–F10), or
  observed until it is promoted for a site or you enforce it.
- **Your app gets** describes the default response when it is enforced. Where a threat has different responses,
  *software* means the reply goes to code (for example a JSON extraction) rather than to a person.

The same catalog is available from the command line: `immune threats` and `immune explain <threat>`.

### Inbound text

| Threat | What it catches | Check | By default | Your app gets |
| --- | --- | --- | --- | --- |
| `inbound.smuggled_characters` | Invisible or look-alike Unicode characters used to hide text | Reflex → Jev | Enforced (F1) | The characters are removed and the call is re-sent; your app only gets the reply to the cleaned text |
| `inbound.hidden_content` | Hidden HTML in documents and tool results: comments, invisible styles | Reflex → Jev | Enforced (F1) | The hidden regions are stripped and the call is re-sent; your app only gets the reply to the cleaned data |
| `inbound.template_tokens` | Chat-template control tokens and fake transcripts pasted into text | Reflex → Jev | Observed | The safe redirect (software: a refusal); in data, the item is neutralized |
| `inbound.encoded_payload` | Base64, hex or percent-encoded blobs that decode to readable text | Reflex → Jev | Observed | A note in the verdict; it feeds the obfuscation detector |
| `inbound.guard_addressed` | Text aimed at a safety filter or grader, such as fake scores and verdicts | Reflex → Jev | Observed | A note in the verdict; it feeds the judge-manipulation detector |

### The user's message

| Threat | What it catches | Check | By default | Your app gets |
| --- | --- | --- | --- | --- |
| `input.override` | Attempts to change, cancel or replace the operator's rules or role | Jev | Observed | The safe redirect (software: a refusal) |
| `input.framing_bypass` | Stories, games or roleplay used to get content the assistant would refuse | Jev | Observed | The safe redirect (software: a refusal) |
| `input.extraction` | Requests to reveal the system prompt, configuration, tools or hidden rules | Jev | Observed | The safe redirect (software: a refusal) |
| `input.obfuscation` | Encoded, disguised or split-up text that hides its meaning | Jev | Observed | The safe redirect (software: a refusal) |
| `input.judge_manipulation` | Text aimed at a classifier, grader or reviewer rather than the assistant | Jev | Observed | The safe redirect (software: a refusal) |
| `input.off_task` | Requests unrelated to the task in the operator's instructions | Jev | Observed | The safe redirect (software: a refusal) |
| `input.resource_abuse` | Requests for extremely long, endless or bulk output | Jev | Observed | The safe redirect (software: a refusal) |
| `input.severe_harm` | Requests for mass-casualty weapons or sexual content involving minors | Jev | Enforced (F8) | The safe redirect; the reply never reaches your app |
| `input.harmful_request` | Requests for other harmful content: cyberattacks, fraud, hate, dangerous activities and more | Jev | Observed | The safe redirect (software: a refusal) |
| `input.crisis` | A user describing a personal crisis | Jev | Enforced (F9) on user-facing sites | Crisis resources appended to the reply |
| `care.minor_signals` | Signs the user is under 18 (care organ) | Jev | Observed | A note in the verdict |

### Outside data: documents and tool results

| Threat | What it catches | Check | By default | Your app gets |
| --- | --- | --- | --- | --- |
| `data.instructions` | Instructions addressed to an AI inside data | Jev | Enforced (F7) at high confidence | The item is replaced by a notice before the model reads it |
| `data.exfil_request` | Data asking for information to be sent out or placed in a link | Jev | Observed | The item is replaced by a notice |
| `data.impersonation` | Data claiming to come from the system, the developer, the operator or the user | Jev | Observed | The item is replaced by a notice |

### Tool calls

| Threat | What it catches | Check | By default | Your app gets |
| --- | --- | --- | --- | --- |
| `tool.destination_provenance` | A call sending to an address, host or URL that only untrusted content mentioned | Reflex → Jev | Enforced (F4) | The call is held; the model is told why |
| `tool.secret_egress` | An outgoing call whose arguments carry secrets or personal data | Reflex → Jev | Enforced (F5) | The call is held; the model is told why |
| `tool.loop` | The same call repeated too often, or too many calls in one turn | Reflex → Jev | Enforced (F6) | Extra calls are blocked; the model is told why |
| `tool.tainted_irreversible` | An irreversible action after untrusted content entered the session | Reflex → Jev | Enforced (F10) | The user is asked to confirm, with a sealed reference |
| `tool.dangerous_command` | Destructive shell or SQL commands, parsed with a real shell parser (coding organ) | Reflex → Jev | Observed | The call is held |
| `tool.description_poisoning` | MCP tool descriptions with hidden instructions, steering other tools, or changing after first use (agent organ) | Reflex → Jev | Observed | A note in the verdict |
| `tool.unrequested` | An action the user's request doesn't need | Jev | Observed | The user is asked to confirm |
| `tool.content_induced` | An action that follows instructions from data rather than from the user | Jev | Observed | The call is blocked; the model is told why |
| `tool.exceeds_request` | More items, recipients or scope than the user asked for | Jev | Observed | The user is asked to confirm |

### The reply

| Threat | What it catches | Check | By default | Your app gets |
| --- | --- | --- | --- | --- |
| `output.exfil_link` | Links or images that carry conversation data to a host nobody trusted | Reflex → Jev | Enforced (F2) | The link is replaced with `[link removed]` |
| `output.unsafe_markup` | Script-capable HTML in replies that a browser renders | Reflex → Jev | Enforced (F2) on rendered sites | The dangerous parts (scripts, event handlers, `javascript:` links) are removed |
| `output.secret_leak` | API keys, tokens and private keys in the reply | Reflex → Jev | Enforced (F3) | The secret is replaced with `[redacted]` |
| `output.canary_leak` | The deployment canary from the system prompt appearing in the reply | Reflex → Jev | Enforced (F3) | The canary is replaced with `[redacted]` |
| `output.prompt_copy` | A long run copied from the system prompt | Reflex → Jev | Enforced (F3) | The copied span is replaced with `[redacted]` |
| `output.task_deviation` | A reply that doesn't do the operator's task | Jev | Observed | The rewrite message (software: a refusal) |
| `output.follows_data_instructions` | A reply that follows instructions found in data | Jev | Observed | The rewrite message (software: a refusal) |
| `output.reveals_instructions` | A reply that reveals or paraphrases the system prompt | Jev | Observed | The rewrite message (software: a refusal) |
| `output.private_data` | Credentials or personal data about someone other than the user | Jev | Observed | The rewrite message (software: a refusal) |
| `output.severe_harm` | Mass-casualty weapons content or sexual content involving minors | Jev | Enforced (F8) | The rewrite message (software: a refusal) |
| `output.harmful` | Other harmful content | Jev | Observed | The rewrite message (software: a refusal) |
| `output.ungrounded` | Claims that the provided documents don't support, or contradict | Jev | Observed | The rewrite message (software: a note) |
| `output.false_action_claim` | Saying an action succeeded when no tool result shows it | Jev | Observed | The rewrite message |
| `output.claims_human` | Saying or implying the assistant is a person | Jev | Observed | The rewrite message |
| `output.solicits_secrets` | Asking the user for a password, one-time code, full card number or a login link | Jev | Observed | The rewrite message |
| `output.manipulation` | Guilt, false urgency or emotional pressure (care organ) | Jev | Observed | The rewrite message |
| `output.echo_disagreement` | A structured-output field that disagrees with Jev's independent reading (pipeline organ) | Reflex on Jev's answers | Observed | Software: a refusal; text: a note |
| `business.unauthorized_commitment` | Prices, discounts, refunds or guarantees the instructions don't back (business organ) | Jev | Observed | The rewrite message |
| `business.ungrounded_number` | Numbers such as prices and percentages that appear in no instruction or document (business organ) | Reflex → Jev | Observed | The rewrite message |
| `care.minor_unsafe` | Content unsuitable for someone under 18 (care organ) | Jev | Observed | The rewrite message |

### The session

| Threat | What it catches | Check | By default | Your app gets |
| --- | --- | --- | --- | --- |
| `session.adversarial` | Repeated attacks raising the session's risk above the threshold | Reflex on accumulated risk | Observed | The session is closed |

The messages your app receives are:

| Response | Text |
| --- | --- |
| Redirect | `I can't help with that here. Is there something else I can help you with?` |
| Rewrite | `I can't provide that response. Could you rephrase your request or ask about something else?` |
| Refusal | `This request can't be completed.` |
| Held tool call | `The action <call> was not carried out: <reason>.` |
| Blocked tool call | `The action <call> was blocked: <reason>.` |
| Confirmation | `Before I do that, please confirm: <call>. Reply "yes" to continue. (<reference>)` |
| Closed session | `This conversation has been closed.` |
| Crisis resources | `If you are in crisis or thinking about harming yourself, please contact local emergency services or a crisis line right away. In the US you can call or text 988.` |
| Neutralized data | `[Immune removed content from <origin> because it contained instructions addressed to the assistant. …]` |

*Organs* are groups of checks that switch on when a site's profile calls for them: **agent** (sites with tools),
**coding** (code assistants and tools that execute), **pipeline** (extraction, classification and summarizing),
**business** (assistants that speak for a business) and **care** (companions and apps minors may use). Switch one
on for a site with `sites.<site>.organs`.

## OWASP mapping

Each threat carries the OWASP ids it helps with (`frameworks` in `spec/threats.yaml`; `immune explain` prints them).
"Helps with" is deliberate: Immune reduces these risks at runtime, and none of them is solved by one control.

| OWASP risk | Threats | Floor rules |
| --- | --- | --- |
| LLM01:2025 Prompt Injection | `inbound.smuggled_characters`, `inbound.hidden_content`, `inbound.template_tokens`, `inbound.encoded_payload`, `inbound.guard_addressed`, `input.override`, `input.framing_bypass`, `input.obfuscation`, `input.judge_manipulation`, `input.severe_harm`, `input.harmful_request`, `data.instructions`, `data.exfil_request`, `data.impersonation`, `tool.content_induced`, `output.task_deviation`, `output.follows_data_instructions`, `output.echo_disagreement`, `session.adversarial` | F1, F7, F8 |
| LLM02:2025 Sensitive Information Disclosure | `data.exfil_request`, `tool.secret_egress`, `output.secret_leak`, `output.private_data` | F3, F5 |
| LLM05:2025 Improper Output Handling | `tool.dangerous_command`, `output.exfil_link`, `output.unsafe_markup` | F2 |
| LLM06:2025 Excessive Agency | `tool.destination_provenance`, `tool.tainted_irreversible`, `tool.unrequested`, `tool.exceeds_request`, `business.unauthorized_commitment` | F4, F10 |
| LLM07:2025 System Prompt Leakage | `input.extraction`, `output.canary_leak`, `output.prompt_copy`, `output.reveals_instructions` | F3 |
| LLM09:2025 Misinformation | `output.ungrounded`, `output.false_action_claim`, `business.unauthorized_commitment`, `business.ungrounded_number` | - |
| LLM10:2025 Unbounded Consumption | `input.off_task`, `input.resource_abuse`, `tool.loop` | F6 |
| ASI01 Agent Goal Hijack | `inbound.hidden_content`, `data.instructions`, `data.exfil_request`, `tool.destination_provenance`, `tool.content_induced`, `output.exfil_link`, `output.follows_data_instructions` | F1, F2, F4, F7 |
| ASI02 Tool Misuse and Exploitation | `tool.destination_provenance`, `tool.secret_egress`, `tool.tainted_irreversible`, `tool.unrequested` | F4, F5, F10 |
| ASI04 Agentic Supply Chain Vulnerabilities | `tool.description_poisoning` | - |
| ASI05 Unexpected Code Execution | `tool.dangerous_command` | - |
| ASI06 Memory and Context Poisoning | `data.instructions` | F7 |
| ASI08 Cascading Failures | `tool.loop` | F6 |
| ASI09 Human-Agent Trust Exploitation | `output.false_action_claim`, `output.claims_human`, `output.solicits_secrets`, `output.manipulation` | - |

Not covered at runtime: LLM03 Supply Chain, LLM04 Data and Model Poisoning and LLM08 Vector and Embedding Weaknesses
(build-time and data-pipeline risks), and ASI03 Identity and Privilege Abuse, ASI07 Insecure Inter-Agent
Communication and ASI10 Rogue Agents (identity and infrastructure controls). Beyond OWASP, Immune also covers duty of
care (`input.crisis`, F9, and the care organ), harmful output (`output.severe_harm`, F8, and `output.harmful`)
and business commitments.

Sources: [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/) and
[OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/).
