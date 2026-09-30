# Anatomy of an LLM call

Codecs translate every provider's wire format into five channels.

| Channel | Examples | Trust | May it carry instructions? |
| --- | --- | --- | --- |
| Operator | System and developer prompts, tool definitions, response schemas | Trusted: defines the self | Yes, the only legitimate source of rules |
| User | User messages | Semi-trusted | Requests within the task, never rule changes |
| Data | Tool results, retrieved documents, web pages, emails, files, other agents' messages | Untrusted | No |
| Output | Model text, tool calls, structured output | Checked before release | — |
| Sinks | Rendering, tool execution, downstream code, memory | Where harm lands | — |

Data pasted into a prompt is found by `immune.untrusted()` markers, XML-style blocks such as `<document>`, labeled
sections such as `Context:`, and text that matches earlier data by content fingerprint.
