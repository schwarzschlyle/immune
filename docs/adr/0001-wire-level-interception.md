# ADR 0001: Wire-level interception

**Status:** accepted

## Context

Integrating per framework does not scale. A typical app has many LLM call sites across several SDKs and frameworks,
and the call sites nobody configured are the ones attackers reach.

## Decision

Intercept at the `httpx2` and `httpx` transport layer, which the OpenAI, Anthropic and Google GenAI SDKs share. Decode
known endpoints with codecs and synthesize replacement responses as provider JSON, so every SDK parses them into its
own types. Keep `immune.protect(client)` as the explicit alternative, and provide hooks for runtimes that bypass HTTP
in the current process (the Claude Agent SDK).

## Consequences

One patch point covers every framework and agent loop. Immune depends on the transport interfaces of two HTTP
libraries; this is mitigated by version guards, fail-open behavior and the `IMMUNE_DISABLED` kill switch.
