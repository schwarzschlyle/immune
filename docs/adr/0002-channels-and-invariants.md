# ADR 0002: Channels and universal invariants

**Status:** accepted

## Context

Guardrails that scan text blobs cannot tell a user's request from an instruction hidden in a tool result, and per-use-case policies need configuration nobody writes.

## Decision

Model every call as operator, user, data, output and sink channels, and express defenses as ten invariants that hold for any application.

## Consequences

Defenses apply to any use case without configuration. Data pasted into prompts must be segmented heuristically; markers make it precise.
