# ADR 0005: Destination provenance

**Status:** accepted; amended by [ADR 0006](0006-code-nominates-jev-decides.md): the provenance check now nominates
a candidate and Jev decides whether the user asked for that destination.

## Context

Detection of indirect prompt injection is never complete, and exfiltration through tools is the costliest outcome.

## Decision

Hold any egress or state-changing tool call whose destination appears only in untrusted data and never in operator or user text.

## Consequences

A deterministic rule that survives adversarial evasion. Legitimate agents that send to addresses found only in documents need an allowlist or user confirmation.
