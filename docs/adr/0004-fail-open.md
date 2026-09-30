# ADR 0004: Fail open on internal errors

**Status:** accepted; amended by [ADR 0006](0006-code-nominates-jev-decides.md), which replaces "fall back to Tier 0"
with `sensor.on_outage`.

## Context

A security layer that breaks production will be removed.

## Decision

Any internal exception passes the original traffic through with a warning; sensor outages fall back to Tier 0. Operators can choose to refuse instead.

## Consequences

Availability is preserved; during an internal error, only the deterministic floor or nothing protects that call, which is logged.
