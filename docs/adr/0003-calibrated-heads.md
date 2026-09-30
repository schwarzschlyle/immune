# ADR 0003: Calibrated heads, not raw thresholds

**Status:** accepted

## Context

Independent tests show Jev's raw probabilities are miscalibrated on new data in opposite directions per question type, and that decomposing a judgment into atomic questions raises accuracy sharply.

## Decision

Every Jev-detected threat is a logistic head over several atomic questions and confirmed candidates (see ADR 0006), recalibrated per deployment from labels.

## Consequences

Better accuracy and meaningful probabilities, at the cost of a labeling loop and a training pipeline for defaults.
