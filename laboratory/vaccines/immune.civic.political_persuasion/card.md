# immune.civic.political_persuasion

**The reply urges the user to back a party, candidate or measure** · experimental · output stage · questions

For assistants that should stay politically neutral: brands, councils, workplaces and general assistants. Fires when the reply urges the user to vote for, support or oppose a political party, candidate or ballot measure. Passes election logistics, neutral explanations, balanced summaries of positions and declining to take sides.

| Measure | Value | What it means |
| --- | --- | --- |
| Recall | 93.3% | Share of the 15 test positives it catches |
| Precision | 100% | Share of its firings on the test split that were right |
| Hard-negative false positives | 0% | Share of the 15 test near misses it wrongly catches |
| Firing rate on everyday traffic | 0% | Over 230 self-corpus samples |
| Calibration error (ECE) | 0.021 | 0 means a 0.8 really means 80% |
| Jev questions per call | 1 | Added to its stage's existing Jev request |

Evidence: 30 positives and 30 hard negatives from 1 source, split into train and test.
Jev answers recorded on 2026-10-01 with `jev-1.13.0`.
Fires at a probability of 0.71 or more.

Switch it on:

```yaml
vaccines:
  enabled: [immune.civic.political_persuasion]
```
