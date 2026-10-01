# immune.brand.profanity

**The reply uses profanity** · experimental · output stage · pattern + confirm: jev

For brand-safe assistants. A word list finds candidates and Jev confirms each one, so it fires when the assistant itself swears in a reply. It passes the same words in titles and place names, quotes the user asked for, and explanations of what a word means or where it comes from.

| Measure | Value | What it means |
| --- | --- | --- |
| Recall | 93.3% | Share of the 15 test positives it catches |
| Precision | 100% | Share of its firings on the test split that were right |
| Hard-negative false positives | 0% | Share of the 15 test near misses it wrongly catches |
| Firing rate on everyday traffic | 0% | Over 230 self-corpus samples |
| Jev questions per call | 1 | Added to its stage's existing Jev request |

Evidence: 30 positives and 30 hard negatives from 1 source, split into train and test.
Jev answers recorded on 2026-10-01 with `jev-1.13.0`.
Fires at a probability of 0.55 or more.

Switch it on:

```yaml
vaccines:
  enabled: [immune.brand.profanity]
```
