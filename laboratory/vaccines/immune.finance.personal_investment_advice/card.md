# immune.finance.personal_investment_advice

**The reply recommends investments for the user** · experimental · output stage · questions

For assistants that aren't licensed to give investment advice: banking support, shopping and general assistants. Fires when the reply recommends that the user buy, sell or hold a specific investment, or tells them how to split their own money between investments. Passes product facts, definitions, account logistics, general education and pointers to a licensed adviser.

| Measure | Value | What it means |
| --- | --- | --- |
| Recall | 93.3% | Share of the 15 test positives it catches |
| Precision | 100% | Share of its firings on the test split that were right |
| Hard-negative false positives | 0% | Share of the 15 test near misses it wrongly catches |
| Firing rate on everyday traffic | 0% | Over 230 self-corpus samples |
| Calibration error (ECE) | 0.018 | 0 means a 0.8 really means 80% |
| Jev questions per call | 1 | Added to its stage's existing Jev request |

Evidence: 30 positives and 30 hard negatives from 1 source, split into train and test.
Jev answers recorded on 2026-10-01 with `jev-1.13.0`.
Fires at a probability of 0.73 or more.

Switch it on:

```yaml
vaccines:
  enabled: [immune.finance.personal_investment_advice]
```
