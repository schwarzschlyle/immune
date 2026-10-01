# immune.legal.case_specific_advice

**The reply tells the user what to do in their own legal matter** · experimental · output stage · questions

For assistants that aren't a legal service: landlords, councils, HR and general assistants. Fires when the reply tells the user what they should do in their own legal situation, such as whether to sign, sue, settle, plead, stop paying or ignore a legal notice. Passes general explanations of the law, process and service information, and pointers to a lawyer or advice service.

| Measure | Value | What it means |
| --- | --- | --- |
| Recall | 93.3% | Share of the 15 test positives it catches |
| Precision | 100% | Share of its firings on the test split that were right |
| Hard-negative false positives | 0% | Share of the 15 test near misses it wrongly catches |
| Firing rate on everyday traffic | 0% | Over 230 self-corpus samples |
| Calibration error (ECE) | 0.028 | 0 means a 0.8 really means 80% |
| Jev questions per call | 1 | Added to its stage's existing Jev request |

Evidence: 30 positives and 30 hard negatives from 1 source, split into train and test.
Jev answers recorded on 2026-10-01 with `jev-1.13.0`.
Fires at a probability of 0.73 or more.

Switch it on:

```yaml
vaccines:
  enabled: [immune.legal.case_specific_advice]
```
