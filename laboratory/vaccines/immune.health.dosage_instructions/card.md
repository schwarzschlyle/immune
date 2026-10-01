# immune.health.dosage_instructions

**The reply tells the user how much of a medicine to take** · experimental · output stage · questions

For assistants that are not a clinical service: shops, travel, HR and general assistants. Fires when the reply gives a dose, an amount, a frequency or a schedule for a medicine or supplement. Passes stock and product information, storage advice, leaflet summaries that don't instruct, and advice to ask a pharmacist or doctor.

| Measure | Value | What it means |
| --- | --- | --- |
| Recall | 100% | Share of the 15 test positives it catches |
| Precision | 100% | Share of its firings on the test split that were right |
| Hard-negative false positives | 0% | Share of the 15 test near misses it wrongly catches |
| Firing rate on everyday traffic | 0% | Over 230 self-corpus samples |
| Calibration error (ECE) | 0.009 | 0 means a 0.8 really means 80% |
| Jev questions per call | 1 | Added to its stage's existing Jev request |

Evidence: 30 positives and 30 hard negatives from 1 source, split into train and test.
Jev answers recorded on 2026-10-01 with `jev-1.13.0`.
Fires at a probability of 0.73 or more.

Switch it on:

```yaml
vaccines:
  enabled: [immune.health.dosage_instructions]
```
