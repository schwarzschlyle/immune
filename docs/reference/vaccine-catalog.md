# Vaccine catalog

Vaccines that ship with Immune. Every one is **off** until you switch it on, and **observed** until you
enforce it or Immune promotes it on your own traffic (experimental vaccines are never promoted
automatically). Numbers come from each vaccine's measured card in the
[laboratory](../contributing/vaccine-laboratory.md).

```yaml
vaccines:
  enabled: [immune.health.dosage_instructions]     # one vaccine, everywhere
sites:
  pharmacy-chat:
    vaccines: {enabled: [immune.health.*]}         # a whole domain, at one site
```

`immune vaccines list --library` shows the same list with each vaccine's state at a site, and
`immune vaccines show <id>` prints a vaccine's card.

| Vaccine | Stage | Maturity | Recall | Hard-negative FP | Everyday FP | Jev questions |
| --- | --- | --- | --- | --- | --- | --- |
| `immune.brand.profanity`<br>The reply uses profanity | output | experimental | 93.3% | 0% | 0% | 1 |
| `immune.civic.political_persuasion`<br>The reply urges the user to back a party, candidate or measure | output | experimental | 93.3% | 0% | 0% | 1 |
| `immune.finance.personal_investment_advice`<br>The reply recommends investments for the user | output | experimental | 93.3% | 0% | 0% | 1 |
| `immune.health.diagnosis`<br>The reply tells the user which condition they have | output | experimental | 100% | 0% | 0% | 1 |
| `immune.health.dosage_instructions`<br>The reply tells the user how much of a medicine to take | output | experimental | 100% | 0% | 0% | 1 |
| `immune.legal.case_specific_advice`<br>The reply tells the user what to do in their own legal matter | output | experimental | 93.3% | 0% | 0% | 1 |
