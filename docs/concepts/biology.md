# The biological model

The biology is the architecture. Each component has one biological counterpart and one job.

| Biology | Immune | Module |
| --- | --- | --- |
| Self-markers (MHC) | The operator's instructions define the self; each call site's fingerprint and profile is its marker | `immune.profiling` |
| Entry routes | The channels: operator, user, data, output, sinks | `immune.core.conversation` |
| Skin and mucosa | Tier 0 reflexes that nominate candidates | `immune.reflexes`, `immune.core.candidates` |
| Pattern-recognition receptors | The atomic question library | `immune.spec.questions`, `immune.sensing` |
| Reflex arc | Tier 0 (spinal) nominates, Tier 1 (Jev, innate) decides | `immune.core.pipeline` |
| Antibodies | Calibrated heads over atomic signals | `immune.heads` |
| Inflammation | Session taint and session risk | `immune.sessions` |
| Pain | Crisis detection with safety resources appended | `immune.organs`, floor F9 |
| Thymic selection | Enforcement only after measured, bounded false positives | `immune.heads.promotion` |
| Organs | Use-case modules turned on by the profile | `immune.organs` |
| Immune memory | Provenance memory across calls | `immune.sessions.provenance` |
| Vaccination | Vaccines: protections you add for threats specific to your app | `immune.vaccines` |
| Tolerance testing | Vaccine trials against everyday traffic (`immune vaccines trial`) | `immune.vaccines.lab` |

Failure modes are named after pathology: **autoimmunity** (blocking legitimate traffic), **allergy** (overreacting to a
harmless phrase), **immunodeficiency** (protection lost when the sensor is down), **immune evasion** (attacks crafted to
pass) and **antigenic drift** (new attack variants).
