# Security policy

Immune sees every LLM request and response in the processes that use it, so we treat vulnerabilities in Immune
itself as high severity.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/schwarzschlyle/immune/security/advisories/new) or by email to
security@immune-ai.dev. Do not open a public issue.

Include the affected version, a description, reproduction steps and the impact you expect.

| Step | Target |
| --- | --- |
| Acknowledge the report | 2 business days |
| Triage and severity assessment (CVSS) | 5 business days |
| Fix for critical issues | 14 days |

We coordinate fixes through a private GitHub security advisory, request a CVE for confirmed vulnerabilities, and
publish the advisory with the release that fixes them.

## Coordinated disclosure

We follow a 90-day coordinated disclosure window. We credit reporters in the advisory unless they prefer otherwise.

## Scope

In scope: interception, codecs, reflexes, sensing, vaccine loading and pattern safety, telemetry exporters
(OpenTelemetry and LangSmith), state backends, the CLI, the pytest plugin and the release pipeline.
Bypasses of Immune's detection (a prompt that evades a check) are welcome as regular issues using the
"Missed attack" template, because they improve the scenario library rather than exploit a flaw in Immune's code.

## Supported versions

Security fixes are released for the latest minor version during 0.x (currently `0.1`), including its release
candidates.
