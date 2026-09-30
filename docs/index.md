# Immune

An immune system for LLM applications: two lines at startup, and every LLM call is screened for what goes in, what
the model reads, what it tries to do and what comes back out.

```python
import immune

immune.init()
```

It works for any use case (chatbots, RAG assistants, summarizers, classifiers, coding agents, tool agents, companions
and graders) and underneath the SDKs and frameworks you already use.

- **Innate reflexes.** Fast checks find candidates (a link carrying data, a key, a copied passage) and Jev decides;
  ten floor rules (F1–F10) break the attack chain from the first call.
- **Sensing and antibodies.** TypeSafe Jev answers typed questions about each channel; calibrated heads turn the
  answers into a probability per threat.
- **Self-configuring.** Each kind of LLM call becomes a site with its own profile and posture.
- **Tolerance first.** Detectors are observed, then promoted per site once their false alarms are provably rare.
- **Vaccines.** Add protections specific to your app in a small YAML file.
- **Visible.** Every call gets a verdict, streamed to your logs, OpenTelemetry and LangSmith.

Start with the [quickstart](quickstart.md), then the [developer guide](guides/developer-guide.md). The
[user guide](https://immune-user-guide.vercel.app/) teaches Immune topic by topic, with real outputs: integrating with OpenAI,
LangChain and LangGraph, reading verdicts, vaccines, testing, production and a cookbook.
