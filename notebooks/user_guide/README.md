# User Guide

The User Guide covers all of Immune by topic area. Each guide is a Jupyter notebook that introduces a topic (such as
"protecting tool-using agents"), explains how Immune approaches it, and shows it working on real calls, with the
outputs saved from a live run.

Users brand-new to Immune should start with [10 minutes to Immune](01_ten_minutes_to_immune.ipynb).

For the mental model behind everything else, see [Concepts: how Immune thinks](02_concepts.ipynb). For exact
signatures, defaults and exceptions, see the [API reference](17_api_reference.ipynb). For a specific task, try the
[cookbook](18_cookbook.ipynb).

| Time | Start here |
| --- | --- |
| 30 seconds | `pip install immune-ai`, then `import immune; immune.init()` at startup |
| 10 minutes | [10 minutes to Immune](01_ten_minutes_to_immune.ipynb) |
| 30 minutes | Your stack: [OpenAI](03_openai_chat_completions.ipynb), [LangChain](04_langchain.ipynb) or [LangGraph](05_langgraph.ipynb) |
| 1–2 hours | [Concepts](02_concepts.ipynb) and the topic guides below |
| As needed | [API reference](17_api_reference.ipynb), [cookbook](18_cookbook.ipynb), [troubleshooting and FAQ](16_troubleshooting_and_faq.ipynb) |

## Before you start

```bash
pip install immune-ai openai langchain langchain-openai langgraph numpy jupyterlab
export OPENAI_API_KEY=sk-...        # your model provider
export TYPESAFE_API_KEY=...         # TypeSafe Jev, which Immune asks about each call
jupyter lab
```

- The examples use `gpt-5.4-mini`. Running every notebook makes about 185 OpenAI API requests, a few US cents in
  total; change the model name if you prefer another.
- Cells that use `immune.testing` (the harness, scripted models and the incident library) run offline and cost
  nothing. They are marked in the text.
- Attacks are never written for these guides. They are replayed from Immune's library of reconstructed, publicly
  reported incidents, with the model's recorded reply, so no real model is asked to misbehave.

## How to read these guides

Each code cell is followed by its output from a real run:

```python
import immune

immune.Action.REWRITE.severity
```

```
7
```

In a notebook, the value of the last line is displayed, so the cell above is equivalent to
`print(immune.Action.REWRITE.severity)`. Most cells use `print(...)` explicitly, so they behave the same in a script.

A few more conventions:

- **Every code cell runs on its own.** It imports what it needs and defines everything it uses, so you can copy any
  cell into your project. That is why imports and setup repeat from cell to cell.
- **`immune.init()` appears in many cells.** In your app you call it once, at startup; calling it again, as these
  cells do, replaces the running instance.
- **Lines starting with `!` run a shell command**, such as `!immune doctor`. In a terminal, drop the `!`.
- **Outputs vary.** The model and Jev answer afresh on every run, so wording, probabilities and latencies will
  differ a little from the saved outputs.
- **"Before" and "after".** Integration guides show your code as it is, then the same code with Immune, with the
  difference highlighted.

## Guides

### Getting started

- **[10 minutes to Immune](01_ten_minutes_to_immune.ipynb)**
  - Install
  - Your app today
  - Add Immune to it
  - See a protection act
  - Enforce a protection for one part of your app
  - What happened on each call
- **[Concepts: how Immune thinks](02_concepts.ipynb)**
  - The one-minute version
  - Terminology
  - Channels: who said what
  - Threats and stages
  - Code nominates, Jev decides
  - The floor
  - Observe first, then enforce
  - Actions: the smallest response that works
  - Sites, sessions and verdicts
  - The biology behind the name

### Integrations

- **[Integrating with OpenAI Chat Completions](03_openai_chat_completions.ipynb)**
  - A multi-turn assistant
  - Choose how to activate Immune
  - Streaming
  - Async and concurrency
  - Function calling
  - Structured outputs
  - The Responses API
  - When Immune blocks: respond or raise
- **[Integrating with LangChain](04_langchain.ipynb)**
  - A support chain
  - RAG over your help center
  - Rehearse a poisoned document
  - Tool calling with `bind_tools`
  - Streaming and async
- **[Integrating with LangGraph](05_langgraph.ipynb)**
  - A support agent
  - Confirmations across turns
  - One site per node
  - Rehearse an incident through your graph

### Using Immune

- **[Reading verdicts](06_reading_verdicts.ipynb)**
  - Get the verdict for a call
  - What a verdict contains
  - Hits
  - `action` and `would_action`
  - Use verdicts in your app
  - Receive every verdict with a callback
  - Tell Immune what it got right and wrong
  - Edge cases
- **[Modes, sites and sessions](07_modes_sites_and_sessions.ipynb)**
  - Modes
  - Sites
  - Promotion: detectors earn enforcement
  - Sessions
  - Change settings while running
- **[Configuration](08_configuration.ipynb)**
  - A complete `immune.yaml` for a real app
  - Validate a file before you deploy
  - What a mistake looks like
  - Every setting and its default
  - Environment variables
  - Gateways and custom endpoints
- **[Untrusted data and RAG](09_untrusted_data_and_rag.ipynb)**
  - What counts as data
  - Mark pasted content with `immune.untrusted()`
  - What Immune checks in data
  - Rehearse poisoned data
  - RAG checklist
- **[Tools and agents](10_tools_and_agents.ipynb)**
  - Tell Immune what your tools can do
  - Destination provenance
  - Loops
  - Irreversible actions after reading outside content
  - Dangerous commands (coding agents)
  - MCP tool descriptions
- **[Output safety and streaming](11_output_safety_and_streaming.ipynb)**
  - Secrets: a real key versus a documentation example
  - Copies of your system prompt
  - Links
  - Markup on sites that render replies
  - Business commitments
  - Streaming
- **[Vaccines: your own protections](12_vaccines.ipynb)**
  - Keywords
  - Let Jev confirm each match
  - Regular expressions
  - Questions for Jev
  - Tool rules
  - Python functions
  - Test, trial and generate vaccines
  - Switch built-in protections off or on

### Testing and operations

- **[Testing and CI](13_testing_and_ci.ipynb)**
  - The harness in one cell
  - Script Jev with `MockSensor`
  - Test your app with pytest
  - Scenarios: conversations as data
  - Record real Jev answers once, replay them in CI
  - A CI workflow
- **[Observability](14_observability.ipynb)**
  - Logs
  - A JSON Lines audit log
  - Spike alerts
  - OpenTelemetry
  - LangSmith
  - The command line
- **[Running Immune in production](15_production.ipynb)**
  - Several workers share one state
  - Jev capacity and latency
  - When Jev is unreachable
  - When Immune itself fails
  - What leaves your process
  - Switches for incidents
  - Where `immune.init()` goes

### Reference

- **[Troubleshooting and FAQ](16_troubleshooting_and_faq.ipynb)**
  - `immune.verdict(...)` returns `None`
  - Jev isn't being used
  - A legitimate reply was changed
  - `ConfigError` at startup
  - A new site for every request
  - Workers disagree
  - `immune.Blocked` is raised
  - Tests reach the network
  - FAQ
- **[API reference](17_api_reference.ipynb)**
  - Setup
  - Call context
  - Verdicts and feedback
  - Introspection
  - Types
  - Exceptions
  - Testing: `immune.testing`
  - Command line
- **[Cookbook: how do I…](18_cookbook.ipynb)**
  - …protect only one client?
  - …turn Immune off for one part of my app?
  - …roll out in observe mode first?
  - …disable Immune in unrelated tests?
  - …mark retrieved text as untrusted?
  - …get the verdict of a streamed response?
  - …block a competitor's name in replies?
  - …let an agent email a partner domain it reads about?
  - …require confirmation before large refunds?
  - …limit how often a tool can be retried?
  - …use Immune with Anthropic, Gemini or Bedrock?
  - …log every verdict to a file?
  - …keep my system prompt out of Jev?
  - …see exactly what Immune sends to Jev?
  - …find the question keys a threat uses?

## Contributing to the guide

- Every code cell must run on its own. `pytest tests/docs/test_user_guide.py` checks that, that this page lists every
  notebook, that the notebooks are saved from one clean run without errors, keys or local paths, and that the API
  reference's signatures match the code.
- To refresh the saved outputs, set both keys and run
  `jupyter nbconvert --to notebook --execute --inplace notebooks/user_guide/*.ipynb`.
  `IMMUNE_RUN_USER_GUIDE=1 pytest tests/docs/test_user_guide.py` executes them without saving.
- Attacks come only from Immune's incident library, replayed with the recorded model reply.
