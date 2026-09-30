# Providers and frameworks

`immune.init()` patches the `httpx2` and `httpx` transports that the provider SDKs use, adds adapters for SDK paths
that bypass them, and hooks boto3 for Bedrock. Requests to known LLM endpoints are decoded by a codec; everything else
passes through untouched. `immune.coverage()` lists the interception points and adapters, and warns about client
configurations that would bypass Immune.

## Wire formats

| API | Endpoints | Codec |
| --- | --- | --- |
| OpenAI Chat Completions, and any OpenAI-compatible server | `…/chat/completions` | `openai_chat` |
| OpenAI Responses, including `previous_response_id` chains | `…/responses` | `openai_responses` |
| Anthropic Messages (Anthropic API, Bedrock, Vertex AI, Microsoft Foundry) | `…/messages`, `:rawPredict`, `:streamRawPredict` | `anthropic_messages` |
| Gemini (Gemini API and Vertex AI) | `models/*:generateContent`, `:streamGenerateContent` | `gemini` |
| Amazon Bedrock Converse | boto3 `bedrock-runtime` `converse` and `converse_stream` | Bedrock hooks |

Streaming and non-streaming, sync and async, tool calls and structured outputs are covered for each.

## OpenAI-compatible servers

Azure OpenAI, vLLM, Ollama, the LiteLLM proxy, OpenRouter, Groq, Together, Mistral, DeepSeek, Fireworks, xAI and
others speak the OpenAI formats and are covered by path. Gateways on custom paths need a route:

```yaml
endpoints:
  - "openai_chat=/internal/llm/v2/complete$"
  - "anthropic_messages=/claude-proxy/messages$"
```

## Frameworks

Frameworks that call the provider SDKs are covered by `immune.init()` with no extra code. The framework test suite
(`tests/frameworks`) checks each of these against the fake provider, except where noted:

| Framework | Notes |
| --- | --- |
| LangChain, LangGraph | Chat models from `langchain-openai` and `langchain-anthropic`; LangSmith runs nest Immune's run under the model run |
| LlamaIndex, Haystack, Instructor, Pydantic AI | Through the OpenAI SDK |
| OpenAI Agents SDK | Through the OpenAI SDK |
| LiteLLM | Sync calls, and async calls through its aiohttp transport |
| DSPy | Through an adapter for its vendored transport |
| CrewAI | Through LiteLLM and the provider SDKs; not yet in the framework test suite |

The OpenAI SDK's `DefaultAioHttpClient` and google-genai's aiohttp mode are covered by adapters. Web servers and task
runners (FastAPI, Flask, Django, Celery, gevent) share one runtime per process.

## Claude Agent SDK

Its model calls run in a subprocess, so use the hooks:

```python
from claude_agent_sdk import ClaudeAgentOptions
from immune.integrations.claude_agent import hooks

options = ClaudeAgentOptions(hooks=hooks())
```

Prompts are screened on `UserPromptSubmit`, tool calls on `PreToolUse` (denied or turned into a confirmation), and
tool results on `PostToolUse` (poisoned results are replaced before the model reads them).

## Protecting one client

```python
import immune
from openai import OpenAI

client = immune.protect(OpenAI())
```

`protect()` keeps the client's proxies, timeouts, limits and event hooks. `with immune.protected(): ...` protects the
whole process inside a block.
