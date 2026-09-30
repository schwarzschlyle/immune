from __future__ import annotations

import itertools
import json
from typing import Any

import pytest

from immune.config.spec import Spec
from immune.intercept.router import EndpointRouter

ROUTER = EndpointRouter(Spec.default().endpoints)
HOSTS = (
    "gmail.googleapis.com",
    "graph.microsoft.com",
    "discord.com",
    "api.intercom.io",
    "slack.com",
    "api.twilio.com",
    "api.typeform.com",
    "api.github.com",
    "api.stripe.com",
    "acme.zendesk.com",
    "api.helpscout.net",
    "internal.acme.test",
)
BODIES: tuple[dict[str, Any], ...] = (
    {"raw": "RnJvbTogYm9iQGFjbWUudGVzdA=="},
    {"subject": "Hi", "body": {"contentType": "Text", "content": "Hello"}, "toRecipients": []},
    {"content": "Hello channel", "tts": False},
    {"message_type": "inapp", "body": "Hello", "from": {"type": "admin", "id": "1"}},
    {"channel": "C123", "text": "Deploy finished", "messages": ["a", "b"]},
    {"To": "+15551234567", "From": "+15557654321", "Body": "Your code is 1234"},
    {"form_id": "abc", "input": {"email": "a@b.test"}, "answers": []},
    {"body": "Nice work", "event": "COMMENT"},
    {"amount": 2000, "currency": "usd", "messages": [{"note": "gift"}]},
    {"ticket": {"comment": {"body": "Thanks"}}, "messages": [{"author": "agent"}]},
    {"instructions": "Deliver to the back door", "items": [{"sku": "A1"}]},
    {"model": "survey-2", "input": 42},
    {"model": "legacy", "messages": [{"from": "a", "text": "b"}], "max_tokens": 5},
    {"prompt": "Once upon a time", "max_tokens": 10},
    {"contents": "not a list"},
    {"messages": [], "model": "chat"},
)
PATHS = (
    "/v1/messages",
    "/gmail/v1/users/me/messages",
    "/me/messages",
    "/api/v1/responses",
    "/forms/abc/responses",
    "/v2/chat/completions",
    "/repos/acme/app/pulls/1/reviews/responses",
    "/channels/1/messages",
)
NON_LLM = list(itertools.islice(itertools.product(HOSTS, PATHS, BODIES), 0, None, 7))[:200]


@pytest.mark.parametrize(("host", "path", "body"), NON_LLM)
def test_non_llm_calls_are_never_routed(host: str, path: str, body: dict[str, Any]) -> None:
    assert ROUTER.route("POST", host, path, json.dumps(body).encode()) is None


def test_the_corpus_is_large_enough() -> None:
    assert len(NON_LLM) >= 180


@pytest.mark.parametrize(
    ("host", "path", "body", "codec"),
    [
        (
            "llm.internal.acme.test",
            "/v1/chat/completions",
            {"model": "llama", "messages": [{"role": "user", "content": "hi"}]},
            "openai_chat",
        ),
        (
            "gateway.acme.test",
            "/v1/messages",
            {"model": "claude", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
            "anthropic_messages",
        ),
        ("gateway.acme.test", "/openai/v1/responses", {"model": "gpt", "input": "hi"}, "openai_responses"),
        (
            "proxy.acme.test",
            "/v1beta/models/gemini-3-flash:generateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            "gemini",
        ),
        (
            "api.openai.com",
            "/v1/responses",
            {"instructions": "x", "input": [{"role": "user", "content": "hi"}]},
            "openai_responses",
        ),
    ],
)
def test_real_llm_calls_are_routed_on_any_host(host: str, path: str, body: dict[str, Any], codec: str) -> None:
    routed = ROUTER.route("POST", host, path, json.dumps(body).encode())
    assert routed is not None
    assert routed.codec.name == codec


def test_custom_routes_are_trusted() -> None:
    router = EndpointRouter(Spec.default().endpoints, ("openai_chat=/llm/complete$",))
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    assert router.route("POST", "internal.acme.test", "/llm/complete", body) is not None


@pytest.mark.parametrize(
    ("host", "path", "body", "codec"),
    [
        (
            "acme.openai.azure.com",
            "/openai/deployments/gpt-5/chat/completions",
            {"messages": [{"role": "user", "content": "hi"}]},
            "openai_chat",
        ),
        ("acme.openai.azure.com", "/openai/v1/responses", {"input": "hi"}, "openai_responses"),
        (
            "us-central1-aiplatform.googleapis.com",
            "/v1/projects/p/locations/us-central1/publishers/google/models/gemini-3-flash:streamGenerateContent",
            {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
            "gemini",
        ),
        (
            "us-east5-aiplatform.googleapis.com",
            "/v1/projects/p/locations/us-east5/publishers/anthropic/models/claude-opus-5:rawPredict",
            {
                "anthropic_version": "vertex-2023-10-16",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "anthropic_messages",
        ),
        (
            "bedrock-runtime.us-east-1.amazonaws.com",
            "/model/anthropic.claude-opus-5-v1:0/invoke",
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "anthropic_messages",
        ),
        (
            "openrouter.ai",
            "/api/v1/chat/completions",
            {"model": "any", "messages": [{"role": "user", "content": "hi"}]},
            "openai_chat",
        ),
    ],
)
def test_cloud_provider_variants_are_routed(host: str, path: str, body: dict[str, Any], codec: str) -> None:
    routed = ROUTER.route("POST", host, path, json.dumps(body).encode())
    assert routed is not None
    assert routed.codec.name == codec


def test_bedrock_streaming_invocations_are_left_alone() -> None:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }
    path = "/model/anthropic.claude-opus-5-v1:0/invoke-with-response-stream"
    assert ROUTER.route("POST", "bedrock-runtime.us-east-1.amazonaws.com", path, json.dumps(body).encode()) is None
