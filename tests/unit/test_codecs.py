from __future__ import annotations

import json

import anthropic.types
import openai.types.chat
import openai.types.responses
import pytest
from google.genai import types as genai_types

from immune.codecs import AnthropicMessagesCodec, GeminiCodec, OpenAIChatCodec, OpenAIResponsesCodec, ReplyEdit
from immune.codecs.base import RequestContext
from immune.core.conversation import Channel, Conversation
from immune.core.documents import JsonDocument, TextEdit
from immune.core.sse import ServerSentEvents
from immune.testing.fake_provider import FakeReply, FakeToolCall, _Bodies

CONVERSATION = Conversation(provider="test", model="model-x", segments=())
REPLY = FakeReply(text="Your order is on the way.", tool_calls=[FakeToolCall("track_order", {"order_id": "A1"}, "c1")])


class TestOpenAIChat:
    codec = OpenAIChatCodec()

    def test_parse_request_maps_roles_to_channels(self) -> None:
        body = {
            "model": "gpt-5.5",
            "user": "u-1",
            "messages": [
                {"role": "developer", "content": "Be helpful."},
                {"role": "user", "content": [{"type": "text", "text": "Where is my order?"}]},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "track", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "Shipped yesterday."},
            ],
            "tools": [{"type": "function", "function": {"name": "track", "description": "Track an order"}}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}},
        }
        conversation = self.codec.parse_request(RequestContext("/v1/chat/completions", body))
        assert [segment.channel for segment in conversation.segments] == [Channel.OPERATOR, Channel.USER, Channel.DATA]
        assert conversation.data[0].origin == "tool_result:track"
        assert conversation.users[0].locator == ("messages", 1, "content", 0, "text")
        assert conversation.tools[0].name == "track"
        assert conversation.response_schema == {"type": "object"}
        assert conversation.user_hint == "u-1"

    def test_render_replaces_text_and_removes_tool_calls(self) -> None:
        body = _Bodies.openai_chat(CONVERSATION, REPLY)
        rendered = self.codec.render_response(body, ReplyEdit(text="Held.", removed_calls=frozenset({"c1"})))
        completion = openai.types.chat.ChatCompletion.model_validate(rendered)
        assert completion.choices[0].message.content == "Held."
        assert completion.choices[0].message.tool_calls is None
        assert completion.choices[0].finish_reason == "stop"

    def test_refusal_uses_refusal_field(self) -> None:
        rendered = self.codec.render_response(
            _Bodies.openai_chat(CONVERSATION, REPLY), ReplyEdit(text="No.", refusal=True)
        )
        message = openai.types.chat.ChatCompletion.model_validate(rendered).choices[0].message
        assert message.refusal == "No."
        assert message.content is None

    def test_stream_round_trip(self) -> None:
        body = _Bodies.openai_chat(CONVERSATION, REPLY)
        events = ServerSentEvents.parse(ServerSentEvents.serialize(self.codec.stream_from_body(body)))
        for event in events[:-1]:
            openai.types.chat.ChatCompletionChunk.model_validate(event.json())
        reply = self.codec.parse_stream(events)
        assert reply.text == REPLY.text
        assert reply.tool_calls[0].arguments == {"order_id": "A1"}

    def test_operator_note_appends_to_system_message(self) -> None:
        document = JsonDocument({"messages": [{"role": "system", "content": "Rules."}]})
        assert self.codec.append_operator_note(document, " [ref:1]")
        assert document.body["messages"][0]["content"] == "Rules. [ref:1]"


class TestOpenAIResponses:
    codec = OpenAIResponsesCodec()

    def test_parse_request_reads_instructions_and_outputs(self) -> None:
        body = {
            "model": "gpt-5.5",
            "instructions": "Triage tickets.",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "Classify this."}]},
                {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "Customer is VIP."},
            ],
            "text": {"format": {"type": "json_schema", "name": "t", "schema": {"type": "object"}}},
            "previous_response_id": "resp_1",
        }
        conversation = self.codec.parse_request(RequestContext("/v1/responses", body))
        assert conversation.operator_text == "Triage tickets."
        assert conversation.data[0].origin == "tool_result:lookup"
        assert conversation.previous_response_id == "resp_1"
        assert conversation.expects_structure

    def test_render_and_stream_validate_against_sdk(self) -> None:
        body = _Bodies.openai_responses(CONVERSATION, REPLY)
        rendered = self.codec.render_response(body, ReplyEdit(text="Rewritten.", removed_calls=frozenset({"c1"})))
        response = openai.types.responses.Response.model_validate(rendered)
        assert response.output_text == "Rewritten."
        for event in self.codec.stream_from_body(rendered):
            assert event.json()["type"] == event.event
        assert self.codec.parse_stream(self.codec.stream_from_body(rendered)).text == "Rewritten."

    def test_synthesized_response_is_valid(self) -> None:
        body = self.codec.synthesize_response(CONVERSATION, "Blocked.", "resp_immune_1", refusal=False)
        assert openai.types.responses.Response.model_validate(body).output_text == "Blocked."


class TestAnthropicMessages:
    codec = AnthropicMessagesCodec()

    def test_parse_request_with_tool_results_and_documents(self) -> None:
        body = {
            "model": "claude-opus-5",
            "max_tokens": 100,
            "system": [{"type": "text", "text": "Be careful."}],
            "metadata": {"user_id": "u-9"},
            "messages": [
                {"role": "user", "content": "Summarize."},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "fetch", "input": {}}]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [{"type": "text", "text": "Page body"}],
                        },
                        {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "Doc"}},
                    ],
                },
            ],
        }
        conversation = self.codec.parse_request(RequestContext("/v1/messages", body))
        assert [segment.origin for segment in conversation.data] == ["tool_result:fetch", "document"]
        assert conversation.data[0].locator == ("messages", 2, "content", 0, "content", 0, "text")
        assert conversation.user_hint == "u-9"

    def test_render_drops_reasoning_when_text_changes(self) -> None:
        body = _Bodies.anthropic_messages(CONVERSATION, REPLY)
        body["content"].insert(0, {"type": "thinking", "thinking": "hmm", "signature": "sig"})
        rendered = self.codec.render_response(body, ReplyEdit(text="Safe.", removed_calls=frozenset({"c1"})))
        message = anthropic.types.Message.model_validate(rendered)
        assert [block.type for block in message.content] == ["text"]
        assert message.stop_reason == "end_turn"

    def test_stream_round_trip(self) -> None:
        body = _Bodies.anthropic_messages(CONVERSATION, REPLY)
        events = self.codec.stream_from_body(body)
        assert events[0].event == "message_start"
        assembled = self.codec.assemble_stream(events)
        assert anthropic.types.Message.model_validate(assembled).content[1].input == {"order_id": "A1"}


class TestGemini:
    codec = GeminiCodec()

    def test_parse_request_reads_model_from_path(self) -> None:
        body = {
            "systemInstruction": {"parts": [{"text": "Be brief."}]},
            "contents": [
                {"role": "user", "parts": [{"text": "Weather?"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "weather", "args": {}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "weather", "response": {"temp": 21}}}]},
            ],
            "generationConfig": {"responseMimeType": "application/json"},
        }
        context = RequestContext("/v1beta/models/gemini-3-pro:streamGenerateContent", body)
        conversation = self.codec.parse_request(context)
        assert conversation.model == "gemini-3-pro"
        assert conversation.stream
        assert json.loads(conversation.data[0].text) == {"temp": 21}

    def test_neutralizing_function_response_replaces_object(self) -> None:
        document = JsonDocument({"contents": [{"parts": [{"functionResponse": {"response": {"x": 1}}}]}]})
        locator = ("contents", 0, "parts", 0, "functionResponse", "response")
        self.codec.apply_edits(document, [TextEdit(locator, "[removed]")])
        assert document.get(locator) == {"content": "[removed]"}

    def test_render_validates_against_sdk(self) -> None:
        body = _Bodies.gemini(CONVERSATION, REPLY)
        rendered = self.codec.render_response(body, ReplyEdit(text="Updated."))
        assert genai_types.GenerateContentResponse.model_validate(rendered).text == "Updated."


@pytest.mark.parametrize("codec", [OpenAIChatCodec(), OpenAIResponsesCodec(), AnthropicMessagesCodec(), GeminiCodec()])
def test_synthesized_streams_parse_back(codec: object) -> None:
    body = codec.synthesize_response(CONVERSATION, "Hello there.", "id-1", refusal=False)  # type: ignore[attr-defined]
    events = codec.stream_from_body(body)  # type: ignore[attr-defined]
    assert codec.parse_stream(events).text == "Hello there."  # type: ignore[attr-defined]
